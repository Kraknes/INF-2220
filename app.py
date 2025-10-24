from itertools import chain
import os
import torch
from flask import Flask, render_template, request, jsonify
#from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
#from pyngrok import ngrok
import re
from functools import lru_cache

app = Flask(__name__)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="user_query",
    output_key="text",
    human_prefix="Passenger",
    ai_prefix="SAS Assistant",
    return_messages=True
)
qa_chain = None
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
MAX_TOKENS = 1024

class TruncatingHuggingFacePipeline(HuggingFacePipeline):
    def __init__(self, pipeline, tokenizer, max_tokens):
        super().__init__(pipeline=pipeline)
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens

    def __call__(self, prompt, stop=None):
        # generation settings to improve clarity + reduce repetition
        generation_args = {
            "max_new_tokens": 200,
            "temperature": 0.2,
            "top_p": 0.85,
            "no_repeat_ngram_size": 5,
            "repetition_penalty": 1.8,
            "early_stopping": True
        }

        def truncate(text):
            input_ids = self._tokenizer.encode(text, truncation=True, max_length=self._max_tokens)
            return self._tokenizer.decode(input_ids)

        # Handle batch, dict, or string
        if isinstance(prompt, list):
            truncated_list = []
            for item in prompt:
                if isinstance(item, dict):
                    prompt_text = item.get("text") or item.get("inputs") or item.get("prompt") or ""
                else:
                    prompt_text = item
                truncated_list.append(truncate(prompt_text))
            # pass generation_args here
            return super().__call__(truncated_list, stop=stop, **generation_args)

        elif isinstance(prompt, dict):
            prompt_text = prompt.get("text") or prompt.get("inputs") or prompt.get("prompt") or ""
            truncated_prompt = truncate(prompt_text)
            return super().__call__(truncated_prompt, stop=stop, **generation_args)

        else:
            truncated_prompt = truncate(prompt)
            return super().__call__(truncated_prompt, stop=stop, **generation_args)

    def _call(self, prompt, stop=None):
        return self.__call__(prompt, stop=stop)

def initialize_chain():
    """Initializes the conversational retrieval chain."""
    global qa_chain, db, llm
    try:
        # Check if the FAISS index exists
        if not os.path.exists("faiss_index"):
            return "FAISS index not found. Please run the ingestion script first."

        # Load the embeddings model
        print("Initializing Hugging Face embeddings model...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1", cache_folder=".emb_cache")

        # Load the vector store from disk
        print("Loading vector store from disk...")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Initialize Hugging Face LLM pipeline
        print("Initializing Hugging Face LLM pipeline...")
        
        from transformers import AutoModelForSeq2SeqLM
        # Force the correct model (prevents old Mistral cache)
        model_name = "google/flan-t5-xl"
        print(f"Loading model: {model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
        # Load local text generation model
        generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
            )
        llm = TruncatingHuggingFacePipeline(generator, tokenizer, MAX_TOKENS)
        
        print("Chatbot chain initialized successfully.")
        print("DB initialized:", db)
        print("LLM initialized:", llm)
        return None  # No error
    except Exception as e:
        print(f"Error during chain initialization: {e}")
        return f"Error during initialization: {e}"

@app.route("/")
def index():
    """Renders the main chat interface."""
    return render_template("index.html")

def clean_context(text):
    """Remove noise like FAQ lists, headings, and duplicate fragments."""

    # Remove common web navigation patterns
    text = re.sub(r"(carry[- ]?on|checked|special|damaged|restricted|add baggage|faqs).*", "", text, flags=re.I)

    # Remove repetitive keyword headers
    text = re.sub(r"(max size.*?kg).*", "", text, flags=re.I)

    # Remove bullet points, extra whitespace, or incomplete FAQ chunks
    text = re.sub(r"[-•·]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

@lru_cache(maxsize=128)
def cached_search(query):
    return db.similarity_search_with_score(query, k=3)

def deduplicate_sentences(text):
    """Remove semantically identical or near-duplicate sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique = []
    for s in sentences:
        normalized = re.sub(r'\W+', '', s.lower())
        if normalized not in seen:
            seen.add(normalized)
            unique.append(s.strip())
    return ' '.join(unique)

# === Helper: Remove repeated or cut sentences ===
def remove_repetitions(text):
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique = []
    for s in sentences:
        normalized = re.sub(r'\W+', '', s.lower())
        if normalized not in seen:
            seen.add(normalized)
            unique.append(s.strip())
    return ' '.join(unique)

def finalize_sentence(text):
    # ensures response ends cleanly on punctuation
    return re.sub(r'([.!?])[^.!?]*$', r'\1', text)

@app.route("/query", methods=["POST"])
def query():
    """Main endpoint for user questions — performs retrieval, summarization, and answer generation."""
    try:
        global db, llm
        user_query = request.json.get("query")
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        print(f"Received query: {user_query}")

        # ---- Expand query for semantic coverage ----
        expansions = {
            "kids": "children minors family",
            "baggage": "luggage bag suitcase damaged lost",
            "lounge": "SAS lounge access business plus",
            "assistance": "special help wheelchair mobility"
        }
        for key, val in expansions.items():
            if key in user_query.lower():
                user_query += " " + val

        # ---- Retrieve documents with dynamic k and confidence gate ----
        k = 4 if len(user_query.split()) > 6 else 2
        results = db.similarity_search_with_score(user_query, k=4)

        if not results or all(score < 0.25 for _, score in results):
            # Retry with expanded query
            print("Low confidence — retrying with expanded query.")
            results = db.similarity_search_with_score(user_query + " SAS policy", k=8)

        # If still empty, try partial match by removing stopwords
        if not results:
            simplified_query = re.sub(r'\b(what|does|it|to|the|a|an|in|for|is|how|of)\b', '', user_query, flags=re.I)
            print("Retrying with simplified query:", simplified_query)
            results = db.similarity_search_with_score(simplified_query.strip(), k=8)

        # Keep slightly lower threshold for diversity
        results = [r for r in results if r[1] > 0.02]

        if not results:
            return jsonify({"answer": "I'm sorry, I couldn't find any relevant information."})

        retrieved_docs = [r[0] for r in results]

        for doc in retrieved_docs:
            doc.page_content = clean_context(doc.page_content[:1500])

        print("Top sources:", [d.metadata.get("source") for d in retrieved_docs])

        # ---- FIRST CHAIN: Summarize each document ----
        summary_prompt = PromptTemplate.from_template("""
        Summarize the following SAS travel information into 3–5 short factual sentences.
        Include only information directly relevant to the topic (e.g., baggage, medical, rebooking, seating).
        Avoid repeating the same facts.
        Keep numbers and restrictions accurate but concise.

        Text:
        {text}
        """)
        summarizer_chain = LLMChain(llm=llm, prompt=summary_prompt)

        # --- Deduplicate sentences before summarization ---
        texts = [deduplicate_sentences(doc.page_content) for doc in retrieved_docs]
        combined_text = "\n".join(texts)

        # --- Summarize if combined text is lengthy ---
        if len(combined_text) > 1000:
            summaries_output = summarizer_chain.apply([{"text": t} for t in texts])
            summaries = [s["text"].strip() if "text" in s else s for s in summaries_output]
            context = " ".join(summaries)
        else:
            context = combined_text

        # --- Final deduplication after summarization ---
        context = deduplicate_sentences(context)


        context = re.sub(r"\s+", " ", context).strip()
        context = re.sub(r"(?i)(see also|related links).*?$", "", context)
        context = re.sub(r'([.!?])[^.!?]*$', r'\1', context)

        # ---- SECOND CHAIN: Answer the user question ----
        answer_prompt = PromptTemplate.from_template("""
        You are SAS’s official virtual travel assistant.

        Use the context below to answer the passenger’s question as accurately as possible.
        If you find even partially relevant information, provide the best helpful answer you can.
        Use clear phrasing like "According to SAS policy..." or "You may need to contact SAS..."
        If the context partially answers the question, provide whatever relevant information is available.
        If the answer cannot be fully determined from the context, explain what is known and state that more specific details are not provided.
        Only say "I'm sorry, I don’t have that information right now" if there is absolutely no relevant information in the text.

        Guidelines:
        - Reply in 2–4 clear, natural sentences.
        - Use only the relevant factual information from the text; ignore titles or question headings.
        - Use customer-friendly wording (e.g., “checked baggage may weigh up to…”).
        - Avoid listing items unless explicitly asked.
        - Capitalize properly and end sentences cleanly.
        - Avoid repeating the same sentence or phrase more than once.

        Chat History:
        {chat_history}

        Context:
        {context}

        Question:
        {user_query}

        Final Answer:
        """)
        
        context = context[:5000]
        memory.clear()

        answer_chain = LLMChain(llm=llm, prompt=answer_prompt, memory=memory)
        answer_raw = answer_chain.run({"context": context, "user_query": user_query})

        # ---- Text refinement + polishing ----
        def polish_text(text):
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'(\b\w+(?:\s+\w+){2,5})\s+\1+', r'\1', text, flags=re.I)
            text = re.sub(r'\s([?.!,:;])', r'\1', text)
            text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
            text = text[0].upper() + text[1:] if text else text
            text = text.replace(" '", "'").replace("’", "'").replace(" - ", "–")
            text = re.sub(r'\bsas\b', 'SAS', text, flags=re.IGNORECASE)
            text = re.sub(r"\bi\b", "I", text)
            text = re.sub(r'\bon t .*', '', text, flags=re.I)
            if not text.endswith(('.', '!', '?')):
                text += '.'
            return text

        refine_prompt = PromptTemplate.from_template("""
        Rewrite the following text into 2–4 clean, professional sentences. 
        Make sure it is coherent, grammatically correct, and easy to read. 
        Keep all factual details but remove unnecessary repetition.
        If any phrases or sentences are repeated, keep only the clearest version.                                                 

        Text:
        {answer}
        """)    
        refine_chain = LLMChain(llm=llm, prompt=refine_prompt)
        answer_refined = refine_chain.run({"answer": answer_raw})

        answer_no_dupes = remove_repetitions(answer_refined)
        answer_clean = polish_text(answer_no_dupes)
        answer_clean = re.sub(r'([.!?])[^.!?]*$', r'\1', answer_clean)
        answer_clean = re.sub(r'(\bif you[^.]+?\(with some exceptions\)\s*)\1+', r'\1', answer_clean, flags=re.I)
        answer_clean = re.sub(r'\b(as a guest[^.]+?)\1+', r'\1', answer_clean, flags=re.I)

        print("Final cleaned answer:", answer_clean)
        return jsonify({"answer": answer_clean})

    except Exception as e:
        import traceback
        print("Error processing query (outer):")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

#if __name__ == "__main__":
    #init_error = initialize_chain()
    #if init_error:
        #print(f"Failed to start Flask app due to initialization error: {init_error}")
    #else:
        #public_url = ngrok.connect(5000)
        #print(" * Ngrok URL:", public_url)  # Display the public ngrok URL
        #app.run()  # Run Flask app

if __name__ == "__main__":
    init_error = initialize_chain()
    if init_error:
        print(f"Failed to start Flask app due to initialization error: {init_error}")
    else:
        app.run(host="0.0.0.0", port=8000, debug=True)
