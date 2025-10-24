import time
import os
import re
import hashlib
import requests
from urllib.parse import urljoin, urlparse
from urllib import robotparser
from bs4 import BeautifulSoup
from queue import Queue

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from requests_html import HTMLSession

# ==============================
# CONFIG
# ==============================
SEED = "https://www.flysas.com/en/travel-info"
BASE_DOMAIN = "flysas.com"
ALLOWED_PATH_PREFIX = "/en/travel-info"
USER_AGENT = "MyCrawler/1.0 (+https://example.com/contact)"
REQUEST_TIMEOUT = 10
RATE_LIMIT_SECONDS = 0.5
MAX_PAGES = 300
MAX_DEPTH = 4
SITEMAP_URL = "https://www.sas.no/sitemap/content/sitemap.xml"
FAISS_DIR = "faiss_index"
CACHE_DIR = "html_cache"

if not os.path.exists(CACHE_DIR):   
    os.makedirs(CACHE_DIR)

# ==============================
# HELPERS
# ==============================
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})
html_session = HTMLSession()

def allowed_by_robots(url):
    rp = robotparser.RobotFileParser()
    robots_url = urljoin(url, "/robots.txt")
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception as e:
        print("Could not read robots.txt:", e)
        return False

# Fetch and render HTML with caching
def fetch_rendered_html(url):
    filename = hashlib.md5(url.encode("utf-8")).hexdigest() + ".html"
    filepath = os.path.join(CACHE_DIR, filename)

    # 1. Use cache if available
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
        print(f"Loaded from cache: {url}")
        return html

    # 2. Fetch and render JS
    try:
        print(f"Fetching & rendering: {url}")
        r = html_session.get(url, timeout=REQUEST_TIMEOUT)
        r.html.render(timeout=20, sleep=2, scrolldown=0)
        html = r.html.html

        # 3. Save to cache
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        return html
    except Exception as e:
        print(f"JS render failed for {url}: {e}")
        return None

def get_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.netloc.endswith(BASE_DOMAIN) and parsed.path.startswith(ALLOWED_PATH_PREFIX):
            normalized = parsed.scheme + "://" + parsed.netloc + parsed.path
            links.add(normalized)
    return links

# ==============================
# TEXT EXTRACTION
# ==============================
def extract_text(html):
    """Extract normal text and FAQ pairs (button + hidden div answers)."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for s in soup(["script", "style", "header", "footer", "nav", "aside"]):
        s.decompose()

    main = soup.find("main") or soup
    paragraphs, faqs = [], []

    # Regular paragraphs, bullet points, and headings
    for p in main.find_all(["p", "li", "h2", "h3"]):
        txt = p.get_text(" ", strip=True)
        if txt and len(txt) > 40:
            paragraphs.append(txt)

    # --- Extract FAQ questions and answers ---
    # On SAS, buttons hold the question and the following div (class u1zlc1e...) has the answer
    for button in main.find_all("button"):
        q_text = button.get_text(" ", strip=True)
        # Look for next <div> sibling or any div with the FAQ answer class
        answer_div = button.find_next_sibling("div")
        if not answer_div:
            answer_div = main.find("div", class_=re.compile(r"u1zlc1e"))
        if answer_div:
            a_text = " ".join(
                p.get_text(" ", strip=True)
                for p in answer_div.find_all(["p", "li"])
                if p.get_text(strip=True)
            )
            if q_text and a_text and len(a_text) > 30:
                faqs.append(f"Q: {q_text}")
                faqs.append(f"A: {a_text}")

    combined = "\n".join(paragraphs + faqs)
    return combined

# ==============================
# CRAWLER
# ==============================
def crawl(seed, max_pages=MAX_PAGES, max_depth=MAX_DEPTH):
    if not allowed_by_robots(seed):
        raise RuntimeError(f"Crawling disallowed by robots.txt for {seed}")

    q = Queue()
    q.put((seed, 0))
    visited = set()
    results = []

    while not q.empty() and len(visited) < max_pages:
        url, depth = q.get()
        if url in visited or depth > max_depth:
            continue
        try:
            html = fetch_rendered_html(url)
            if not html:
                visited.add(url)
                continue

            text = extract_text(html)
            if text and len(text.strip()) > 50:
                results.append((url, text))
                print(f"Parsed {url}")

            visited.add(url)

            if depth < max_depth and len(visited) < max_pages:
                links = get_links(html, url)
                for link in links:
                    if link not in visited:
                        q.put((link, depth + 1))

            time.sleep(RATE_LIMIT_SECONDS)
        except Exception as e:
            print("Error fetching", url, e)
            visited.add(url)
            time.sleep(RATE_LIMIT_SECONDS)
    return results

# ==============================
# SITEMAP PARSER
# ==============================
def parse_sitemap(sitemap_url):
    try:
        r = session.get(sitemap_url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        import xml.etree.ElementTree as ET
        tree = ET.fromstring(r.content)
        urls = []
        for loc in tree.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
            u = loc.text
            if u and urlparse(u).netloc.endswith(BASE_DOMAIN) and urlparse(u).path.startswith(ALLOWED_PATH_PREFIX):
                urls.append(u)
        return urls
    except Exception as e:
        print("Sitemap parse error:", e)
        return None

# ==============================
# FAISS INGESTION
# ==============================
def ingest_to_faiss(pages):
    docs = [Document(page_content=text, metadata={"source": url}) for url, text in pages]

    from langchain_text_splitters import SentenceTransformersTokenTextSplitter
    splitter = SentenceTransformersTokenTextSplitter(chunk_size=512, chunk_overlap=100)
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_documents([doc]))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    db = FAISS.from_documents(chunks, embeddings)

    if not os.path.exists(FAISS_DIR):
        os.makedirs(FAISS_DIR)
    db.save_local(FAISS_DIR)
    print("Saved FAISS index to", FAISS_DIR)

# ==============================
# MAIN
# ==============================
def main():
    pages = []
    sitemap_urls = parse_sitemap(SITEMAP_URL)

    if sitemap_urls:
        print(f"Using sitemap, found {len(sitemap_urls)} urls")
        count = 0
        for url in sitemap_urls:
            if count >= MAX_PAGES:
                break
            if not allowed_by_robots(url):
                continue
            html = fetch_rendered_html(url)
            if not html:
                continue
            text = extract_text(html)
            if text and len(text) > 50:
                pages.append((url, text))
                count += 1
                print(f"Processed ({count}) {url}")
            time.sleep(RATE_LIMIT_SECONDS)
    else:
        print("No sitemap found — fallback to crawl()")
        pages = crawl(SEED)

    if pages:
        ingest_to_faiss(pages)
    else:
        print("No pages collected — nothing to ingest")

if __name__ == "__main__":
    main()
