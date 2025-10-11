import os, requests, html2text
from bs4 import BeautifulSoup
from tqdm import tqdm

# ==== CONFIG ====
URLS = [

    "https://cloud.google.com/vertex-ai/generative-ai/docs/overview",
    "https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview",
    "https://cloud.google.com/vertex-ai/docs/workbench/introduction",
    "https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview",
    "https://cloud.google.com/vertex-ai/docs/training/custom-training",
    "https://cloud.google.com/vertex-ai/generative-ai/docs/models",
    "https://cloud.google.com/vertex-ai/docs/reference/rest",
    "https://cloud.google.com/vertex-ai/docs/tutorials",
    "https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/mistral",
    "https://cloud.google.com/vision-ai/docs",
    "https://cloud.google.com/vertex-ai/generative-ai/docs/samples/googlegenaisdk-textgen-with-multi-local-img",
    "https://cloud.google.com/vertex-ai",
    "https://cloudchipr.com/blog/vertex-ai",
    "https://www.upwork.com/resources/vertex-ai",
    "https://cloud.google.com/generative-ai-studio?hl=en",
    "https://sasmaster.medium.com/my-experience-with-googles-vertex-ai-c604964888f0",
    "https://medium.com/@ironhack/what-is-vertex-ai-b1f457cd7d0b",
    "https://medium.com/@williamwarley/mastering-gcp-vertex-ai-with-javascript-a-comprehensive-guide-for-beginners-and-experts-6ea4e8ef139b",
    "https://codelabs.developers.google.com/devsite/codelabs/building-ai-agents-vertexai#2"
]

OUT_DIR = "input_docs/raw_html"
TXT_DIR = "input_docs/docs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

def clean_html(html):
    """Strip navbars / scripts and return main readable content."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()
    return str(soup)

def html_to_text(html):
    """Convert HTML to Markdown-like plain text."""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    return h.handle(html)

def scrape_page(url):
    """Download, clean, and save as .txt"""
    name = url.split("/")[-1] or url.split("/")[-2]
    html_path = os.path.join(OUT_DIR, f"{name}.html")
    txt_path = os.path.join(TXT_DIR, f"vertex_ai_{name}.txt")

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        cleaned = clean_html(r.text)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        text = html_to_text(cleaned)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"Error on {url}: {e}")
        return False

if __name__ == "__main__":
    print("Scraping Vertex AI docs ...")
    for url in tqdm(URLS):
        scrape_page(url)
    print("Done. Text files ready in:", TXT_DIR)
