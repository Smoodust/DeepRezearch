import time
from typing import Optional

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain.chat_models import init_chat_model

model = init_chat_model("qwen3:0.6b", model_provider="ollama")
search_engine = DDGS()
prompt = """You are an expert web content extractor. Your task is to identify and extract the main content text from the provided HTML document while removing all non-essential elements.

**Instructions:**

1. **Analyze the HTML structure** and identify the primary content container
2. **Extract only the main content** - the core article, blog post, or informational text
3. **Remove completely:**
   - Navigation menus and headers
   - Sidebars and widgets
   - Footer content
   - Advertisement blocks
   - Social media buttons/links
   - Comment sections
   - Metadata (author, date, tags) unless specifically requested
   - Script tags and style elements
   - Redundant whitespace and line breaks

4. **Preserve:**
   - Article headings and subheadings
   - Paragraph text
   - Lists (ordered and unordered)
   - Tables when they contain relevant data
   - Important inline formatting (bold, italics) when semantically meaningful
   - Logical paragraph spacing

5. **Output requirements:**
   - Clean, readable plain text
   - Maintain logical flow and reading order
   - Remove all HTML tags while preserving text structure
   - Normalize whitespace (collapse multiple spaces, proper line breaks)
   - Output should be well-formatted and immediately readable

**Special considerations:**
- Look for semantic HTML tags like `<article>`, `<main>`, `<section>`
- Identify content-heavy divs with classes like 'content', 'article', 'post', 'main'
- Use text density and structural analysis to find primary content
- If multiple content sections exist, prioritize the largest coherent text block

**HTML Input:**
<html>
{html_content}
</html>

**Output:**"""


def _extract_text_from_url(url: str) -> Optional[str]:
    """Extract readable text from a web page"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean it up
        if soup.body is None:
            return None

        text = soup.body.prettify()
        return model.invoke(prompt.format(html_content=text)).text

    except Exception as e:
        print(f"Error extracting text from {url}: {str(e)}")
        return None


def search_internet(query: str, limit: int = 10) -> list[str]:
    """Search the DuckDuckGo matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    search_results = search_engine.text(query, max_results=limit)
    extracted_texts = []

    for i, result in enumerate(search_results):
        url = result.get("href", "")
        title = result.get("title", "No title")

        print(f"Processing result {i+1}: {title}")

        # Extract text from URL
        page_text = _extract_text_from_url(url)
        if page_text:
            model.invoke()
            extracted_texts.append(
                f"## Result {len(extracted_texts)+1}: {title}\nContent: {page_text}\n"
            )

        # Add small delay to be respectful to servers
        time.sleep(1)
    return extracted_texts


for x in search_internet(query="python programming"):
    print(x)
    print("=" * 30)
