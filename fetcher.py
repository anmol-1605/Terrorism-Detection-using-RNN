
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
import time

class WebContentFetcher:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.timeout = 10

    def is_valid_url(self, url):
        """Check if the URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def fetch_url_content(self, url):
        """Fetch content from URL and return processed text"""
        if not self.is_valid_url(url):
            return {"error": "Invalid URL format"}

        try:
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            # Make request
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Extract text content
            text_content = self.extract_meaningful_text(soup)

            # Process and clean the text
            processed_text = self.clean_text(text_content)

            return {
                "url": url,
                "title": soup.title.string if soup.title else "No title",
                "processed_text": processed_text,
                "word_count": len(processed_text.split()),
                "status": "success"
            }

        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}

    def extract_meaningful_text(self, soup):
        """Extract meaningful text from BeautifulSoup object"""
        # Priority tags for content
        content_tags = ['article', 'main', 'content', 'post']

        # Try to find main content area
        main_content = None
        for tag in content_tags:
            main_content = soup.find(tag)
            if main_content:
                break

        if not main_content:
            # If no main content found, use body
            main_content = soup.find('body') or soup

        # Extract text from paragraphs, headings, and lists
        text_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div'])

        text_content = []
        for element in text_elements:
            text = element.get_text(strip=True)
            if text and len(text) > 20:  # Filter out very short text
                text_content.append(text)

        return ' '.join(text_content)

    def clean_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', ' ', text)

        # Remove multiple consecutive punctuation
        text = re.sub(r'[.,!?;:-]{2,}', '.', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Limit text length to prevent memory issues
        max_length = 5000
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text

    def get_site_info(self, url):
        """Get basic information about the website"""
        try:
            parsed_url = urlparse(url)
            return {
                "domain": parsed_url.netloc,
                "scheme": parsed_url.scheme,
                "path": parsed_url.path
            }
        except:
            return {"domain": "unknown", "scheme": "unknown", "path": "unknown"}

# Test function
def test_fetcher():
    fetcher = WebContentFetcher()
    test_url = "https://example.com"
    result = fetcher.fetch_url_content(test_url)
    print("Test result:", result)

if __name__ == "__main__":
    test_fetcher()
