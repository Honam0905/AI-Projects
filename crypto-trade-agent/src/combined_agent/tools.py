from typing import Any, List
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
# Need to add requests and beautifulsoup4 to dependencies
import requests 
from bs4 import BeautifulSoup

from .schemas import SearchInput, ScrapeWebsiteInput

# --- Standard Tools ---

# Adapting search tool from react-agent example
# Note: Requires TAVILY_API_KEY environment variable
@tool("tavily_search", args_schema=SearchInput)
async def search(query: str) -> str:
    """Search for general web results using Tavily.
    Useful for answering questions about current events or finding general information.
    """
    # TODO: Make max_results configurable
    tavily_tool = TavilySearchResults(max_results=3)
    return await tavily_tool.ainvoke({"query": query})


# Adapting scrape_website tool from data-enrichment example (simplified)
@tool("scrape_website", args_schema=ScrapeWebsiteInput)
async def scrape_website(url: str) -> str:
    """Scrape text content from a given website URL.
    Useful for extracting specific information from a webpage.
    """
    try:
        response = requests.get(url, timeout=10) # Added timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        
        # Check if content type is HTML
        if 'text/html' not in response.headers.get('Content-Type', ''):
            return f"Error: Content at {url} is not HTML."

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
            
        # Get text, strip leading/trailing whitespace, and reduce multiple newlines
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit length to avoid overwhelming the context
        max_length = 5000 
        return text[:max_length] + ('...' if len(text) > max_length else '')

    except requests.exceptions.RequestException as e:
        return f"Error scraping website {url}: {e}"
    except Exception as e:
        # Catch other potential errors during parsing
        return f"Error processing website {url}: {e}"

# Combine all standard tools for the agent
STANDARD_TOOLS: List[Any] = [search, scrape_website] 