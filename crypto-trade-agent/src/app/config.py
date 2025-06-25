import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env file first if it exists
load_dotenv()

class Settings(BaseSettings):
    # --- General App Settings ---
    APP_NAME: str = "Crypto Trading Agent API"
    API_V1_STR: str = "/api/v1"
    # Set to 'development', 'production', or 'testing'
    ENVIRONMENT: str = "development"

    # --- LLM & Agent Settings (from examples) ---
    # Groq API Key (from task_maistro example)
    GROQ_API_KEY: str = ""

    # Model selection (can choose based on environment or specific need)
    # Defaulting to groq model from task_maistro example
    AGENT_MODEL: str = "llama3-70b-8192"
    ENRICHMENT_MODEL: str = "anthropic/claude-3-5-sonnet-20240620" # From enrichment example

    # Task Maistro/React Agent specific settings
    DEFAULT_USER_ID: str = "default-user"
    DEFAULT_CATEGORY: str = "crypto_trading" # Make it more specific
    REACT_AGENT_ROLE: str = (
        "You are a crypto trading assistant. You help the user analyze market data, "
        "manage their portfolio, and potentially execute trades based on their instructions. "
        "You can use tools to fetch real-time data and enrich information."
    )

    # Data Enrichment specific settings
    ENRICHMENT_PROMPT: str = ( # Simplified default, consider moving complex prompts elsewhere
        "Enrich the following topic: {topic} with this existing information: {info}"
    )
    MAX_SEARCH_RESULTS: int = 5 # Reduced default
    MAX_INFO_TOOL_CALLS: int = 3
    MAX_ENRICHMENT_LOOPS: int = 6

    # --- External Service APIs (Example: Add others as needed) ---
    TAVILY_API_KEY: str | None = None # If using Tavily for search tool

    # --- Database/Storage Settings (Placeholder - adapt if using DB/MemorySaver) ---
    # Example: REDIS_URL: str | None = None
    # Example: POSTGRES_DSN: str | None = None
    MEMORY_STORE_TYPE: str = "in_memory" # or "redis", "postgres", etc.

    # --- Pydantic Settings ---
    # Load from .env file in addition to environment variables
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

# Instantiate settings
settings = Settings()

# You can add helper functions here, e.g., to get specific LLM clients based on settings
# from langchain_groq import ChatGroq
# from langchain_anthropic import ChatAnthropic # Need to install langchain-anthropic
#
# def get_react_agent_llm():
#     if "groq" in settings.AGENT_MODEL.lower():
#         return ChatGroq(model=settings.AGENT_MODEL, groq_api_key=settings.GROQ_API_KEY, temperature=0)
#     # Add other providers (Anthropic, OpenAI, etc.) here
#     else:
#         raise ValueError(f"Unsupported agent model provider for: {settings.AGENT_MODEL}")
#
# def get_enrichment_llm():
#     if "anthropic" in settings.ENRICHMENT_MODEL.lower():
#         # Make sure ANTHROPIC_API_KEY is set in .env or environment
#         return ChatAnthropic(model=settings.ENRICHMENT_MODEL, temperature=0)
#     # Add other providers here
#     else:
#         raise ValueError(f"Unsupported enrichment model provider for: {settings.ENRICHMENT_MODEL}") 