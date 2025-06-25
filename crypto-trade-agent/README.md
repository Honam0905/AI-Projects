# Crypto Trading Agent API

A FastAPI-based backend for a reactive crypto trading assistant, powered by LLMs and LangGraph.

## Overview

This project provides a conversational AI agent specialized in crypto trading assistance. It combines:

1. **Reactive Agent**: A conversational bot that maintains memory about user portfolios, preferences, and trading history.
2. **Data Enrichment**: A system that fetches and processes real-time crypto market data.

The system is built using FastAPI for the API layer and LangGraph for the agent orchestration.

## Features

- **Conversational Interface**: Chat with the agent about trading strategies, portfolio management, and market insights.
- **Market Data**: Get real-time information about cryptocurrency prices, market caps, and trends.
- **Portfolio Tracking**: Store and analyze your cryptocurrency holdings.
- **Trade Planning**: Plan and track potential trades.
- **Personalized Advice**: Receive recommendations based on your risk tolerance and preferences.

## Architecture

The project is organized into several key components:

- `src/app/main.py`: Main FastAPI application entry point
- `src/app/config.py`: Configuration using Pydantic Settings
- `src/app/services/`: Agent implementations
  - `react_agent_service.py`: Main conversational agent
  - `enrichment_agent_service.py`: Data enrichment agent
- `src/app/routes/`: API routes
- `src/app/models/`: Pydantic data models
- `src/app/utils/`: Utility functions

## Installation

### Prerequisites

- Python 3.9+
- API keys for LLM providers (Groq, Anthropic, etc.)
- Optional: Tavily API key for web search

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/crypto-trade-agent-backend.git
cd crypto-trade-agent-backend
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create environment variables file:

```bash
cp .env-example .env
```

5. Edit the `.env` file and add your API keys:

```
GROQ_API_KEY=your_actual_groq_api_key
TAVILY_API_KEY=your_actual_tavily_api_key
```

## Usage

### Running the Server

Start the FastAPI server:

```bash
# From the project root directory
python -m src.app.main
```

Alternatively, use uvicorn directly:

```bash
uvicorn src.app.main:app --reload
```

The API will be available at http://localhost:8000, with interactive documentation at http://localhost:8000/docs.

### API Endpoints

#### Root Endpoint

- `GET /`: Basic information about the API

#### Health Check

- `GET /health`: Health check endpoint

#### Agent Interaction

- `POST /api/v1/chat/{user_id}`: Chat with the crypto trading agent
  - Request body:
    ```json
    {
      "message": "What's the current price of Bitcoin?",
      "category": "crypto_trading"
    }
    ```

#### Data Enrichment

- `POST /api/v1/enrich`: Enrich data about a crypto-related topic
  - Request body:
    ```json
    {
      "topic": "Bitcoin price trends",
      "existing_info": "" 
    }
    ```

#### Market Data

- `GET /api/v1/market/{symbol}`: Get market information for a specific cryptocurrency

## Example Interactions

### Chat with the Agent

```bash
curl -X POST "http://localhost:8000/api/v1/chat/user123" \
  -H "Content-Type: application/json" \
  -d '{"message": "I own 0.5 BTC and 5 ETH. How is my portfolio doing?"}'
```

### Get Market Data

```bash
curl -X GET "http://localhost:8000/api/v1/market/BTC"
```

## Current Limitations

- The current implementation uses mock data for market prices. To get real prices, you'll need to integrate with cryptocurrency APIs (like CoinGecko, Binance, etc.).
- The agent's memory is in-memory by default. For production use, configure a persistent storage option.

## Extending the System

### Adding New Cryptocurrencies

Update the mock data in `src/app/utils/api_helpers.py` or integrate with a real crypto API.

### Adding New Agent Capabilities

1. Add new tools to the enrichment agent in `src/app/services/enrichment_agent_service.py`
2. Add new state update functions in `src/app/services/react_agent_service.py`
3. Update the agent's node graph as needed

## Development

### Project Structure

```
.
├── .env-example              # Template for environment variables
├── requirements.txt          # Python dependencies
├── src/                      # Source code
│   └── app/
│       ├── main.py           # FastAPI application
│       ├── config.py         # Configuration
│       ├── models/           # Data models
│       │   ├── __init__.py
│       │   └── crypto.py     # Crypto-specific models
│       ├── routes/           # API routes
│       │   └── agent_routes.py
│       ├── services/         # Agent services
│       │   ├── react_agent_service.py    # Main agent
│       │   └── enrichment_agent_service.py  # Data enrichment
│       └── utils/            # Utilities
│           ├── __init__.py
│           └── api_helpers.py  # Helper functions
```

## License

[MIT License](LICENSE)

## Acknowledgements

This project was inspired by and adapted from:
- LangGraph examples for reactive agent patterns
- FastAPI best practices for building robust APIs 