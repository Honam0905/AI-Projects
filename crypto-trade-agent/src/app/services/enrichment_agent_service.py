"""
Data Enrichment Agent Service

This service provides a framework for enriching data on cryptocurrencies, market conditions,
and portfolio information using LLMs and external tools.
"""

from typing import Dict, List, Optional, TypedDict, Annotated, Literal, Union, Any
from pydantic import BaseModel, Field

from langchain_core.agents import AgentFinish
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
import requests

from langgraph.graph import StateGraph, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from ..config import settings

# --- State Types ---

class TopicState(TypedDict):
    """The state for the enrichment agent."""
    topic: str
    info: str
    loop_count: int
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    tools_used: Dict[str, int]

class SearchResult(BaseModel):
    """Search result from a tool like Tavily."""
    title: str
    url: str
    content: str

# --- Tools ---

def get_search_tool():
    @tool
    def search(query: str) -> str:
        """
        Search for information on a given query.
        Returns a summary of search results as a string.
        """
        # Implement real search here - mock for now
        # In a real implementation, you'd use the Tavily API or another search API
        
        # Placeholder implementation
        if "bitcoin" in query.lower():
            return """
            Bitcoin (BTC) is a decentralized cryptocurrency created in 2009. 
            Current price: $67,241 (example price)
            24h change: +2.3%
            Market cap: $1.32T
            Volume (24h): $42.5B
            
            Recent news: Bitcoin reached an all-time high last month after the approval of spot ETFs.
            """
        elif "ethereum" in query.lower():
            return """
            Ethereum (ETH) is a blockchain platform with its native cryptocurrency. 
            Current price: $3,214 (example price)
            24h change: -0.5%
            Market cap: $386B
            Volume (24h): $15.2B
            
            Recent news: Ethereum continues its transition to proof-of-stake with upcoming updates.
            """
        else:
            return f"No specific information found for {query}, please try another query."
    
    return search

def get_market_data_tool():
    @tool
    def get_market_data(symbol: str) -> str:
        """
        Get detailed market data for a specific cryptocurrency by symbol.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., BTC, ETH)
            
        Returns:
            String with detailed market data
        """
        # Mock implementation - in a real scenario, you'd call a crypto API like CoinGecko or Binance
        mock_data = {
            "BTC": {
                "name": "Bitcoin",
                "price_usd": 67241.50,
                "market_cap_usd": 1320000000000,
                "volume_24h_usd": 42500000000,
                "change_24h": 2.3,
                "last_updated": "2023-06-14T12:30:45Z"
            },
            "ETH": {
                "name": "Ethereum",
                "price_usd": 3214.75,
                "market_cap_usd": 386000000000,
                "volume_24h_usd": 15200000000,
                "change_24h": -0.5,
                "last_updated": "2023-06-14T12:31:22Z"
            }
        }
        
        symbol = symbol.upper()
        if symbol in mock_data:
            data = mock_data[symbol]
            return f"""
            {data['name']} ({symbol})
            Price: ${data['price_usd']}
            Market Cap: ${data['market_cap_usd']:,}
            24h Volume: ${data['volume_24h_usd']:,}
            24h Change: {data['change_24h']}%
            Last Updated: {data['last_updated']}
            """
        else:
            return f"No market data available for {symbol}"

    return get_market_data

def get_portfolio_analysis_tool():
    @tool
    def analyze_portfolio(holdings_json: str) -> str:
        """
        Analyze a crypto portfolio and provide insights.
        
        Args:
            holdings_json: A JSON string containing portfolio holdings in format:
                           {"BTC": amount1, "ETH": amount2, ...}
                           
        Returns:
            Analysis and recommendations
        """
        # In a real implementation, this would do a comprehensive analysis
        # based on current market data and historical trends
        return """
        Portfolio Analysis (Mock):
        - Bitcoin (BTC) position is strong with potential for 10-15% growth in next quarter
        - Ethereum (ETH) shows moderate volatility but remains a solid long-term hold
        - Consider diversifying with some mid-cap altcoins for better risk distribution
        - Overall portfolio risk level: Medium
        """
    
    return analyze_portfolio

# --- Node Functions ---

def enrichment_agent(state: TopicState, config: RunnableConfig, store: BaseStore) -> TopicState:
    """
    Main enrichment agent node that processes and enriches information.
    
    Args:
        state: The current state with topic and current info
        config: Runtime configuration 
        store: The persistent store
    
    Returns:
        Updated state
    """
    # Check if we've reached the maximum number of loops
    if state["loop_count"] >= settings.MAX_ENRICHMENT_LOOPS:
        # Add a message indicating we've reached the loop limit
        state["messages"].append(
            AIMessage(content="I've gathered as much information as I can given the constraints.")
        )
        return state
    
    # Get the prompt
    prompt = settings.ENRICHMENT_PROMPT.format(
        topic=state["topic"],
        info=state["info"]
    )
    
    # Create tools
    tools = [
        get_search_tool(),
        get_market_data_tool(),
        get_portfolio_analysis_tool()
    ]
    
    # Use LLM with tools to process the enrichment request
    # In a real implementation, this would use the proper LLM with tool calling
    # For now, just simulate a response
    
    # Simulate LLM thinking about what tools to use based on the topic
    if "price" in state["topic"].lower() or "market" in state["topic"].lower():
        # Simulate using the market data tool
        symbol = "BTC"  # This would be extracted from the topic by the LLM
        if "ethereum" in state["topic"].lower():
            symbol = "ETH"
            
        tool_name = "get_market_data"
        tool_result = get_market_data_tool()(symbol)
        
        # Record tool usage
        if tool_name in state["tools_used"]:
            state["tools_used"][tool_name] += 1
        else:
            state["tools_used"][tool_name] = 1
            
        # Update info
        updated_info = state["info"]
        if updated_info:
            updated_info += "\n\n" + tool_result
        else:
            updated_info = tool_result
            
        # Add messages
        state["messages"].append(
            AIMessage(content=f"I'll fetch market data for {symbol}.")
        )
        state["messages"].append(
            AIMessage(content=f"Here's what I found:\n{tool_result}")
        )
        
        # Update state
        state["info"] = updated_info
        state["loop_count"] += 1
        
    elif "portfolio" in state["topic"].lower() or "analysis" in state["topic"].lower():
        # Simulate using the portfolio analysis tool
        # In a real scenario, we'd parse actual holdings from the conversation
        mock_holdings = '{"BTC": 0.5, "ETH": 5.0}'
        
        tool_name = "analyze_portfolio"
        tool_result = get_portfolio_analysis_tool()(mock_holdings)
        
        # Record tool usage
        if tool_name in state["tools_used"]:
            state["tools_used"][tool_name] += 1
        else:
            state["tools_used"][tool_name] = 1
            
        # Update info
        updated_info = state["info"]
        if updated_info:
            updated_info += "\n\n" + tool_result
        else:
            updated_info = tool_result
            
        # Add messages
        state["messages"].append(
            AIMessage(content=f"I'll analyze your crypto portfolio.")
        )
        state["messages"].append(
            AIMessage(content=f"Here's my analysis:\n{tool_result}")
        )
        
        # Update state
        state["info"] = updated_info
        state["loop_count"] += 1
        
    else:
        # Default to general search
        tool_name = "search"
        tool_result = get_search_tool()(state["topic"])
        
        # Record tool usage
        if tool_name in state["tools_used"]:
            state["tools_used"][tool_name] += 1
        else:
            state["tools_used"][tool_name] = 1
            
        # Update info
        updated_info = state["info"]
        if updated_info:
            updated_info += "\n\n" + tool_result
        else:
            updated_info = tool_result
            
        # Add messages
        state["messages"].append(
            AIMessage(content=f"I'll search for information on '{state['topic']}'.")
        )
        state["messages"].append(
            AIMessage(content=f"Here's what I found:\n{tool_result}")
        )
        
        # Update state
        state["info"] = updated_info
        state["loop_count"] += 1
        
    return state

def should_continue(state: TopicState) -> Literal["enrichment_agent", END]:
    """
    Determine if we should continue enriching or stop.
    """
    # Stop if we've reached max loops
    if state["loop_count"] >= settings.MAX_ENRICHMENT_LOOPS:
        return END
    
    # Stop if we've used the search tool maximum times
    if state["tools_used"].get("search", 0) >= settings.MAX_INFO_TOOL_CALLS:
        return END
    
    # Otherwise, continue enrichment
    return "enrichment_agent"

# --- Graph Construction ---

def build_enrichment_graph() -> StateGraph:
    """
    Build the graph for the enrichment agent.
    
    Returns:
        StateGraph: The constructed graph
    """
    # Create workflow
    workflow = StateGraph(TopicState)
    
    # Add nodes
    workflow.add_node("enrichment_agent", enrichment_agent)
    
    # Define edges
    workflow.add_edge("enrichment_agent", should_continue)
    workflow.add_edge(should_continue, "enrichment_agent")
    workflow.add_edge(should_continue, END)
    
    # Set entry point
    workflow.set_entry_point("enrichment_agent")
    
    # Compile and return
    return workflow.compile()

# --- Service Interface Functions ---

def initialize_enrichment_service():
    """
    Initialize the enrichment service.
    
    Returns:
        tuple: (graph, store)
    """
    graph = build_enrichment_graph()
    store = InMemoryStore()
    return graph, store

def enrich_topic(topic: str, existing_info: str = "") -> Dict[str, Any]:
    """
    Enrich a topic with additional information.
    
    Args:
        topic: The topic to enrich (e.g., "Bitcoin price trends", "Ethereum vs Solana")
        existing_info: Optional existing information about the topic
        
    Returns:
        dict: Enriched information and messages
    """
    # Initialize the service
    graph, store = initialize_enrichment_service()
    
    # Build initial state
    state = {
        "topic": topic,
        "info": existing_info,
        "loop_count": 0,
        "messages": [HumanMessage(content=f"Please provide information about: {topic}")],
        "tools_used": {}
    }
    
    # Invoke the graph
    result = graph.invoke(state)
    
    # Return the enriched information
    return {
        "topic": topic,
        "enriched_info": result["info"],
        "messages": [m.content for m in result["messages"]],
        "tools_used": result["tools_used"]
    } 