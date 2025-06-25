"""
Reactive Agent Service for Crypto Trading

This service provides a reactive agent that helps users with crypto trading tasks.
It's based on LangGraph and can maintain state about market conditions, user portfolio,
and trading preferences.
"""

from typing import Literal, Optional, TypedDict, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from ..config import settings

# ----- Schema Definitions -----

class Portfolio(BaseModel):
    """User's crypto portfolio information"""
    total_value_usd: float = Field(description="Total portfolio value in USD", default=0.0)
    holdings: Dict[str, float] = Field(
        description="Map of cryptocurrency symbol to amount held",
        default_factory=dict
    )
    average_buy_prices: Dict[str, float] = Field(
        description="Map of cryptocurrency symbol to average buy price in USD",
        default_factory=dict
    )
    
class MarketInfo(BaseModel):
    """Information about a cryptocurrency market"""
    symbol: str = Field(description="Trading symbol (e.g., BTC, ETH)")
    name: str = Field(description="Full name (e.g., Bitcoin, Ethereum)")
    current_price_usd: float = Field(description="Current price in USD")
    market_cap_usd: Optional[float] = Field(description="Market capitalization in USD", default=None)
    change_24h_percent: Optional[float] = Field(description="24-hour price change percentage", default=None)
    volume_24h_usd: Optional[float] = Field(description="24-hour trading volume in USD", default=None)
    last_updated: Optional[datetime] = Field(description="When this market data was last updated", default=None)

class TradeOrder(BaseModel):
    """A cryptocurrency trade order"""
    symbol: str = Field(description="Trading symbol (e.g., BTC, ETH)")
    type: Literal["buy", "sell"] = Field(description="Order type")
    amount: float = Field(description="Amount of cryptocurrency to trade")
    price: Optional[float] = Field(description="Limit price in USD (if applicable)", default=None)
    status: Literal["planned", "executed", "cancelled"] = Field(description="Order status", default="planned")
    notes: Optional[str] = Field(description="Additional notes about this order", default=None)
    timestamp: Optional[datetime] = Field(description="When this order was created/executed", default=None)

class UserPreferences(BaseModel):
    """User's preferences for trading and portfolio management"""
    risk_tolerance: Literal["low", "medium", "high"] = Field(description="User's risk tolerance level", default="medium")
    preferred_currencies: List[str] = Field(description="User's preferred cryptocurrencies", default_factory=list)
    investment_goals: List[str] = Field(description="User's investment goals", default_factory=list)
    trade_frequency: Literal["low", "medium", "high"] = Field(description="User's trading frequency preference", default="medium")

# ----- Update Memory Type -----

class UpdateMemory(TypedDict):
    """Decision on what memory type to update"""
    update_type: Literal['portfolio', 'market', 'orders', 'preferences']

# ----- Utility Functions -----

def get_model(config: RunnableConfig):
    """Get the LLM model based on configuration."""
    # This could be expanded to support different model providers
    return ChatGroq(
        model=settings.AGENT_MODEL,
        temperature=0,
        groq_api_key=settings.GROQ_API_KEY
    )

# ----- Node Functions -----

def crypto_agent(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """
    Main agent node that processes user input and generates responses.
    
    Args:
        state: The current messages state
        config: Runtime configuration
        store: The memory store for persistent data
    
    Returns:
        Updated messages state
    """
    # Get user configuration
    user_id = config.get("configurable", {}).get("user_id", settings.DEFAULT_USER_ID)
    category = config.get("configurable", {}).get("category", settings.DEFAULT_CATEGORY)
    
    # Retrieve portfolio memory
    namespace = ("portfolio", category, user_id)
    memories = store.search(namespace)
    portfolio = memories[0].value if memories else None
    
    # Retrieve market info memory
    namespace = ("market", category, user_id)
    memories = store.search(namespace)
    market_info = "\n".join(f"{mem.value}" for mem in memories) if memories else ""
    
    # Retrieve orders memory
    namespace = ("orders", category, user_id)
    memories = store.search(namespace)
    orders = "\n".join(f"{mem.value}" for mem in memories) if memories else ""
    
    # Retrieve preferences memory
    namespace = ("preferences", category, user_id)
    memories = store.search(namespace)
    preferences = memories[0].value if memories else None
    
    # Build system message
    system_message = f"""
    {settings.REACT_AGENT_ROLE}
    
    You have a long-term memory that tracks:
    1. The user's portfolio (crypto holdings)
    2. Market information about cryptocurrencies
    3. The user's trade orders (planned and executed)
    4. The user's preferences for trading
    
    Here is the current User Portfolio (may be empty if no information collected yet):
    <portfolio>
    {portfolio if portfolio else "No portfolio information yet."}
    </portfolio>
    
    Here is the current Market Information (may be empty if no information collected yet):
    <market_info>
    {market_info if market_info else "No market information yet."}
    </market_info>
    
    Here is the current Trade Order List (may be empty if no orders yet):
    <orders>
    {orders if orders else "No trade orders yet."}
    </orders>
    
    Here are the user's trading preferences (may be empty if not specified yet):
    <preferences>
    {preferences if preferences else "No trading preferences specified yet."}
    </preferences>
    
    Please analyze the user's message and provide helpful information about cryptocurrencies,
    trading strategies, or portfolio management. If they ask about specific crypto prices or
    market conditions, you can use the UpdateMemory tool to indicate what information needs
    to be fetched or updated.
    """
    
    # Get the human message
    human_message = state["messages"][-1]
    
    # Call the model
    model = get_model(config)
    response = model.invoke([
        SystemMessage(content=system_message),
        *state["messages"]
    ])
    
    # Return the updated state with the model's response
    return {"messages": [*state["messages"], response]}

def update_portfolio(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """
    Update the user's portfolio information based on the conversation.
    
    This would typically call the enrichment service to get current portfolio data
    or update based on executed trades.
    """
    # Implementation would go here, similar to update_profile in task_maistro.py
    # For now, return unmodified state as placeholder
    return state

def update_market_info(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """
    Update market information based on user query.
    
    This would typically call the enrichment service or crypto data APIs
    to fetch current market data.
    """
    # Implementation would go here
    # For now, return unmodified state as placeholder
    return state

def update_orders(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """
    Update trade orders based on user instructions.
    
    This might involve creating new orders, cancelling existing ones,
    or marking orders as executed.
    """
    # Implementation would go here
    # For now, return unmodified state as placeholder
    return state

def update_preferences(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """
    Update user preferences based on the conversation.
    """
    # Implementation would go here
    # For now, return unmodified state as placeholder
    return state

def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_portfolio", "update_market_info", "update_orders", "update_preferences"]:
    """
    Route the message to the appropriate node based on what needs to be updated.
    """
    # Get model
    model = get_model(config)
    
    # Extract the last AI message to check for tools
    last_message = state["messages"][-1]
    
    # Check if we need to route to an update node
    try:
        # In a real implementation, we'd extract the tool calls from the message
        # and determine which node to route to based on what the model wants to update
        # For simplicity, defaulting to END
        return END
    except Exception as e:
        print(f"Error in routing: {str(e)}")
        return END

# ----- Graph Construction -----

def build_agent_graph(store=None) -> StateGraph:
    """
    Build the state graph for the reactive agent
    
    Args:
        store: Optional, an existing BaseStore to use (defaults to InMemoryStore)
        
    Returns:
        StateGraph: The constructed graph
    """
    # Create nodes
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("crypto_agent", crypto_agent)
    workflow.add_node("update_portfolio", update_portfolio)
    workflow.add_node("update_market_info", update_market_info)
    workflow.add_node("update_orders", update_orders)
    workflow.add_node("update_preferences", update_preferences)
    
    # Define edges
    workflow.add_edge(START, "crypto_agent")
    workflow.add_edge("crypto_agent", route_message)
    workflow.add_edge(route_message, END)
    workflow.add_edge(route_message, "update_portfolio")
    workflow.add_edge(route_message, "update_market_info")
    workflow.add_edge(route_message, "update_orders")
    workflow.add_edge(route_message, "update_preferences")
    workflow.add_edge("update_portfolio", END)
    workflow.add_edge("update_market_info", END)
    workflow.add_edge("update_orders", END)
    workflow.add_edge("update_preferences", END)
    
    # Compile and return
    return workflow.compile()

# ----- Service Interface Functions -----

def initialize_agent_service():
    """Initialize the agent service with in-memory store"""
    store = InMemoryStore()
    graph = build_agent_graph(store)
    memory_saver = MemorySaver(graph)
    return graph, memory_saver, store

def invoke_agent(user_id: str, message: str, category: str = None):
    """
    Invoke the agent with a user message
    
    Args:
        user_id: The user's ID
        message: The user's message
        category: Optional category (defaults to settings.DEFAULT_CATEGORY)
        
    Returns:
        Response from the agent
    """
    # Initialize the agent components
    graph, memory_saver, store = initialize_agent_service()
    
    # Set up the configurable parameters
    config = {
        "configurable": {
            "user_id": user_id,
            "category": category or settings.DEFAULT_CATEGORY,
        }
    }
    
    # Create input state
    input_state = {"messages": [HumanMessage(content=message)]}
    
    # Invoke the graph
    result = graph.invoke(input_state, config)
    
    # Return the final AI message
    return result["messages"][-1] 