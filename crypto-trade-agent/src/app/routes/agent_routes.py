"""
Agent Routes

This module contains the API routes for interacting with the crypto trading agent.
"""

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ..services.react_agent_service import invoke_agent
from ..services.enrichment_agent_service import enrich_topic
from ..config import settings

# --- Models ---

class AgentQuery(BaseModel):
    """Request model for agent interaction."""
    message: str = Field(..., description="The user's message to the agent")
    category: Optional[str] = Field(None, description="Optional category for agent context")

class EnrichmentQuery(BaseModel):
    """Request model for data enrichment."""
    topic: str = Field(..., description="The topic to enrich with data")
    existing_info: Optional[str] = Field("", description="Existing information about the topic")

class AgentResponse(BaseModel):
    """Response model for agent interaction."""
    user_id: str = Field(..., description="The user's ID")
    response: str = Field(..., description="The agent's response")
    updates: Optional[Dict[str, Any]] = Field(None, description="Any memory or state updates")

class EnrichmentResponse(BaseModel):
    """Response model for data enrichment."""
    topic: str = Field(..., description="The topic that was enriched")
    enriched_info: str = Field(..., description="The enriched information")
    messages: List[str] = Field(..., description="The messages from the enrichment process")
    tools_used: Dict[str, int] = Field(..., description="Tools used and their counts")

# --- Router ---

router = APIRouter(tags=["Agent"])

# --- Routes ---

@router.post("/chat/{user_id}", response_model=AgentResponse)
async def chat_with_agent(
    user_id: str,
    query: AgentQuery,
    background_tasks: BackgroundTasks
):
    """
    Chat with the crypto trading agent.
    
    This endpoint sends the user's message to the reactive agent, which processes it
    and responds appropriately.
    
    The agent maintains memory of the user's portfolio, market information, and preferences.
    """
    try:
        # Invoke the agent
        response = invoke_agent(
            user_id=user_id,
            message=query.message,
            category=query.category
        )
        
        # Extract the response content
        # In a real implementation, you might want to extract more detailed information
        # about memory updates, etc.
        return AgentResponse(
            user_id=user_id,
            response=response.content if hasattr(response, 'content') else str(response),
            updates=None  # In a real implementation, you'd include actual memory updates
        )
    except Exception as e:
        # Log the error
        print(f"Error in chat_with_agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@router.post("/enrich", response_model=EnrichmentResponse)
async def enrich_data(query: EnrichmentQuery):
    """
    Enrich data about a crypto-related topic.
    
    This endpoint uses the enrichment agent to gather and process information about
    the specified topic, potentially using tools like search, market data APIs, etc.
    """
    try:
        # Call the enrichment service
        result = enrich_topic(
            topic=query.topic,
            existing_info=query.existing_info
        )
        
        # Return the result
        return EnrichmentResponse(
            topic=result["topic"],
            enriched_info=result["enriched_info"],
            messages=result["messages"],
            tools_used=result["tools_used"]
        )
    except Exception as e:
        # Log the error
        print(f"Error in enrich_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enrichment error: {str(e)}")

@router.get("/market/{symbol}")
async def get_market_info(symbol: str):
    """
    Get market information for a specific cryptocurrency.
    
    This is a convenience endpoint that directly accesses market data
    without going through the full agent workflow.
    """
    try:
        # Use the enrichment service to get market data
        result = enrich_topic(f"market data for {symbol}")
        
        # Extract just the relevant market data from the enriched info
        return {
            "symbol": symbol.upper(),
            "data": result["enriched_info"],
        }
    except Exception as e:
        print(f"Error in get_market_info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Market data error: {str(e)}") 