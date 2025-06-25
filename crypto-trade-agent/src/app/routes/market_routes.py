"""
Market Routes

This module contains API routes for market data.
"""

from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, Query, Path

from ..controllers.market_controller import MarketController
from ..models.entities import MarketData

router = APIRouter(prefix="/market", tags=["Market Data"])

@router.get("/data")
async def get_market_data(symbol: Optional[str] = Query(None)):
    """
    Get market data for cryptocurrencies.
    
    Args:
        symbol: Optional symbol to filter for a specific cryptocurrency
    """
    data = await MarketController.get_market_data(symbol)
    # Convert MarketData objects to dictionaries
    result = {sym: market_data.dict() for sym, market_data in data.items()}
    return result

@router.get("/summary")
async def get_market_summary():
    """
    Get a formatted summary of current market conditions.
    """
    summary = await MarketController.get_market_summary()
    return {"summary": summary}

@router.get("/info/{symbol}")
async def get_enriched_market_info(symbol: str = Path(...)):
    """
    Get enriched market information using the data enrichment service.
    
    Args:
        symbol: Cryptocurrency symbol
    """
    return await MarketController.get_enriched_market_info(symbol)

@router.get("/history/{symbol}")
async def get_price_history(
    symbol: str = Path(...),
    days: int = Query(30, ge=1, le=365)
):
    """
    Get price history for a cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol
        days: Number of days of history to get (1-365)
    """
    history = await MarketController.get_price_history(symbol, days)
    if "error" in history:
        raise HTTPException(status_code=404, detail=history["error"])
    return history

@router.get("/trends/{symbol}")
async def analyze_trend(
    symbol: str = Path(...),
    days: int = Query(30, ge=1, le=365)
):
    """
    Analyze price trend for a cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol
        days: Number of days to analyze (1-365)
    """
    analysis = await MarketController.analyze_trend(symbol, days)
    if "error" in analysis:
        raise HTTPException(status_code=404, detail=analysis["error"])
    return analysis 