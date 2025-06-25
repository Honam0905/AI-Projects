"""
Portfolio Routes

This module contains API routes for portfolio management.
"""

from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Path, Body

from ..controllers.portfolio_controller import PortfolioController
from ..models.entities import Portfolio

router = APIRouter(prefix="/portfolios", tags=["Portfolios"])

@router.post("", response_model=Portfolio)
async def create_portfolio(
    user_id: str = Body(...),
    name: str = Body("Default Portfolio"),
    description: Optional[str] = Body(None)
):
    """
    Create a new portfolio for a user.
    """
    portfolio = await PortfolioController.create_portfolio(user_id, name, description)
    return portfolio

@router.get("", response_model=List[Portfolio])
async def get_user_portfolios(user_id: str = Query(...)):
    """
    Get all portfolios for a user.
    """
    return await PortfolioController.get_user_portfolios(user_id)

@router.get("/{portfolio_id}", response_model=Portfolio)
async def get_portfolio(portfolio_id: str = Path(...)):
    """
    Get a portfolio by ID.
    """
    portfolio = await PortfolioController.get_portfolio(portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio

@router.put("/{portfolio_id}", response_model=Portfolio)
async def update_portfolio(
    portfolio_id: str = Path(...),
    name: Optional[str] = Body(None),
    description: Optional[str] = Body(None)
):
    """
    Update a portfolio's information.
    """
    # Build update dict with only provided fields
    update_data = {}
    if name is not None:
        update_data["name"] = name
    if description is not None:
        update_data["description"] = description
        
    portfolio = await PortfolioController.update_portfolio(portfolio_id, **update_data)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio

@router.delete("/{portfolio_id}")
async def delete_portfolio(portfolio_id: str = Path(...)):
    """
    Delete a portfolio.
    """
    success = await PortfolioController.delete_portfolio(portfolio_id)
    if not success:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return {"message": "Portfolio deleted successfully"}

@router.post("/{portfolio_id}/holdings", response_model=Portfolio)
async def add_holding(
    portfolio_id: str = Path(...),
    symbol: str = Body(...),
    amount: float = Body(...),
    purchase_price: Optional[float] = Body(None)
):
    """
    Add a cryptocurrency holding to a portfolio.
    """
    portfolio = await PortfolioController.add_holding(portfolio_id, symbol, amount, purchase_price)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio

@router.delete("/{portfolio_id}/holdings/{symbol}", response_model=Portfolio)
async def remove_holding(
    portfolio_id: str = Path(...),
    symbol: str = Path(...),
    amount: float = Query(...)
):
    """
    Remove a cryptocurrency holding from a portfolio.
    """
    portfolio = await PortfolioController.remove_holding(portfolio_id, symbol, amount)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio

@router.get("/{portfolio_id}/value")
async def update_portfolio_value(portfolio_id: str = Path(...)):
    """
    Update and get the total value of a portfolio.
    """
    value = await PortfolioController.update_portfolio_value(portfolio_id)
    if value is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return {"portfolio_id": portfolio_id, "total_value_usd": value}

@router.get("/{portfolio_id}/performance")
async def get_portfolio_performance(portfolio_id: str = Path(...)):
    """
    Get performance metrics for a portfolio.
    """
    performance = await PortfolioController.get_portfolio_performance(portfolio_id)
    if not performance:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return performance 