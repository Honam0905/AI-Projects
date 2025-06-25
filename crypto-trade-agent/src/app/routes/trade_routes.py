"""
Trade Routes

This module contains API routes for trade order management.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Path, Body

from ..controllers.trade_controller import TradeController
from ..models.entities import TradeOrder

router = APIRouter(prefix="/trades", tags=["Trades"])

@router.post("", response_model=TradeOrder)
async def create_order(
    user_id: str = Body(...),
    symbol: str = Body(...),
    order_type: str = Body(...),
    amount: float = Body(...),
    price: Optional[float] = Body(None),
    portfolio_id: Optional[str] = Body(None),
    notes: Optional[str] = Body(None)
):
    """
    Create a new trade order.
    """
    order = await TradeController.create_order(
        user_id=user_id,
        symbol=symbol,
        order_type=order_type,
        amount=amount,
        price=price,
        portfolio_id=portfolio_id,
        notes=notes
    )
    return order

@router.get("", response_model=List[TradeOrder])
async def get_user_orders(
    user_id: str = Query(...),
    status: Optional[str] = Query(None)
):
    """
    Get all trade orders for a user.
    """
    return await TradeController.get_user_orders(user_id, status)

@router.get("/{order_id}", response_model=TradeOrder)
async def get_order(order_id: str = Path(...)):
    """
    Get a trade order by ID.
    """
    order = await TradeController.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order

@router.post("/{order_id}/cancel", response_model=TradeOrder)
async def cancel_order(order_id: str = Path(...)):
    """
    Cancel a trade order.
    """
    order = await TradeController.cancel_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order

@router.post("/{order_id}/execute", response_model=TradeOrder)
async def execute_order(
    order_id: str = Path(...),
    execution_price: Optional[float] = Body(None)
):
    """
    Mark a trade order as executed.
    """
    order = await TradeController.execute_order(order_id, execution_price)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order

@router.get("/summary/{user_id}")
async def get_order_summary(user_id: str = Path(...)):
    """
    Get a summary of a user's trade orders.
    """
    summary = await TradeController.get_order_summary(user_id)
    return summary

@router.get("/suggest/{user_id}")
async def suggest_trade(
    user_id: str = Path(...),
    portfolio_id: Optional[str] = Query(None)
):
    """
    Suggest a trade based on market data and user's portfolio.
    """
    suggestion = await TradeController.suggest_trade(user_id, portfolio_id)
    return suggestion 