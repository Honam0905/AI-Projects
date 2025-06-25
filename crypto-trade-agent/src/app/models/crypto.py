"""
Crypto Models

This module contains Pydantic models for representing cryptocurrency-related data.
"""

from datetime import datetime
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class CryptoCurrency(BaseModel):
    """Model representing a cryptocurrency."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTC, ETH)")
    name: str = Field(..., description="Full name (e.g., Bitcoin, Ethereum)")
    current_price_usd: float = Field(..., description="Current price in USD")
    market_cap_usd: Optional[float] = Field(None, description="Market capitalization in USD")
    volume_24h_usd: Optional[float] = Field(None, description="24-hour trading volume in USD")
    change_24h_percent: Optional[float] = Field(None, description="24-hour price change percentage")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")

class PortfolioHolding(BaseModel):
    """Model representing a holding in a user's portfolio."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTC, ETH)")
    amount: float = Field(..., description="Amount of cryptocurrency held")
    avg_buy_price_usd: Optional[float] = Field(None, description="Average buy price in USD")
    current_value_usd: Optional[float] = Field(None, description="Current value in USD")
    profit_loss_usd: Optional[float] = Field(None, description="Unrealized profit/loss in USD")
    profit_loss_percent: Optional[float] = Field(None, description="Unrealized profit/loss percentage")

class Portfolio(BaseModel):
    """Model representing a user's cryptocurrency portfolio."""
    holdings: Dict[str, PortfolioHolding] = Field(default_factory=dict, description="Map of symbol to holding details")
    total_value_usd: float = Field(default=0.0, description="Total portfolio value in USD")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")

class TradeOrder(BaseModel):
    """Model representing a cryptocurrency trade order."""
    order_id: Optional[str] = Field(None, description="Unique order ID (generated on creation)")
    symbol: str = Field(..., description="Trading symbol (e.g., BTC, ETH)")
    order_type: Literal["buy", "sell"] = Field(..., description="Order type")
    amount: float = Field(..., description="Amount of cryptocurrency to trade")
    price: Optional[float] = Field(None, description="Limit price in USD (if applicable)")
    status: Literal["planned", "executed", "cancelled"] = Field(default="planned", description="Order status")
    notes: Optional[str] = Field(None, description="Additional notes about this order")
    created_at: Optional[datetime] = Field(None, description="When this order was created")
    executed_at: Optional[datetime] = Field(None, description="When this order was executed (if applicable)")

class UserPreferences(BaseModel):
    """Model representing a user's trading preferences."""
    risk_tolerance: Literal["low", "medium", "high"] = Field(default="medium", description="User's risk tolerance level")
    preferred_currencies: List[str] = Field(default_factory=list, description="User's preferred cryptocurrencies")
    investment_goals: List[str] = Field(default_factory=list, description="User's investment goals")
    trade_frequency: Literal["low", "medium", "high"] = Field(default="medium", description="User's trading frequency preference")
    notification_preferences: Dict[str, bool] = Field(default_factory=dict, description="User's notification preferences") 