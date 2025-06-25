"""
Entity Models

This module contains the core entity models used throughout the application.
These models represent the main business objects stored in the database.
"""

from datetime import datetime
from uuid import uuid4, UUID
from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field, model_validator

class User(BaseModel):
    """Model representing a user of the system."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    username: str = Field(..., description="User's username")
    email: Optional[str] = Field(None, description="User's email address")
    created_at: datetime = Field(default_factory=datetime.now, description="When this user was created")
    last_active: Optional[datetime] = Field(None, description="When user was last active")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")

class Portfolio(BaseModel):
    """Model representing a user's cryptocurrency portfolio."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    user_id: str = Field(..., description="ID of the user who owns this portfolio")
    name: str = Field("Default Portfolio", description="Portfolio name")
    description: Optional[str] = Field(None, description="Portfolio description")
    created_at: datetime = Field(default_factory=datetime.now, description="When this portfolio was created")
    updated_at: Optional[datetime] = Field(None, description="When this portfolio was last updated")
    
    # Portfolio contents
    holdings: Dict[str, float] = Field(default_factory=dict, description="Map of cryptocurrency symbol to amount held")
    total_value_usd: float = Field(default=0.0, description="Total portfolio value in USD")
    purchase_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of purchases")

class TradeOrder(BaseModel):
    """Model representing a cryptocurrency trade order."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    user_id: str = Field(..., description="ID of the user who created this order")
    portfolio_id: Optional[str] = Field(None, description="Portfolio this order affects")
    symbol: str = Field(..., description="Trading symbol (e.g., BTC, ETH)")
    order_type: Literal["buy", "sell"] = Field(..., description="Order type")
    amount: float = Field(..., description="Amount of cryptocurrency to trade")
    price: Optional[float] = Field(None, description="Limit price in USD (if applicable)")
    status: Literal["planned", "executed", "cancelled"] = Field(default="planned", description="Order status")
    notes: Optional[str] = Field(None, description="Additional notes about this order")
    created_at: datetime = Field(default_factory=datetime.now, description="When this order was created")
    executed_at: Optional[datetime] = Field(None, description="When this order was executed (if applicable)")
    
    @model_validator(mode='after')
    def set_timestamps(self):
        if self.status == "executed" and not self.executed_at:
            self.executed_at = datetime.now()
        return self

class MarketData(BaseModel):
    """Model representing market data for a cryptocurrency."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTC, ETH)")
    name: str = Field(..., description="Full name (e.g., Bitcoin, Ethereum)")
    current_price_usd: float = Field(..., description="Current price in USD")
    market_cap_usd: Optional[float] = Field(None, description="Market capitalization in USD")
    volume_24h_usd: Optional[float] = Field(None, description="24-hour trading volume in USD")
    change_24h_percent: Optional[float] = Field(None, description="24-hour price change percentage")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    data_source: str = Field("mock", description="Source of this market data")
    historical_prices: Optional[Dict[str, float]] = Field(None, description="Historical price data (timestamp: price)")

class Alert(BaseModel):
    """Model representing a price or market condition alert."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    user_id: str = Field(..., description="ID of the user who created this alert")
    symbol: str = Field(..., description="Trading symbol the alert is for")
    condition_type: Literal["price_above", "price_below", "percent_change"] = Field(..., description="Type of condition")
    threshold_value: float = Field(..., description="Threshold value for the alert")
    is_active: bool = Field(default=True, description="Whether this alert is active")
    created_at: datetime = Field(default_factory=datetime.now, description="When this alert was created")
    triggered_at: Optional[datetime] = Field(None, description="When this alert was triggered (if applicable)")
    notification_sent: bool = Field(default=False, description="Whether notification was sent")

class AnalysisReport(BaseModel):
    """Model representing an analysis report generated by the system."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    user_id: Optional[str] = Field(None, description="ID of the user this report was generated for (if applicable)")
    report_type: str = Field(..., description="Type of analysis report")
    content: str = Field(..., description="Report content")
    symbols: List[str] = Field(default_factory=list, description="Symbols included in this report")
    created_at: datetime = Field(default_factory=datetime.now, description="When this report was generated")
    data_sources: List[str] = Field(default_factory=list, description="Data sources used for this report") 