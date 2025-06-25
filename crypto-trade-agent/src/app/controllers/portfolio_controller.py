"""
Portfolio Controller

This module handles portfolio-related business logic.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from ..models.entities import Portfolio
from ..utils.api_helpers import get_mock_market_data

# In-memory store for demo purposes
# In a real app, this would be replaced with database interactions
_portfolios: Dict[str, Portfolio] = {}

class PortfolioController:
    """Controller for portfolio-related operations."""
    
    @staticmethod
    async def create_portfolio(user_id: str, name: str = "Default Portfolio", description: Optional[str] = None) -> Portfolio:
        """
        Create a new portfolio for a user.
        
        Args:
            user_id: The user's ID
            name: Portfolio name
            description: Optional description
            
        Returns:
            The created portfolio
        """
        portfolio = Portfolio(
            user_id=user_id,
            name=name,
            description=description
        )
        _portfolios[portfolio.id] = portfolio
        return portfolio
    
    @staticmethod
    async def get_portfolio(portfolio_id: str) -> Optional[Portfolio]:
        """
        Get a portfolio by ID.
        
        Args:
            portfolio_id: The portfolio's ID
            
        Returns:
            The portfolio if found, None otherwise
        """
        return _portfolios.get(portfolio_id)
    
    @staticmethod
    async def get_user_portfolios(user_id: str) -> List[Portfolio]:
        """
        Get all portfolios for a user.
        
        Args:
            user_id: The user's ID
            
        Returns:
            List of portfolios
        """
        return [p for p in _portfolios.values() if p.user_id == user_id]
    
    @staticmethod
    async def update_portfolio(portfolio_id: str, **kwargs) -> Optional[Portfolio]:
        """
        Update a portfolio's information.
        
        Args:
            portfolio_id: The portfolio's ID
            **kwargs: Fields to update
            
        Returns:
            The updated portfolio if found, None otherwise
        """
        portfolio = _portfolios.get(portfolio_id)
        if not portfolio:
            return None
            
        # Update portfolio fields
        for key, value in kwargs.items():
            if hasattr(portfolio, key):
                setattr(portfolio, key, value)
                
        # Set updated_at timestamp
        portfolio.updated_at = datetime.now()
        
        # Update in storage
        _portfolios[portfolio_id] = portfolio
        
        return portfolio
    
    @staticmethod
    async def delete_portfolio(portfolio_id: str) -> bool:
        """
        Delete a portfolio.
        
        Args:
            portfolio_id: The portfolio's ID
            
        Returns:
            True if portfolio was deleted, False if not found
        """
        if portfolio_id in _portfolios:
            del _portfolios[portfolio_id]
            return True
        return False
    
    @staticmethod
    async def add_holding(portfolio_id: str, symbol: str, amount: float, purchase_price: Optional[float] = None) -> Optional[Portfolio]:
        """
        Add a cryptocurrency holding to a portfolio.
        
        Args:
            portfolio_id: The portfolio's ID
            symbol: The cryptocurrency symbol
            amount: Amount to add
            purchase_price: Optional purchase price per unit
            
        Returns:
            The updated portfolio if found, None otherwise
        """
        portfolio = _portfolios.get(portfolio_id)
        if not portfolio:
            return None
            
        # Normalize symbol
        symbol = symbol.upper()
        
        # Add to holdings or update if exists
        current_amount = portfolio.holdings.get(symbol, 0)
        portfolio.holdings[symbol] = current_amount + amount
        
        # Record purchase history
        if purchase_price:
            purchase_record = {
                "symbol": symbol,
                "amount": amount,
                "price_usd": purchase_price,
                "timestamp": datetime.now().isoformat()
            }
            portfolio.purchase_history.append(purchase_record)
        
        # Update portfolio value
        await PortfolioController.update_portfolio_value(portfolio_id)
        
        # Set updated_at timestamp
        portfolio.updated_at = datetime.now()
        
        # Update in storage
        _portfolios[portfolio_id] = portfolio
        
        return portfolio
    
    @staticmethod
    async def remove_holding(portfolio_id: str, symbol: str, amount: float) -> Optional[Portfolio]:
        """
        Remove a cryptocurrency holding from a portfolio.
        
        Args:
            portfolio_id: The portfolio's ID
            symbol: The cryptocurrency symbol
            amount: Amount to remove
            
        Returns:
            The updated portfolio if found, None otherwise
        """
        portfolio = _portfolios.get(portfolio_id)
        if not portfolio:
            return None
            
        # Normalize symbol
        symbol = symbol.upper()
        
        # Check if symbol exists in holdings
        if symbol not in portfolio.holdings:
            return portfolio
            
        # Remove amount (or all if amount >= current holdings)
        current_amount = portfolio.holdings.get(symbol, 0)
        new_amount = max(0, current_amount - amount)
        
        if new_amount > 0:
            portfolio.holdings[symbol] = new_amount
        else:
            # Remove entirely if zero
            del portfolio.holdings[symbol]
        
        # Update portfolio value
        await PortfolioController.update_portfolio_value(portfolio_id)
        
        # Set updated_at timestamp
        portfolio.updated_at = datetime.now()
        
        # Update in storage
        _portfolios[portfolio_id] = portfolio
        
        return portfolio
    
    @staticmethod
    async def update_portfolio_value(portfolio_id: str) -> Optional[float]:
        """
        Update the total value of a portfolio based on current market prices.
        
        Args:
            portfolio_id: The portfolio's ID
            
        Returns:
            The updated total value if portfolio found, None otherwise
        """
        portfolio = _portfolios.get(portfolio_id)
        if not portfolio:
            return None
            
        # Get current market data for all symbols in the portfolio
        market_data = get_mock_market_data()
        
        # Calculate total value
        total_value = 0.0
        for symbol, amount in portfolio.holdings.items():
            if symbol in market_data:
                price = market_data[symbol].get("current_price_usd", 0)
                total_value += price * amount
        
        # Update portfolio
        portfolio.total_value_usd = total_value
        portfolio.updated_at = datetime.now()
        
        # Update in storage
        _portfolios[portfolio_id] = portfolio
        
        return total_value
    
    @staticmethod
    async def get_portfolio_performance(portfolio_id: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for a portfolio.
        
        Args:
            portfolio_id: The portfolio's ID
            
        Returns:
            Dictionary of performance metrics if portfolio found, None otherwise
        """
        portfolio = _portfolios.get(portfolio_id)
        if not portfolio:
            return None
            
        # Get current market data
        market_data = get_mock_market_data()
        
        # Calculate performance metrics
        performance = {
            "total_value_usd": portfolio.total_value_usd,
            "holdings_count": len(portfolio.holdings),
            "holdings_detail": []
        }
        
        # Calculate per-holding metrics
        for symbol, amount in portfolio.holdings.items():
            if symbol in market_data:
                crypto_data = market_data[symbol]
                current_price = crypto_data.get("current_price_usd", 0)
                current_value = current_price * amount
                
                # Find purchase records for this symbol to calculate profit/loss
                purchase_records = [
                    pr for pr in portfolio.purchase_history 
                    if pr["symbol"] == symbol
                ]
                
                avg_purchase_price = 0
                if purchase_records:
                    total_amount = 0
                    total_cost = 0
                    for record in purchase_records:
                        rec_amount = record.get("amount", 0)
                        rec_price = record.get("price_usd", 0)
                        total_amount += rec_amount
                        total_cost += rec_amount * rec_price
                    
                    avg_purchase_price = total_cost / total_amount if total_amount > 0 else 0
                
                # Calculate profit/loss
                profit_loss = 0
                profit_loss_percent = 0
                if avg_purchase_price > 0:
                    profit_loss = (current_price - avg_purchase_price) * amount
                    profit_loss_percent = ((current_price / avg_purchase_price) - 1) * 100
                
                holding_detail = {
                    "symbol": symbol,
                    "name": crypto_data.get("name", symbol),
                    "amount": amount,
                    "current_price_usd": current_price,
                    "current_value_usd": current_value,
                    "avg_purchase_price_usd": avg_purchase_price,
                    "profit_loss_usd": profit_loss,
                    "profit_loss_percent": profit_loss_percent,
                    "share_of_portfolio": (current_value / portfolio.total_value_usd * 100) if portfolio.total_value_usd > 0 else 0
                }
                
                performance["holdings_detail"].append(holding_detail)
        
        return performance 