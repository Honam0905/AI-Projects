"""
Trade Controller

This module handles trade-related business logic.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from ..models.entities import TradeOrder
from ..controllers.portfolio_controller import PortfolioController
from ..utils.api_helpers import get_mock_market_data

# In-memory store for demo purposes
# In a real app, this would be replaced with database interactions
_orders: Dict[str, TradeOrder] = {}

class TradeController:
    """Controller for trade-related operations."""
    
    @staticmethod
    async def create_order(
        user_id: str,
        symbol: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        portfolio_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> TradeOrder:
        """
        Create a new trade order.
        
        Args:
            user_id: The user's ID
            symbol: Cryptocurrency symbol
            order_type: "buy" or "sell"
            amount: Amount to trade
            price: Optional limit price
            portfolio_id: Optional portfolio ID to associate with
            notes: Optional notes
            
        Returns:
            The created trade order
        """
        # Normalize inputs
        symbol = symbol.upper()
        order_type = order_type.lower()
        
        # Create order
        order = TradeOrder(
            user_id=user_id,
            symbol=symbol,
            order_type=order_type,
            amount=amount,
            price=price,
            portfolio_id=portfolio_id,
            notes=notes,
            status="planned"
        )
        
        # Store order
        _orders[order.id] = order
        
        return order
    
    @staticmethod
    async def get_order(order_id: str) -> Optional[TradeOrder]:
        """
        Get a trade order by ID.
        
        Args:
            order_id: The order's ID
            
        Returns:
            The order if found, None otherwise
        """
        return _orders.get(order_id)
    
    @staticmethod
    async def get_user_orders(user_id: str, status: Optional[str] = None) -> List[TradeOrder]:
        """
        Get all trade orders for a user.
        
        Args:
            user_id: The user's ID
            status: Optional status to filter by
            
        Returns:
            List of orders
        """
        orders = [o for o in _orders.values() if o.user_id == user_id]
        
        if status:
            orders = [o for o in orders if o.status == status]
            
        # Sort by created time, newest first
        orders.sort(key=lambda o: o.created_at, reverse=True)
        
        return orders
    
    @staticmethod
    async def cancel_order(order_id: str) -> Optional[TradeOrder]:
        """
        Cancel a trade order.
        
        Args:
            order_id: The order's ID
            
        Returns:
            The updated order if found, None otherwise
        """
        order = _orders.get(order_id)
        if not order:
            return None
            
        # Only allow cancelling planned orders
        if order.status != "planned":
            return order
            
        # Update status
        order.status = "cancelled"
        
        # Update in storage
        _orders[order_id] = order
        
        return order
    
    @staticmethod
    async def execute_order(order_id: str, execution_price: Optional[float] = None) -> Optional[TradeOrder]:
        """
        Mark a trade order as executed.
        
        Args:
            order_id: The order's ID
            execution_price: Optional execution price (defaults to current market price)
            
        Returns:
            The updated order if found, None otherwise
        """
        order = _orders.get(order_id)
        if not order:
            return None
            
        # Only allow executing planned orders
        if order.status != "planned":
            return order
            
        # If no execution price provided, use current market price
        if not execution_price:
            market_data = get_mock_market_data(order.symbol)
            if order.symbol in market_data:
                execution_price = market_data[order.symbol].get("current_price_usd")
            else:
                # Fall back to limit price if available, otherwise set to 0
                execution_price = order.price if order.price else 0
                
        # Update order
        order.status = "executed"
        order.price = execution_price
        order.executed_at = datetime.now()
        
        # Update in storage
        _orders[order_id] = order
        
        # If the order is associated with a portfolio, update the portfolio holdings
        if order.portfolio_id:
            if order.order_type == "buy":
                # Add to portfolio
                await PortfolioController.add_holding(
                    portfolio_id=order.portfolio_id,
                    symbol=order.symbol,
                    amount=order.amount,
                    purchase_price=execution_price
                )
            elif order.order_type == "sell":
                # Remove from portfolio
                await PortfolioController.remove_holding(
                    portfolio_id=order.portfolio_id,
                    symbol=order.symbol,
                    amount=order.amount
                )
                
        return order
    
    @staticmethod
    async def get_order_summary(user_id: str) -> Dict[str, Any]:
        """
        Get a summary of a user's trade orders.
        
        Args:
            user_id: The user's ID
            
        Returns:
            Summary of orders
        """
        orders = await TradeController.get_user_orders(user_id)
        
        # Count orders by status
        status_counts = {
            "planned": 0,
            "executed": 0,
            "cancelled": 0
        }
        
        # Count orders by type
        type_counts = {
            "buy": 0,
            "sell": 0
        }
        
        # Calculate total value by type
        total_values = {
            "buy": 0.0,
            "sell": 0.0
        }
        
        # Count by symbol
        symbol_counts = {}
        
        for order in orders:
            # Count by status
            status_counts[order.status] = status_counts.get(order.status, 0) + 1
            
            # Count by type (only if not cancelled)
            if order.status != "cancelled":
                type_counts[order.order_type] = type_counts.get(order.order_type, 0) + 1
            
                # Calculate value
                price = order.price or 0
                value = price * order.amount
                total_values[order.order_type] += value
                
                # Count by symbol
                symbol_counts[order.symbol] = symbol_counts.get(order.symbol, 0) + 1
        
        return {
            "user_id": user_id,
            "total_orders": len(orders),
            "by_status": status_counts,
            "by_type": type_counts,
            "total_values": total_values,
            "by_symbol": symbol_counts,
            "recent_orders": [o.dict() for o in orders[:5]]  # First 5 orders (newest)
        }
    
    @staticmethod
    async def suggest_trade(user_id: str, portfolio_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Suggest a trade based on market data and user's portfolio.
        
        Args:
            user_id: The user's ID
            portfolio_id: Optional portfolio ID to base suggestion on
            
        Returns:
            Trade suggestion
        """
        # Get market data
        market_data = get_mock_market_data()
        
        # Find highest gaining and losing cryptos
        cryptos = []
        for symbol, data in market_data.items():
            cryptos.append({
                "symbol": symbol,
                "name": data.get("name", symbol),
                "price": data.get("current_price_usd", 0),
                "change_24h": data.get("change_24h_percent", 0)
            })
        
        # Sort by 24h change
        cryptos.sort(key=lambda c: c["change_24h"], reverse=True)
        
        # Get user's portfolio if ID provided
        portfolio = None
        if portfolio_id:
            portfolio = await PortfolioController.get_portfolio(portfolio_id)
        
        # Generate suggestions
        suggestions = []
        
        # If we have a portfolio, suggest selling worst performers and buying best performers
        if portfolio:
            # Find portfolio holdings that are doing poorly
            for symbol, amount in portfolio.holdings.items():
                if symbol in market_data:
                    change = market_data[symbol].get("change_24h_percent", 0)
                    if change < -5:  # Suggest selling if down more than 5%
                        suggestions.append({
                            "type": "sell",
                            "symbol": symbol,
                            "reason": f"Down {change:.2f}% in the last 24 hours",
                            "suggested_amount": amount * 0.5  # Suggest selling half
                        })
            
            # Find good performers not in portfolio
            portfolio_symbols = set(portfolio.holdings.keys())
            for crypto in cryptos:
                if crypto["symbol"] not in portfolio_symbols and crypto["change_24h"] > 5:
                    suggestions.append({
                        "type": "buy",
                        "symbol": crypto["symbol"],
                        "reason": f"Up {crypto['change_24h']:.2f}% in the last 24 hours",
                        "suggested_amount": 1  # Generic suggestion
                    })
        else:
            # Without a portfolio, just suggest the top performers
            for crypto in cryptos[:3]:  # Top 3
                if crypto["change_24h"] > 0:
                    suggestions.append({
                        "type": "buy",
                        "symbol": crypto["symbol"],
                        "reason": f"Up {crypto['change_24h']:.2f}% in the last 24 hours",
                        "suggested_amount": 1  # Generic suggestion
                    })
        
        return {
            "user_id": user_id,
            "portfolio_id": portfolio_id,
            "market_summary": "Market is showing mixed performance in the last 24 hours.",
            "top_gainers": cryptos[:3],
            "top_losers": cryptos[-3:],
            "suggestions": suggestions
        } 