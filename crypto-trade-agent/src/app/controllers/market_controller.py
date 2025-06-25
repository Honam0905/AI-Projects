"""
Market Controller

This module handles market data-related business logic.
"""

from datetime import datetime, timedelta
import random
from typing import Dict, List, Optional, Any

from ..models.entities import MarketData
from ..utils.api_helpers import get_mock_market_data, format_market_data_response
from ..services.enrichment_agent_service import enrich_topic

class MarketController:
    """Controller for market data-related operations."""
    
    @staticmethod
    async def get_market_data(symbol: Optional[str] = None) -> Dict[str, MarketData]:
        """
        Get market data for one or all cryptocurrencies.
        
        Args:
            symbol: Optional symbol to get data for a specific cryptocurrency
            
        Returns:
            Dictionary of market data
        """
        # Get raw data from the API helper
        raw_data = get_mock_market_data(symbol)
        
        # Convert to MarketData objects
        result = {}
        for sym, data in raw_data.items():
            market_data = MarketData(
                symbol=sym,
                name=data.get("name", sym),
                current_price_usd=data.get("current_price_usd", 0),
                market_cap_usd=data.get("market_cap_usd"),
                volume_24h_usd=data.get("volume_24h_usd"),
                change_24h_percent=data.get("change_24h_percent"),
                last_updated=datetime.fromtimestamp(data.get("last_updated")) if isinstance(data.get("last_updated"), (int, float)) else datetime.now(),
                data_source="mock"
            )
            result[sym] = market_data
            
        return result
    
    @staticmethod
    async def get_market_summary() -> str:
        """
        Get a formatted summary of current market conditions.
        
        Returns:
            Formatted string with market summary
        """
        market_data = get_mock_market_data()
        return format_market_data_response(market_data)
    
    @staticmethod
    async def get_enriched_market_info(symbol: str) -> Dict[str, Any]:
        """
        Get enriched market information using the data enrichment service.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Dictionary with enriched data
        """
        # Use the enrichment service to get detailed information
        result = await enrich_topic(f"market data and trends for {symbol}")
        
        # Add basic market data
        symbol = symbol.upper()
        market_data = get_mock_market_data(symbol)
        
        return {
            "symbol": symbol,
            "market_data": market_data.get(symbol, {}),
            "enriched_info": result.get("enriched_info", ""),
            "sources": result.get("tools_used", {})
        }
    
    @staticmethod
    async def get_price_history(symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Get price history for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days of history to get
            
        Returns:
            Dictionary with price history data
        """
        symbol = symbol.upper()
        
        # In a real implementation, this would call a real API
        # For demo purposes, we'll generate mock data
        
        # Get current price as baseline
        market_data = get_mock_market_data(symbol)
        if symbol not in market_data:
            return {"error": f"No data available for {symbol}"}
            
        current_price = market_data[symbol].get("current_price_usd", 10000)
        
        # Generate mock price history
        price_history = []
        volatility = 0.02  # Daily price movement range (2%)
        
        # Pick volatility based on the crypto
        if symbol == "BTC":
            volatility = 0.03
        elif symbol == "ETH":
            volatility = 0.04
        elif symbol == "SOL":
            volatility = 0.07
        else:
            volatility = 0.05
        
        # Start from current price and work backwards with random changes
        price = current_price
        now = datetime.now()
        
        for i in range(days):
            date = now - timedelta(days=i)
            
            # Add some randomness to the price movement
            change_percent = (random.random() - 0.5) * 2 * volatility
            # Move price for the previous day
            price = price / (1 + change_percent)
            
            price_history.append({
                "date": date.strftime("%Y-%m-%d"),
                "timestamp": int(date.timestamp()),
                "price_usd": round(price, 2)
            })
        
        # Reverse so oldest is first
        price_history.reverse()
        
        return {
            "symbol": symbol,
            "name": market_data[symbol].get("name", symbol),
            "days": days,
            "price_history": price_history,
            "current_price_usd": current_price
        }
    
    @staticmethod
    async def analyze_trend(symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Analyze price trend for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        # Get price history first
        history_data = await MarketController.get_price_history(symbol, days)
        
        if "error" in history_data:
            return history_data
            
        price_history = history_data.get("price_history", [])
        
        if not price_history or len(price_history) < 2:
            return {"error": "Insufficient price data for analysis"}
        
        # Calculate simple trend metrics
        oldest_price = price_history[0]["price_usd"]
        newest_price = price_history[-1]["price_usd"]
        
        price_change = newest_price - oldest_price
        price_change_percent = ((newest_price / oldest_price) - 1) * 100
        
        # Calculate volatility (standard deviation of daily returns)
        daily_returns = []
        for i in range(1, len(price_history)):
            yesterday = price_history[i-1]["price_usd"]
            today = price_history[i]["price_usd"]
            daily_return = (today / yesterday) - 1
            daily_returns.append(daily_return)
        
        import statistics
        volatility = statistics.stdev(daily_returns) * 100 if len(daily_returns) > 1 else 0
        
        # Simple moving averages
        ma7 = sum(ph["price_usd"] for ph in price_history[-7:]) / min(7, len(price_history)) if len(price_history) > 0 else 0
        ma30 = sum(ph["price_usd"] for ph in price_history[-30:]) / min(30, len(price_history)) if len(price_history) > 0 else 0
        
        # Determine trend direction
        if price_change_percent > 5:
            trend = "strongly bullish"
        elif price_change_percent > 0:
            trend = "bullish"
        elif price_change_percent > -5:
            trend = "bearish"
        else:
            trend = "strongly bearish"
            
        # Compare to recent MA
        if newest_price > ma7:
            ma_signal = "above short-term MA (bullish)"
        else:
            ma_signal = "below short-term MA (bearish)"
            
        return {
            "symbol": symbol,
            "name": history_data.get("name", symbol),
            "days_analyzed": days,
            "price_start": oldest_price,
            "price_end": newest_price,
            "price_change_usd": price_change,
            "price_change_percent": price_change_percent,
            "volatility_percent": volatility,
            "moving_avg_7day": ma7,
            "moving_avg_30day": ma30,
            "trend": trend,
            "signal": ma_signal,
            "analysis_summary": f"{symbol} has shown a {trend} trend over the {days} day period with {volatility:.2f}% volatility. The price is currently {ma_signal}."
        } 