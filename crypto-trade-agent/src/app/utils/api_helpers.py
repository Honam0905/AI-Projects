"""
API Helper Utilities.

This module contains helper functions for API-related tasks.
"""

import time
import json
from typing import Dict, Any, Optional

# Mock crypto market data cache
_mock_market_data = {}
_mock_market_data_timestamp = 0
_CACHE_EXPIRY = 60  # seconds

def get_mock_market_data(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Get mock market data for cryptocurrencies.
    
    In a real implementation, this would call an actual crypto API.
    
    Args:
        symbol: Optional symbol to filter for a specific cryptocurrency
        
    Returns:
        Dict containing market data
    """
    global _mock_market_data, _mock_market_data_timestamp
    
    # Check if we need to refresh the cache
    current_time = time.time()
    if current_time - _mock_market_data_timestamp > _CACHE_EXPIRY:
        # This would be a real API call in production
        _mock_market_data = {
            "BTC": {
                "name": "Bitcoin",
                "symbol": "BTC",
                "current_price_usd": 67241.50,
                "market_cap_usd": 1320000000000,
                "volume_24h_usd": 42500000000,
                "change_24h_percent": 2.3,
                "last_updated": current_time
            },
            "ETH": {
                "name": "Ethereum",
                "symbol": "ETH",
                "current_price_usd": 3214.75,
                "market_cap_usd": 386000000000,
                "volume_24h_usd": 15200000000,
                "change_24h_percent": -0.5,
                "last_updated": current_time
            },
            "SOL": {
                "name": "Solana",
                "symbol": "SOL",
                "current_price_usd": 138.25,
                "market_cap_usd": 59800000000,
                "volume_24h_usd": 3500000000,
                "change_24h_percent": 5.7,
                "last_updated": current_time
            },
            "ADA": {
                "name": "Cardano",
                "symbol": "ADA",
                "current_price_usd": 0.45,
                "market_cap_usd": 16200000000,
                "volume_24h_usd": 850000000,
                "change_24h_percent": 1.2,
                "last_updated": current_time
            }
        }
        _mock_market_data_timestamp = current_time
    
    # Return data for specific symbol if requested
    if symbol:
        symbol = symbol.upper()
        return {symbol: _mock_market_data.get(symbol, {})} if symbol in _mock_market_data else {}
    
    return _mock_market_data

def format_error_response(error: Exception, status_code: int = 500) -> Dict[str, Any]:
    """
    Format an error response consistently.
    
    Args:
        error: The exception that occurred
        status_code: HTTP status code
        
    Returns:
        Formatted error response
    """
    return {
        "status_code": status_code,
        "error": str(error),
        "error_type": type(error).__name__
    }

def parse_json_safely(json_str: str) -> Dict[str, Any]:
    """
    Safely parse JSON, handling errors.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed JSON as dict
        
    Raises:
        ValueError: If JSON is invalid
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")
        
def format_market_data_response(data: Dict[str, Any]) -> str:
    """
    Format market data into a readable string.
    
    Args:
        data: Market data dictionary
        
    Returns:
        Formatted string
    """
    if not data:
        return "No market data available."
        
    result = []
    for symbol, crypto_data in data.items():
        if not crypto_data:
            continue
            
        price = crypto_data.get("current_price_usd", "N/A")
        change = crypto_data.get("change_24h_percent", "N/A")
        market_cap = crypto_data.get("market_cap_usd", "N/A")
        
        if market_cap != "N/A":
            # Format market cap in billions/trillions
            if market_cap >= 1_000_000_000_000:
                market_cap = f"${market_cap / 1_000_000_000_000:.2f}T"
            elif market_cap >= 1_000_000_000:
                market_cap = f"${market_cap / 1_000_000_000:.2f}B"
            else:
                market_cap = f"${market_cap:,.2f}"
                
        # Format the line
        line = f"{crypto_data.get('name', symbol)} ({symbol}): "
        line += f"${price:,.2f} "
        
        # Add change with ↑/↓ arrows
        if change != "N/A":
            if change > 0:
                line += f"↑{change:+.2f}% "
            elif change < 0:
                line += f"↓{change:.2f}% "
            else:
                line += f"{change:.2f}% "
                
        line += f"Market Cap: {market_cap}"
        
        result.append(line)
        
    return "\n".join(result) 