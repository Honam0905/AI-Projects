"""
User Controller

This module handles user-related business logic.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from ..models.entities import User

# In-memory store for demo purposes
# In a real app, this would be replaced with database interactions
_users: Dict[str, User] = {}

class UserController:
    """Controller for user-related operations."""
    
    @staticmethod
    async def create_user(username: str, email: Optional[str] = None) -> User:
        """
        Create a new user.
        
        Args:
            username: The username for the new user
            email: Optional email address
            
        Returns:
            The created user
        """
        user = User(
            username=username,
            email=email
        )
        _users[user.id] = user
        return user
    
    @staticmethod
    async def get_user(user_id: str) -> Optional[User]:
        """
        Get a user by ID.
        
        Args:
            user_id: The user's ID
            
        Returns:
            The user if found, None otherwise
        """
        return _users.get(user_id)
    
    @staticmethod
    async def get_user_by_username(username: str) -> Optional[User]:
        """
        Get a user by username.
        
        Args:
            username: The username to search for
            
        Returns:
            The user if found, None otherwise
        """
        for user in _users.values():
            if user.username == username:
                return user
        return None
    
    @staticmethod
    async def update_user(user_id: str, **kwargs) -> Optional[User]:
        """
        Update a user's information.
        
        Args:
            user_id: The user's ID
            **kwargs: Fields to update
            
        Returns:
            The updated user if found, None otherwise
        """
        user = _users.get(user_id)
        if not user:
            return None
            
        # Update user fields
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
                
        # Set last activity time
        user.last_active = datetime.now()
        
        # Update in storage
        _users[user_id] = user
        
        return user
    
    @staticmethod
    async def delete_user(user_id: str) -> bool:
        """
        Delete a user.
        
        Args:
            user_id: The user's ID
            
        Returns:
            True if user was deleted, False if not found
        """
        if user_id in _users:
            del _users[user_id]
            return True
        return False
    
    @staticmethod
    async def get_all_users() -> List[User]:
        """
        Get all users.
        
        Returns:
            List of all users
        """
        return list(_users.values())
    
    @staticmethod
    async def update_preferences(user_id: str, preferences: Dict[str, Any]) -> Optional[User]:
        """
        Update a user's preferences.
        
        Args:
            user_id: The user's ID
            preferences: Dictionary of preferences to update
            
        Returns:
            The updated user if found, None otherwise
        """
        user = _users.get(user_id)
        if not user:
            return None
            
        # Update preferences, merging with existing ones
        user.preferences.update(preferences)
        
        # Set last activity time
        user.last_active = datetime.now()
        
        # Update in storage
        _users[user_id] = user
        
        return user 