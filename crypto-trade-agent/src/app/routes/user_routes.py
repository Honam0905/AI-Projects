"""
User Routes

This module contains API routes for user management.
"""

from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Path, Body

from ..controllers.user_controller import UserController
from ..models.entities import User

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("", response_model=User)
async def create_user(username: str = Body(...), email: Optional[str] = Body(None)):
    """
    Create a new user.
    """
    # Check if username already exists
    existing_user = await UserController.get_user_by_username(username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
        
    user = await UserController.create_user(username, email)
    return user

@router.get("", response_model=List[User])
async def get_all_users():
    """
    Get all users.
    """
    return await UserController.get_all_users()

@router.get("/{user_id}", response_model=User)
async def get_user(user_id: str = Path(...)):
    """
    Get a user by ID.
    """
    user = await UserController.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/{user_id}", response_model=User)
async def update_user(user_id: str = Path(...), username: Optional[str] = Body(None), email: Optional[str] = Body(None)):
    """
    Update a user's information.
    """
    # First check if the user exists
    existing_user = await UserController.get_user(user_id)
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Build update dict with only provided fields
    update_data = {}
    if username is not None:
        update_data["username"] = username
    if email is not None:
        update_data["email"] = email
        
    user = await UserController.update_user(user_id, **update_data)
    return user

@router.delete("/{user_id}")
async def delete_user(user_id: str = Path(...)):
    """
    Delete a user.
    """
    success = await UserController.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}

@router.put("/{user_id}/preferences", response_model=User)
async def update_preferences(user_id: str = Path(...), preferences: Dict[str, Any] = Body(...)):
    """
    Update a user's preferences.
    """
    user = await UserController.update_preferences(user_id, preferences)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user 