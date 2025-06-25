import logging
import os
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query, Depends, Path
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field

from src.combined_agent.graph import get_graph_and_store
from src.combined_agent.schemas import Profile, ToDo

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')
log_file = 'logs/agent-api-service.log'
max_log_size = 1 * 1024 * 1024  # 1MB
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s : %(message)s')

file_handler = RotatingFileHandler(log_file, maxBytes=max_log_size, backupCount=3)
file_handler.setFormatter(formatter)
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, logging.StreamHandler()],
    format='%(asctime)s %(levelname)s %(name)s : %(message)s'
)

logger = logging.getLogger("agent-api")

# Load environment variables
load_dotenv()
logger.info("Loading environment variables and initializing graph")

# Get the graph and store
_, store = get_graph_and_store()
logger.info("Agent graph and memory store initialized")

# Initialize FastAPI app
app = FastAPI(
    title="Agent Memory API",
    description="API for managing Combined Agent memory components",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

def get_namespace_params(
    user_id: str = Query("api_user", description="User ID for memory segregation"),
    category: str = Query("general", description="Category for memory segregation")
) -> tuple:
    """Helper to get namespace parameters for API endpoints."""
    return user_id, category

# --- API Models ---

# Profile models
class ProfileUpdate(BaseModel):
    """Model for updating user profile."""
    name: Optional[str] = Field(None, description="User's name")
    location: Optional[str] = Field(None, description="User's location")
    job: Optional[str] = Field(None, description="User's job")
    connections: Optional[List[str]] = Field(None, description="User's connections")
    interests: Optional[List[str]] = Field(None, description="User's interests")

class ProfileResponse(BaseModel):
    """Response model for user profile."""
    id: str = Field(..., description="The profile identifier")
    name: Optional[str] = Field(None, description="User's name")
    location: Optional[str] = Field(None, description="User's location")
    job: Optional[str] = Field(None, description="User's job")
    connections: List[str] = Field(..., description="User's connections")
    interests: List[str] = Field(..., description="User's interests")

# ToDo models
class TodoCreate(BaseModel):
    """Model for creating a new ToDo item."""
    task: str = Field(..., description="The task description")
    context: Optional[str] = Field(None, description="Optional context for the task")
    time_to_complete: Optional[int] = Field(None, description="Time to complete in minutes")
    deadline: Optional[datetime] = Field(None, description="Deadline for the task")
    solutions: List[str] = Field(default_factory=list, description="Potential solutions")
    status: str = Field("not started", description="Current status")

class TodoUpdate(BaseModel):
    """Model for updating a ToDo item."""
    task: Optional[str] = Field(None, description="The task description")
    context: Optional[str] = Field(None, description="Optional context for the task")
    time_to_complete: Optional[int] = Field(None, description="Time to complete in minutes")
    deadline: Optional[datetime] = Field(None, description="Deadline for the task")
    solutions: Optional[List[str]] = Field(None, description="Potential solutions")
    status: Optional[str] = Field(None, description="Current status")

class TodoResponse(BaseModel):
    """Response model for ToDo items."""
    id: str = Field(..., description="The ToDo identifier")
    task: str = Field(..., description="The task description")
    context: Optional[str] = Field(None, description="Optional context for the task")
    time_to_complete: Optional[int] = Field(None, description="Time to complete in minutes")
    deadline: Optional[datetime] = Field(None, description="Deadline for the task")
    solutions: List[str] = Field(..., description="Potential solutions")
    status: str = Field(..., description="Current status")

# Instructions models
class InstructionUpdate(BaseModel):
    """Model for updating agent instructions."""
    content: str = Field(..., description="The instruction content")

class InstructionResponse(BaseModel):
    """Response model for agent instructions."""
    key: str = Field(..., description="The identifier key")
    content: str = Field(..., description="The instruction content")

# --- Formatters ---

def format_profile_response(item_key: str, item_value: Dict[str, Any]) -> ProfileResponse:
    """Convert store profile item to API response format."""
    return ProfileResponse(
        id=item_key,
        name=item_value.get("name"),
        location=item_value.get("location"),
        job=item_value.get("job"),
        connections=item_value.get("connections", []),
        interests=item_value.get("interests", [])
    )

def format_todo_response(item_key: str, item_value: Dict[str, Any]) -> TodoResponse:
    """Convert store todo item to API response format."""
    return TodoResponse(
        id=item_key,
        task=item_value.get("task", ""),
        context=item_value.get("context"),
        time_to_complete=item_value.get("time_to_complete"),
        deadline=item_value.get("deadline"),
        solutions=item_value.get("solutions", []),
        status=item_value.get("status", "not started")
    )

# --- API Routes ---

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "Combined Agent Memory API",
        "version": "0.1.0",
        "documentation": "/api/docs",
        "endpoints": {
            "profiles": "/api/profiles/",
            "todos": "/api/todos/",
            "instructions": "/api/instructions/"
        }
    }

# --- Profile Management Endpoints ---

@app.get("/api/profiles/", response_model=List[ProfileResponse], tags=["Profile"])
async def get_profiles(namespace_params: tuple = Depends(get_namespace_params)):
    """Get all user profiles."""
    user_id, category = namespace_params
    namespace = ("profile", category, user_id)
    logger.info(f"Fetching all profiles for user {user_id}, category {category}")
    
    try:
        items = store.search(query=None, namespace=namespace, k=100)
        return [format_profile_response(item.key, item.value) for item in items]
    except Exception as e:
        logger.error(f"Error retrieving profiles: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving profiles: {str(e)}")

@app.get("/api/profiles/{profile_id}", response_model=ProfileResponse, tags=["Profile"])
async def get_profile(
    profile_id: str = Path(..., description="Profile ID to retrieve"),
    namespace_params: tuple = Depends(get_namespace_params)
):
    """Get a specific user profile by ID."""
    user_id, category = namespace_params
    namespace = ("profile", category, user_id)
    logger.info(f"Fetching profile {profile_id} for user {user_id}, category {category}")
    
    try:
        items = store.search(query=profile_id, namespace=namespace, k=1)
        if not items:
            logger.warning(f"Profile not found: {profile_id}")
            raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
        return format_profile_response(items[0].key, items[0].value)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving profile: {str(e)}")

@app.post("/api/profiles/", response_model=ProfileResponse, tags=["Profile"])
async def create_profile(
    profile_data: ProfileUpdate,
    namespace_params: tuple = Depends(get_namespace_params)
):
    """Create a new user profile."""
    user_id, category = namespace_params
    namespace = ("profile", category, user_id)
    logger.info(f"Creating profile for user {user_id}, category {category}")
    
    # Check if profile already exists
    existing = store.search(query=None, namespace=namespace, k=1)
    if existing:
        logger.warning(f"Profile already exists for user {user_id}")
        raise HTTPException(status_code=400, detail="A profile already exists for this user. Use PUT to update.")
    
    try:
        # Create profile
        profile_id = str(uuid.uuid4())
        profile_data_dict = profile_data.model_dump(exclude_unset=True)
        
        # Ensure lists are initialized
        if "connections" not in profile_data_dict:
            profile_data_dict["connections"] = []
        if "interests" not in profile_data_dict:
            profile_data_dict["interests"] = []
        
        # Store in memory
        store.put(namespace=namespace, key=profile_id, value=profile_data_dict)
        logger.info(f"Profile created successfully with ID {profile_id}")
        return format_profile_response(profile_id, profile_data_dict)
    except Exception as e:
        logger.error(f"Error creating profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating profile: {str(e)}")

@app.put("/api/profiles/{profile_id}", response_model=ProfileResponse, tags=["Profile"])
async def update_profile(
    profile_data: ProfileUpdate,
    profile_id: str = Path(..., description="Profile ID to update"),
    namespace_params: tuple = Depends(get_namespace_params)
):
    """Update an existing user profile."""
    user_id, category = namespace_params
    namespace = ("profile", category, user_id)
    logger.info(f"Updating profile {profile_id} for user {user_id}, category {category}")
    
    # Check if profile exists
    items = store.search(query=profile_id, namespace=namespace, k=1)
    if not items:
        logger.warning(f"Profile not found: {profile_id}")
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
    
    try:
        # Get existing profile
        existing_profile = items[0].value
        
        # Update with new data (only provided fields)
        profile_update = profile_data.model_dump(exclude_unset=True)
        for key, value in profile_update.items():
            existing_profile[key] = value
        
        # Store updated profile
        store.put(namespace=namespace, key=profile_id, value=existing_profile)
        logger.info(f"Profile {profile_id} updated successfully")
        return format_profile_response(profile_id, existing_profile)
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating profile: {str(e)}")

@app.delete("/api/profiles/{profile_id}", tags=["Profile"])
async def delete_profile(
    profile_id: str = Path(..., description="Profile ID to delete"),
    namespace_params: tuple = Depends(get_namespace_params)
):
    """Delete a user profile."""
    user_id, category = namespace_params
    namespace = ("profile", category, user_id)
    logger.info(f"Deleting profile {profile_id} for user {user_id}, category {category}")
    
    try:
        # Check if profile exists
        items = store.search(query=profile_id, namespace=namespace, k=1)
        if not items:
            logger.warning(f"Profile not found: {profile_id}")
            raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
        
        # Delete the profile
        store.delete(namespace=namespace, key=profile_id)
        logger.info(f"Profile {profile_id} deleted successfully")
        return {"message": f"Profile {profile_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting profile: {str(e)}")

# --- ToDo Management Endpoints ---

@app.get("/api/todos/", response_model=List[TodoResponse], tags=["ToDo"])
async def get_todos(
    status: Optional[str] = Query(None, description="Filter by status"),
    namespace_params: tuple = Depends(get_namespace_params)
):
    """Get all todo items, optionally filtered by status."""
    user_id, category = namespace_params
    namespace = ("todo", category, user_id)
    logger.info(f"Fetching todos for user {user_id}, category {category}, status filter: {status}")
    
    try:
        items = store.search(query=None, namespace=namespace, k=100)
        
        # Filter by status if provided
        if status:
            filtered_items = [item for item in items if item.value.get("status") == status]
        else:
            filtered_items = items
            
        return [format_todo_response(item.key, item.value) for item in filtered_items]
    except Exception as e:
        logger.error(f"Error retrieving todos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving todos: {str(e)}")

@app.get("/api/todos/{todo_id}", response_model=TodoResponse, tags=["ToDo"])
async def get_todo(
    todo_id: str = Path(..., description="ToDo ID to retrieve"),
    namespace_params: tuple = Depends(get_namespace_params)
):
    """Get a specific todo item by ID."""
    user_id, category = namespace_params
    namespace = ("todo", category, user_id)
    logger.info(f"Fetching todo {todo_id} for user {user_id}, category {category}")
    
    try:
        items = store.search(query=todo_id, namespace=namespace, k=1)
        if not items:
            logger.warning(f"ToDo not found: {todo_id}")
            raise HTTPException(status_code=404, detail=f"ToDo not found: {todo_id}")
        return format_todo_response(items[0].key, items[0].value)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving todo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving todo: {str(e)}")

@app.post("/api/todos/", response_model=TodoResponse, tags=["ToDo"])
async def create_todo(
    todo_data: TodoCreate,
    namespace_params: tuple = Depends(get_namespace_params)
):
    """Create a new todo item."""
    user_id, category = namespace_params
    namespace = ("todo", category, user_id)
    logger.info(f"Creating todo for user {user_id}, category {category}")
    
    try:
        todo_id = str(uuid.uuid4())
        todo_dict = todo_data.model_dump()
        
        # Store in memory
        store.put(namespace=namespace, key=todo_id, value=todo_dict)
        logger.info(f"Todo created successfully with ID {todo_id}")
        return format_todo_response(todo_id, todo_dict)
    except Exception as e:
        logger.error(f"Error creating todo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating todo: {str(e)}")

@app.put("/api/todos/{todo_id}", response_model=TodoResponse, tags=["ToDo"])
async def update_todo(
    todo_data: TodoUpdate,
    todo_id: str = Path(..., description="ToDo ID to update"),
    namespace_params: tuple = Depends(get_namespace_params)
):
    """Update an existing todo item."""
    user_id, category = namespace_params
    namespace = ("todo", category, user_id)
    logger.info(f"Updating todo {todo_id} for user {user_id}, category {category}")
    
    # Check if todo exists
    items = store.search(query=todo_id, namespace=namespace, k=1)
    if not items:
        logger.warning(f"ToDo not found: {todo_id}")
        raise HTTPException(status_code=404, detail=f"ToDo not found: {todo_id}")
    
    try:
        # Get existing todo
        existing_todo = items[0].value
        
        # Update with new data (only provided fields)
        todo_update = todo_data.model_dump(exclude_unset=True)
        for key, value in todo_update.items():
            existing_todo[key] = value
        
        # Store updated todo
        store.put(namespace=namespace, key=todo_id, value=existing_todo)
        logger.info(f"Todo {todo_id} updated successfully")
        return format_todo_response(todo_id, existing_todo)
    except Exception as e:
        logger.error(f"Error updating todo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating todo: {str(e)}")

@app.delete("/api/todos/{todo_id}", tags=["ToDo"])
async def delete_todo(
    todo_id: str = Path(..., description="ToDo ID to delete"),
    namespace_params: tuple = Depends(get_namespace_params)
):
    """Delete a todo item."""
    user_id, category = namespace_params
    namespace = ("todo", category, user_id)
    logger.info(f"Deleting todo {todo_id} for user {user_id}, category {category}")
    
    try:
        # Check if todo exists
        items = store.search(query=todo_id, namespace=namespace, k=1)
        if not items:
            logger.warning(f"ToDo not found: {todo_id}")
            raise HTTPException(status_code=404, detail=f"ToDo not found: {todo_id}")
        
        # Delete the todo
        store.delete(namespace=namespace, key=todo_id)
        logger.info(f"Todo {todo_id} deleted successfully")
        return {"message": f"ToDo {todo_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting todo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting todo: {str(e)}")

# --- Instructions Management Endpoints ---

@app.get("/api/instructions/", response_model=InstructionResponse, tags=["Instructions"])
async def get_instructions(namespace_params: tuple = Depends(get_namespace_params)):
    """Get the agent's instructions for a user."""
    user_id, category = namespace_params
    namespace = ("instructions", category, user_id)
    instruction_key = "user_instructions"
    logger.info(f"Fetching instructions for user {user_id}, category {category}")
    
    try:
        items = store.search(query=instruction_key, namespace=namespace, k=1)
        if not items:
            return InstructionResponse(key=instruction_key, content="")
        
        # Return the content from memory.content field if available
        content = items[0].value.get("memory", "") if isinstance(items[0].value, dict) else str(items[0].value)
        return InstructionResponse(key=instruction_key, content=content)
    except Exception as e:
        logger.error(f"Error retrieving instructions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving instructions: {str(e)}")

@app.put("/api/instructions/", response_model=InstructionResponse, tags=["Instructions"])
async def update_instructions(
    instruction_data: InstructionUpdate,
    namespace_params: tuple = Depends(get_namespace_params)
):
    """Update the agent's instructions for a user."""
    user_id, category = namespace_params
    namespace = ("instructions", category, user_id)
    instruction_key = "user_instructions"
    logger.info(f"Updating instructions for user {user_id}, category {category}")
    
    try:
        # Format for storage - match the expected format in graph.py
        instruction_value = {"memory": instruction_data.content}
        
        # Store in memory
        store.put(namespace=namespace, key=instruction_key, value=instruction_value)
        logger.info(f"Instructions updated successfully")
        return InstructionResponse(key=instruction_key, content=instruction_data.content)
    except Exception as e:
        logger.error(f"Error updating instructions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating instructions: {str(e)}")

@app.delete("/api/instructions/", tags=["Instructions"])
async def delete_instructions(namespace_params: tuple = Depends(get_namespace_params)):
    """Delete the agent's instructions for a user."""
    user_id, category = namespace_params
    namespace = ("instructions", category, user_id)
    instruction_key = "user_instructions"
    logger.info(f"Deleting instructions for user {user_id}, category {category}")
    
    try:
        # Check if instructions exist
        items = store.search(query=instruction_key, namespace=namespace, k=1)
        if not items:
            logger.warning("No instructions found to delete")
            return {"message": "No instructions found to delete"}
        
        # Delete the instructions
        store.delete(namespace=namespace, key=instruction_key)
        logger.info("Instructions deleted successfully")
        return {"message": "Instructions deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting instructions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting instructions: {str(e)}")

# Run the API server
if __name__ == "__main__":
    # Check environment variables
    required_vars = ["GROQ_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    # Run the app
    logger.info("Starting API server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)