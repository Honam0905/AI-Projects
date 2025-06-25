import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal, Optional, TypedDict, List

from langgraph.graph import MessagesState

# === Schemas from task_maistro ===

class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list
    )
    interests: list[str] = Field(
        description="Interests that the user has",
        default_factory=list
    )

class ToDo(BaseModel):
    """A task to be completed, potentially with details."""
    task: str = Field(description="The core task description.")
    context: Optional[str] = Field(
        description="Optional context or background information about the task.",
        default=None
    )
    time_to_complete: Optional[int] = Field(
        description="Estimated time to complete the task (minutes).",
        default=None
    )
    deadline: Optional[datetime] = Field(
        description="When the task needs to be completed by (if applicable)",
        default=None
    )
    solutions: list[str] = Field(
        description="List of specific, actionable solutions or steps (e.g., specific ideas, service providers, concrete options).",
        default_factory=list
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task",
        default="not started"
    )

class UpdateMemory(TypedDict):
    """Decision on what memory type to update"""
    update_type: Literal['user', 'todo', 'instructions']
    reasoning: str # Add reasoning for the update decision


# === State Definition ===

# We can stick with MessagesState for simplicity, as memory is externalized
# class AgentState(MessagesState):
#     # Potentially add other state elements if needed later
#     pass


# === Tool Schemas (Inputs) ===

class SearchInput(BaseModel):
    query: str = Field(description="The search query.")

# Add other tool input schemas as needed (e.g., ScrapeWebsiteInput)
class ScrapeWebsiteInput(BaseModel):
    url: str = Field(description="The URL of the website to scrape.") 