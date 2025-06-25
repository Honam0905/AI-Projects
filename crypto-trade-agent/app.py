# Placeholder for main execution script 
import os
import sys
from dotenv import load_dotenv
from typing import List, Dict
import asyncio
import json
import uuid

from langchain_core.messages import HumanMessage, BaseMessage
from src.combined_agent.graph import get_graph_and_store

def print_divider(title=None):
    """Print a divider with an optional title for better CLI readability."""
    width = 80
    if title:
        print(f"\n{'=' * 5} {title} {'=' * (width - len(title) - 7)}")
    else:
        print(f"\n{'=' * width}")

async def interactive_chat():
    """Run an interactive chat session with the agent in the command line."""
    print_divider("Combined Personalized ReAct Agent")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'clear' to start a new conversation.")
    print("Type 'save' to save the current conversation to a file.")
    print("Type 'debug' to toggle debug mode.")
    print_divider()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the graph and store
    graph, store = get_graph_and_store()
    
    # Initialize session parameters
    session_id = str(uuid.uuid4())
    user_id = "cli_user"
    category = "general"
    debug_mode = False
    
    config = {"configurable": {"thread_id": session_id, "user_id": user_id, "todo_category": category}}
    
    print(f"Session ID: {session_id}")
    print(f"User ID: {user_id}")
    print(f"Category: {category}")
    print_divider()
    
    # Initialize message history
    current_messages = []
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Handle special commands
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        elif user_input.lower() == "clear":
            current_messages = []
            print("Conversation cleared.")
            continue
        elif user_input.lower() == "save":
            filename = f"conversation_{session_id}.json"
            with open(filename, "w") as f:
                json.dump([msg.dict() for msg in current_messages], f, indent=2)
            print(f"Conversation saved to {filename}")
            continue
        elif user_input.lower() == "debug":
            debug_mode = not debug_mode
            print(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
            continue
        elif user_input.lower() == "help":
            print("\nAvailable commands:")
            print("  exit, quit - End the conversation")
            print("  clear - Start a new conversation")
            print("  save - Save the current conversation to a file")
            print("  debug - Toggle debug mode")
            print("  help - Show this help message")
            continue
        
        # Add the user's message to the history
        current_messages.append(HumanMessage(content=user_input))
        
        try:
            # Invoke the agent
            print("Agent is thinking...")
            
            final_state = graph.invoke({"messages": current_messages}, config=config)
            
            # Update message history
            current_messages = final_state['messages']
            
            # Print the agent's response
            ai_response = current_messages[-1].content
            print(f"\nAgent: {ai_response}")
            
            # Print debug information if enabled
            if debug_mode and len(current_messages) >= 2:
                last_ai_msg = current_messages[-1]
                if hasattr(last_ai_msg, 'tool_calls') and last_ai_msg.tool_calls:
                    print("\nDebug - Tool calls:")
                    for tc in last_ai_msg.tool_calls:
                        print(f"  - Tool: {tc.get('name')}")
                        print(f"    Args: {tc.get('args')}")
        
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("The agent encountered an error. You can continue the conversation or type 'clear' to start fresh.")
    
    print("Session ended.")

if __name__ == "__main__":
    # Check environment variables
    required_vars = ["GROQ_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in a .env file or in your environment.")
        sys.exit(1)
    
    try:
        asyncio.run(interactive_chat())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...") 