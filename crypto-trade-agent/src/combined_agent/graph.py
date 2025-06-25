import uuid
from datetime import datetime
import os # For environment variables
from typing import Literal

from pydantic import BaseModel, Field

# Assuming trustcall is installed and configured
# If not, the memory update logic needs adjustment
try:
    from trustcall import create_extractor
except ImportError:
    print("Warning: 'trustcall' not found. Memory update features will be limited.")
    # Define dummy create_extractor if trustcall isn't available
    def create_extractor(*args, **kwargs):
        raise NotImplementedError("Trustcall is not installed")

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    merge_message_runs,
)
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI # Keep OpenAI for Trustcall compatibility if needed
from langgraph.checkpoint.memory import MemorySaver # Use MemorySaver for persistence
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore # Or use a persistent store
from langgraph.prebuilt import ToolNode, tools_condition

from .schemas import Profile, ToDo, UpdateMemory
from .tools import STANDARD_TOOLS

# --- Configuration ---

# Simple config for now, load from env vars
# Consider a Pydantic model like in examples for more complex config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Needed if using OpenAI/Trustcall
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Placeholder for graph definition

# --- Utilities (Adapted from task_maistro) ---

# Spy class remains useful for debugging Trustcall if used
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model" and r.outputs.get("generations"):
                 try:
                     # Handle potential variations in output structure
                     tool_calls = r.outputs["generations"][0][0]["message"]["kwargs"].get("tool_calls")
                     if tool_calls:
                         self.called_tools.append(tool_calls)
                 except (IndexError, KeyError, TypeError):
                     # Ignore if structure doesn't match expected format
                     pass


def extract_tool_info(tool_calls, schema_name="Memory"):
    changes = []
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                if call['args'].get('patches'): # Use .get for safety
                    changes.append({
                        'type': 'update',
                        'doc_id': call['args'].get('json_doc_id', 'N/A'),
                        'planned_edits': call['args'].get('planned_edits', 'N/A'),
                        'value': call['args']['patches'][0].get('value', 'N/A')
                    })
                else:
                    changes.append({
                        'type': 'no_update',
                        'doc_id': call['args'].get('json_doc_id', 'N/A'),
                        'planned_edits': call['args'].get('planned_edits', 'N/A')
                    })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call.get('args', {})
                })

    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        elif change['type'] == 'no_update':
            result_parts.append(
                f"Document {change['doc_id']} unchanged:\n"
                f"{change['planned_edits']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )
    return "\n\n".join(result_parts)

# --- Model Initialization ---

# Simple model getter for now
# TODO: Make model name configurable
def get_model(config: RunnableConfig = None):
    # Configurable elements could be passed via RunnableConfig if needed
    # For now, relies on global API keys
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    return ChatGroq(model="llama3-70b-8192", temperature=0, groq_api_key=GROQ_API_KEY)

# Trustcall requires an OpenAI compatible model by default
# def get_openai_model(config: RunnableConfig = None):
#     if not OPENAI_API_KEY:
#         raise ValueError("OPENAI_API_KEY environment variable not set for Trustcall.")
#     return ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)


# --- Trustcall Extractors (if available) ---
try:
    # Using Groq for Trustcall might require compatibility adjustments or specific Trustcall versions
    # Sticking to OpenAI for extractor definition as per typical Trustcall examples
    # trustcall_model = get_openai_model()
    # Using Groq directly - experimental, may not work perfectly with trustcall
    trustcall_model = get_model()

    profile_extractor = create_extractor(
        trustcall_model,
        tools=[Profile],
        tool_choice="Profile",
    )

    todo_extractor = create_extractor(
        trustcall_model,
        tools=[ToDo],
        tool_choice="ToDo",
        enable_inserts=True # Allow creating new ToDos
    )
    TRUSTCALL_ENABLED = True
except (ImportError, NotImplementedError, ValueError) as e:
    print(f"Trustcall setup failed: {e}. Memory update nodes will be disabled.")
    profile_extractor = None
    todo_extractor = None
    TRUSTCALL_ENABLED = False


# --- Prompts ---

AGENT_SYSTEM_MESSAGE = """You are a helpful assistant that can use tools and manage long-term memory.

Your capabilities include:
- Engaging in conversation.
- Using tools like web search (`tavily_search`) and website scraping (`scrape_website`) to find information.
- Maintaining a long-term memory about the user's profile, tasks (ToDo list), and interaction preferences (instructions).

Memory Management:
You have access to a special tool called `UpdateMemory`. When the conversation reveals information suitable for long-term storage, you MUST decide whether to update the 'user' profile, the 'todo' list, or the 'instructions'. Provide a brief 'reasoning' field explaining your choice.
- User Profile: Store facts about the user (name, location, job, interests, connections). Update using `UpdateMemory` with `update_type='user'`. **Do not** explicitly tell the user you've updated their profile.
- ToDo List: Track tasks the user mentions or implies. Use `UpdateMemory` with `update_type='todo'`. **Do** tell the user when you update the todo list (e.g., "Okay, I've added 'task description' to your ToDo list."). Err on the side of adding tasks.
- Instructions: Note down user preferences about *how* you should interact or manage the ToDo list (e.g., "Always ask before adding a deadline", "Summarize search results concisely"). Use `UpdateMemory` with `update_type='instructions'`. **Do not** tell the user you've updated instructions.

Current Memory State (provided for context):
<user_profile>
{user_profile}
</user_profile>

<todo>
{todo}
</todo>

<instructions>
{instructions}
</instructions>

Tool Usage:
When you need external information or need to perform an action defined by a tool, formulate your thought process and then call the appropriate tool(s). You can call multiple tools in parallel if needed. If you call `UpdateMemory`, that should typically be your only action in that turn unless specifically requested otherwise.

Respond naturally to the user after completing your actions (tool use or memory update). If no tool or memory update is needed, just provide a conversational response.

System Time: {time}
"""

TRUSTCALL_INSTRUCTION = """Reflect on the following interaction.

Use the provided tools to retain any necessary memories about the user.

Use parallel tool calling to handle updates and insertions simultaneously if applicable for the schema.

System Time: {time}"""

CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to manage ToDo list items or interact with the user. Incorporate any feedback provided. Use the tool provided to save the updated instructions.

Your current instructions are:
<current_instructions>
{current_instructions}
</current_instructions>"""


# --- Graph Nodes ---

def get_memory_store() -> BaseStore:
    # Replace with a persistent store (e.g., Redis, Postgres) for long-term memory
    # For now, using InMemoryStore which resets on restart
    return InMemoryStore()

# Global store instance (consider dependency injection for better testability)
store = get_memory_store()

# Placeholder for configuration object if needed later
# class AgentConfig(BaseModel):
#     user_id: str = "default_user"
#     todo_category: str = "general"
#     # Add other config fields

# Dummy RunnableConfig for nodes that require it but don't use it heavily here
dummy_config: RunnableConfig = {"configurable": {}}

def agent_reasoner(state: MessagesState):
    """Core reasoning node: Loads memory, formats prompt, calls LLM."""
    # Simplified config - assume single user/category for now
    # In a real app, these would come from the request/session config
    user_id = state["config"].get("user_id", "default_user")
    todo_category = state["config"].get("todo_category", "general")


    # Retrieve memories (similar to task_maistro)
    profile_namespace = ("profile", todo_category, user_id)
    memories = store.search(query=None, namespace=profile_namespace) # Search all in namespace
    user_profile = memories[0].value if memories else {} # Assume single profile doc

    todo_namespace = ("todo", todo_category, user_id)
    memories = store.search(query=None, namespace=todo_namespace)
    todos = "\n".join(f"- {mem.value.get('task', 'N/A')} (Status: {mem.value.get('status', 'N/A')})" for mem in memories)

    instructions_namespace = ("instructions", todo_category, user_id)
    memories = store.search(query=None, namespace=instructions_namespace)
    instructions = memories[0].value.get("memory", "") if memories else ""

    system_msg_content = AGENT_SYSTEM_MESSAGE.format(
        user_profile=user_profile if user_profile else "(empty)",
        todo=todos if todos else "(empty)",
        instructions=instructions if instructions else "(empty)",
        time=datetime.now().isoformat()
    )
    system_msg = SystemMessage(content=system_msg_content)

    model = get_model()
    # Bind *all* tools: memory update + standard tools
    # Let the LLM decide which one to call
    all_tools = STANDARD_TOOLS + ([UpdateMemory] if TRUSTCALL_ENABLED else [])
    # Filter out UpdateMemory if trustcall isn't enabled
    usable_tools = [t for t in all_tools if t is not UpdateMemory or TRUSTCALL_ENABLED]

    if not usable_tools:
         # If no tools available (e.g. Trustcall disabled and no standard tools)
         bound_model = model
    else:
         # Use parallel calls for standard tools, but likely UpdateMemory is single call
         # Let the model decide on parallel calls based on the tools provided
         bound_model = model.bind_tools(usable_tools) # Let model decide on parallel calls

    # Filter out placeholder messages if any, ensure alternation
    # simplified_history = [msg for msg in state["messages"] if msg.content] # Basic filter
    # TODO: Implement robust history management (summarization, alternation enforcement)

    response = bound_model.invoke([system_msg] + state["messages"])

    return {"messages": [response]}


# Action node using ToolNode for standard tools
action_node = ToolNode(STANDARD_TOOLS)


# --- Memory Update Nodes (Adapted from task_maistro, require TRUSTCALL_ENABLED) ---

def update_profile(state: MessagesState):
    if not TRUSTCALL_ENABLED:
        return {"messages": [ToolMessage(content="Profile update skipped: Trustcall not enabled.", tool_call_id=state['messages'][-1].tool_calls[0]['id'])]}

    user_id = state["config"].get("user_id", "default_user")
    todo_category = state["config"].get("todo_category", "general")
    namespace = ("profile", todo_category, user_id)

    existing_items = store.search(query=None, namespace=namespace)
    tool_name = "Profile"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items] if existing_items else None)

    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    # Use message history *before* the UpdateMemory tool call AI message
    history_for_update = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    try:
        result = profile_extractor.invoke({"messages": history_for_update, "existing": existing_memories})
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            store.put(namespace=namespace,
                      key=rmeta.get("json_doc_id", str(uuid.uuid4())),
                      value=r.model_dump(mode="json"))
        content = "Profile potentially updated (details omitted for privacy)."
    except Exception as e:
        content = f"Error updating profile: {e}"

    tool_calls = state['messages'][-1].tool_calls
    # Ensure tool_call_id is correctly extracted
    tool_call_id = tool_calls[0]['id'] if tool_calls and isinstance(tool_calls[0], dict) else None
    if not tool_call_id:
         print("Warning: Could not find tool_call_id for profile update response.")
         # Decide how to handle this - maybe add a generic message without id?
         # For now, let it potentially fail if id is strictly required downstream

    return {"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}


def update_todos(state: MessagesState):
    if not TRUSTCALL_ENABLED:
         return {"messages": [ToolMessage(content="ToDo update skipped: Trustcall not enabled.", tool_call_id=state['messages'][-1].tool_calls[0]['id'])]}

    user_id = state["config"].get("user_id", "default_user")
    todo_category = state["config"].get("todo_category", "general")
    namespace = ("todo", todo_category, user_id)

    existing_items = store.search(query=None, namespace=namespace)
    tool_name = "ToDo"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items] if existing_items else None)

    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    history_for_update = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    spy = Spy()
    bound_todo_extractor = todo_extractor.with_listeners(on_end=spy)

    try:
        result = bound_todo_extractor.invoke({"messages": history_for_update, "existing": existing_memories})
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            store.put(namespace=namespace,
                      key=rmeta.get("json_doc_id", str(uuid.uuid4())),
                      value=r.model_dump(mode="json"))
        # Extract changes for user feedback
        todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
        content = f"ToDo list updated:\n{todo_update_msg}"
    except Exception as e:
        content = f"Error updating ToDos: {e}"

    tool_calls = state['messages'][-1].tool_calls
    tool_call_id = tool_calls[0]['id'] if tool_calls and isinstance(tool_calls[0], dict) else None
    if not tool_call_id:
         print("Warning: Could not find tool_call_id for todo update response.")

    return {"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}

def update_instructions(state: MessagesState):
     if not TRUSTCALL_ENABLED: # Instruction update also relies on LLM call, less on Trustcall itself
         # We could potentially implement simple instruction update without trustcall
         tool_calls = state['messages'][-1].tool_calls
         tool_call_id = tool_calls[0]['id'] if tool_calls and isinstance(tool_calls[0], dict) else None
         return {"messages": [ToolMessage(content="Instruction update skipped: Trustcall not enabled.", tool_call_id=tool_call_id)]}

     user_id = state["config"].get("user_id", "default_user")
     todo_category = state["config"].get("todo_category", "general")
     namespace = ("instructions", todo_category, user_id)

     # Use a fixed key for instructions for simplicity
     instruction_key = "user_instructions"
     existing_memory = store.search(query=instruction_key, namespace=namespace) # Search by key

     current_instructions_content = existing_memory[0].value.get("memory", "") if existing_memory else ""

     system_msg_content = CREATE_INSTRUCTIONS.format(current_instructions=current_instructions_content)
     # Use history before the UpdateMemory call + a final instruction
     history_for_update = state['messages'][:-1] + [HumanMessage(content="Based on our conversation, please update the instructions for future interactions.")]

     try:
         model = get_model() # Use the main model for instruction generation
         new_memory = model.invoke([SystemMessage(content=system_msg_content)] + history_for_update)

         # Overwrite the existing memory in the store
         store.put(namespace=namespace, key=instruction_key, value={"memory": new_memory.content})
         content = "Instructions potentially updated."
     except Exception as e:
         content = f"Error updating instructions: {e}"

     tool_calls = state['messages'][-1].tool_calls
     tool_call_id = tool_calls[0]['id'] if tool_calls and isinstance(tool_calls[0], dict) else None
     if not tool_call_id:
         print("Warning: Could not find tool_call_id for instruction update response.")

     return {"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}


# --- Conditional Edge / Routing ---

def route_message(state: MessagesState) -> Literal[
    "action_node", "update_profile", "update_todos", "update_instructions", "__end__"
]:
    """Routes based on the last message's tool calls."""
    last_message = state['messages'][-1]

    if not isinstance(last_message, AIMessage):
         # Should not happen in normal flow, but maybe return to agent or end?
         print("Warning: Expected AIMessage at routing, got", type(last_message))
         return "__end__" # Or agent_reasoner?

    if not last_message.tool_calls:
        return "__end__"

    # Check for UpdateMemory tool first
    # Ensure tool_calls are dicts before accessing 'name'
    update_memory_call = next((
        tc for tc in last_message.tool_calls
        if isinstance(tc, dict) and tc.get("name") == "UpdateMemory"
    ), None)


    if update_memory_call:
        if not TRUSTCALL_ENABLED:
             # If UpdateMemory was called but Trustcall is off, we can't process it.
             # We need to add a ToolMessage indicating the skip.
             print("UpdateMemory called but Trustcall disabled. Adding skip message.")
             # This requires modifying state which isn't ideal in a pure routing function.
             # A better approach might be a dedicated 'skip_update' node.
             # For simplicity here, just route to END. The AI message remains.
             return "__end__"

        update_type = update_memory_call.get('args', {}).get('update_type')
        if update_type == "user":
            return "update_profile"
        elif update_type == "todo":
            return "update_todos"
        elif update_type == "instructions":
            return "update_instructions"
        else:
             # Invalid UpdateMemory args? Route to end or error?
             print(f"Warning: Unknown update_type in UpdateMemory call: {update_type}")
             return "__end__" # Or agent_reasoner?
    else:
        # If other tools (search, scrape) were called
        return "action_node"


# --- Build the Graph ---

# Using MessagesState for simplicity
builder = StateGraph(MessagesState) # Can add config_schema if needed later

# Add nodes
builder.add_node("agent_reasoner", agent_reasoner)
builder.add_node("action_node", action_node)

# Add memory update nodes only if Trustcall is enabled
if TRUSTCALL_ENABLED:
    builder.add_node("update_profile", update_profile)
    builder.add_node("update_todos", update_todos)
    builder.add_node("update_instructions", update_instructions)
else:
     print("Memory update nodes (update_profile, update_todos, update_instructions) are disabled.")


# Define edges
builder.add_edge(START, "agent_reasoner")

# Conditional routing after agent makes a decision
# Map outputs of route_message to node names. Handle disabled Trustcall.
def route_map(route: str):
    if route == "action_node": return "action_node"
    if route == "__end__": return END
    if not TRUSTCALL_ENABLED:
         print(f"Routing: Trustcall disabled, rerouting {route} to agent_reasoner")
         # If trustcall is disabled but a memory update was requested, just loop back
         # A skip message should ideally be added by the update node itself.
         return "agent_reasoner"
    if route == "update_profile": return "update_profile"
    if route == "update_todos": return "update_todos"
    if route == "update_instructions": return "update_instructions"
    # Fallback
    print(f"Warning: Unhandled route '{route}' in route_map. Routing to agent_reasoner.")
    return "agent_reasoner"

builder.add_conditional_edges("agent_reasoner", route_message, route_map)

# Edges back to the agent after actions/updates
builder.add_edge("action_node", "agent_reasoner")
if TRUSTCALL_ENABLED:
    builder.add_edge("update_profile", "agent_reasoner")
    builder.add_edge("update_todos", "agent_reasoner")
    builder.add_edge("update_instructions", "agent_reasoner")

# Compile the graph
graph = builder.compile(
     # Use MemorySaver for persistence across runs if store is persistent
     # checkpointer=MemorySaver(store=store)
     # Using in-memory store, so no checkpointer needed unless explicitly managing checkpoints
)
graph.name = "CombinedPersonalizedReActAgent"

print("Combined agent graph compiled.")
if not TRUSTCALL_ENABLED:
     print("NOTE: Trustcall features are disabled. Memory will not be updated.")

# --- Helper function to get the compiled graph and store ---
def get_graph_and_store():
     # Load environment variables if not already loaded (e.g., using python-dotenv)
     # from dotenv import load_dotenv
     # load_dotenv()
     # Perform checks for API keys here before returning graph?
     if not GROQ_API_KEY: print("Warning: GROQ_API_KEY not set.")
     if not TAVILY_API_KEY: print("Warning: TAVILY_API_KEY not set for search tool.")
     # if TRUSTCALL_ENABLED and not OPENAI_API_KEY: print("Warning: OPENAI_API_KEY not set, needed for Trustcall.")

     # Return both graph and the shared store instance
     return graph, store

# You can add a simple test invocation here if needed
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    print("Running simple test...")
    app, _ = get_graph_and_store() # Get graph, ignore store for this test
    config = {"configurable": {"thread_id": "test-thread-main", "user_id": "test_user", "todo_category": "main_test"}}

    inputs = [
        HumanMessage(content="Hi! My name is Bob and I live in California. Can you search the web for the weather there?"),
        HumanMessage(content="What is my name?"),
        HumanMessage(content="Please add 'buy milk' to my todo list."),
        HumanMessage(content="What's on my todo list?"),
    ]

    current_messages = []
    for i, human_input in enumerate(inputs):
        print(f"\n--- Invocation {i+1} --- User: {human_input.content}")
        current_messages.append(human_input)
        final_state = app.invoke({"messages": current_messages}, config=config)
        # Update message history for next turn
        current_messages = final_state['messages']
        print(f"AI: {current_messages[-1].content}")
        # Optionally print tool calls/results if needed for debugging
        # print("DEBUG: Full final state messages:", current_messages)