from pydantic import BaseModel
from pymongo import MongoClient
from langchain_core.tools import tool
from datetime import datetime
from bson import ObjectId
import atexit
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
import uuid

client = MongoClient("mongodb+srv://akshaymww:PE2YYynnqxZhW5qK@chatbotcluster.tavncb1.mongodb.net/?retryWrites=true&w=majority&appName=ChatBotCluster")
db = client["sttribe"]
events_collection = db["events"]
bookings_collection = db["bookings"]
internships_collection = db["internships"]
applications_collection = db["applications"]

@tool
def search_events(location:str | None = None, category:str | None = None) -> list[dict]:
    """
    Search for Technical events based on location and category.
    Args: 
        - location = The location of the event where it is taking place. It defaults to None.
        - category = The category of the event that is taking place. It defaults to None.
    Returns:
        - list[dict] = A list of events dictionary matching the search criteria.
    """
    pipeline = [
        {
            "$search":{
                "index":"eventSearchIndex",
                "text":{
                    "query":[location, category],
                    "path":["location", "category"]
                },
            }
        }
    ]
    results = list(events_collection.aggregate(pipeline))
    return results

# print(search_events.invoke({"location":"Bangalore", "category":"Machine Learning"}))

@tool
def search_internships(role: str | None = None, location: str | None = None, company: str | None = None) -> list[dict]:
    """
    Search for internships based on role, location, or company.

    Args:
        - role (str, optional): The role of the internship (e.g., "Frontend Developer Intern").
        - location (str, optional): The city where the internship is located (e.g., "Mumbai").
        - company (str, optional): The name of the company offering the internship (e.g., "ABC Solutions").

    Returns:
        - list[dict]: A list of internship dictionaries matching the search criteria.
    """

    pipeline = [
        {
            "$search": {
                "index": "internshipSearchIndex",
                "text": {
                    "query": [role, location],
                    "path": ["role", "location"],
                },
            }
        }
    ]

    results = list(internships_collection.aggregate(pipeline))
    
    return results

# print(search_internships.invoke({"location":"Bangalore", "role":"Machine Learning"}))

@tool
def book_event(student_name:str, student_email:str, event_id:str)->dict:
    """
    Book an event for a student
    Args:
        - student_name(str) : Name of the student.
        - student_email(str) : Email of the student.
        - event_id(str) : The unique id of the event ID.
    Returns:
        - dict : Confirmation message or error message
    """

    try:
        event_id = ObjectId(event_id)
    except Exception:
        return {"error":"Invalid event_id format."}

    event = events_collection.find_one({"_id":event_id})

    if not event:
        return {"error": "Event not found!"}
    
    if event["available_slots"] <= 0:
        return {"error": "No seats available for this event!"}

    booking_data = {
        "student_name": student_name,
        "student_email": student_email,
        "event_id": event_id,
        "event_name": event["name"],
        "location": event["location"],
        "category": event["category"],
        "date": event["date"],
        "booking_time": datetime.utcnow()
    }

    bookings_collection.insert_one(booking_data)

    events_collection.update_one(
        {"_id": event_id}, 
        {"$inc": {"availableSeats": -1}}
    )

    return {
        "success": True,
        "message": f"Booking confirmed for {event['name']} on {event['date']}!",
        "booking_details": booking_data
    }

# print(book_event.invoke({"student_name":"AKSHAY M", "student_email":"akshay@gmail.com", "event_id":"67ebbc36c63f17e894bff674"}))

@tool
def cancel_event(student_email:str, event_id:str)->dict:
    """
    Cancel an event booking for a student.
    
    Args:
        - student_email (str): The email of the student who booked the event.
        - event_id (str): The unique ObjectId of the booked event.
    
    Returns:
        - dict: Confirmation or error message.
    """
    try:
        event_object_id = ObjectId(event_id)
    except Exception:
        return {"error": "Invalid event ID format!"}

    booking = bookings_collection.find_one({"student_email": student_email, "event_id": event_object_id})

    if not booking:
        return {"error": "No booking found for this event!"}

    bookings_collection.delete_one({"_id": booking["_id"]})

    events_collection.update_one(
        {"_id": event_object_id},
        {"$inc": {"available_slots": 1}}
    )

    return {
        "success": True,
        "message": f"Booking for '{booking['event_name']}' has been successfully canceled.",
        "cancellation_time": datetime.utcnow()
    }

# print(cancel_event.invoke({"student_email":"akshay@gmail.com", "event_id":"67ebbc36c63f17e894bff674"}))

@tool
def apply_for_internship(student_name:str, student_email:str, internship_id:str, resume_link:str)->list[dict]:
    """
    Apply for an internship.

    Args:
        - student_name (str): Name of the student applying.
        - student_email (str): Email of the student.
        - internship_id (str): The unique ID of the internship.
        - resume_link (str): Link to the student's resume.

    Returns:
        - dict: Confirmation or error message.
    """

    try:
        internship_object_id = ObjectId(internship_id)
        print(internship_object_id)
    except Exception:
        return {"error": "Invalid internship ID format!"}

    internship = internships_collection.find_one({"_id": internship_object_id})
    if not internship:
        return {"error": "Internship not found!"}

    existing_application = applications_collection.find_one({
        "student_email": student_email,
        "internship_id": internship_object_id
    })

    if existing_application:
        return {"error": "You have already applied for this internship!"}

    application_data = {
        "student_name": student_name,
        "student_email": student_email,
        "internship_id": internship_object_id,
        "internship_role": internship["role"],
        "company": internship["company"],
        "location": internship["location"],
        "resume_link": resume_link,
        "applied_on": datetime.utcnow()
    }

    applications_collection.insert_one(application_data)

    return {
        "success": True,
        "message": f"Application submitted for {internship['role']} at {internship['company']}!",
        "application_details": application_data
    }

# print(apply_for_internship.invoke({"student_name":"AKSHAY M", "student_email":"akshay@gmail.com", "internship_id":"67ebbc81c63f17e894bff675", "resume_link":"http://example.com/resume.pdf"}))

@tool
def cancel_internship_application(student_email: str, internship_id: str) -> dict:
    """
    Cancel an internship application.

    Args:
        - student_email (str): The email of the student who applied.
        - internship_id (str): The unique ObjectId of the internship application.

    Returns:
        - dict: Confirmation or error message.
    """
    try:
        internship_object_id = ObjectId(internship_id)
    except Exception:
        return {"error": "Invalid internship ID format!"}

    application = applications_collection.find_one({
        "student_email": student_email,
        "internship_id": internship_object_id
    })

    if not application:
        return {"error": "No application found for this internship!"}

    applications_collection.delete_one({"_id": application["_id"]})

    return {
        "success": True,
        "message": f"Your application for '{application['internship_role']}' at {application['company']} has been canceled.",
        "cancellation_time": datetime.utcnow()
    }

# print(cancel_internship_application.invoke({"student_email":"akshay@gmail.com", "internship_id":"67ebbc81c63f17e894bff675"}))

def close_mongo_connection():
    client.close()

#The below to code is pretty print the messages for debugging, which also displays the errors.

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

atexit.register(close_mongo_connection)

class State(TypedDict):
    messages:Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable:Runnable):
        self.runnable = runnable
    
    def __call__(self, state:State, config:RunnableConfig):
        result = self.runnable.invoke(state)
        if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                return self(state, config)
        else:
            return{"messages":result}
        return {"messages": result}

llm = AzureChatOpenAI(
    api_key="7rpLSsf8sCoqa99RYfHc8Rrtq6NHMwNSaRNmDWe6P4jOqQLhu9NdJQQJ99BCACi0881XJ3w3AAAAACOG77t4",
    azure_endpoint="https://zuber-m7vtkl7s-japaneast.cognitiveservices.azure.com/",
    model="gpt-4o",
    api_version="2025-01-01-preview"
)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are helpful assistant for a community called Student Tribe."
            "You help students book for technical events and apply for internships, by collecting their details such as name, email, and other details as required."
            "Use the provided tools to search for events and internships, book events and internships, and cancel events and internships based on the user/student's requirement."
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\nCurrent time:{time}",    
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

part_1_safe_tools = [
    search_events,
    search_internships,
]

part_1_sensitive_tools = [
    book_event,
    cancel_event,
    apply_for_internship,
    cancel_internship_application
]

sensitive_tool_names = {t.name for t in part_1_sensitive_tools}

part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    part_1_safe_tools + part_1_sensitive_tools
)

builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("safe_tools", create_tool_node_with_fallback(part_1_safe_tools))
builder.add_node(
    "sensitive_tools", create_tool_node_with_fallback(part_1_sensitive_tools)
)


def route_tools(state: State):
    next_node = tools_condition(state)
    # If no tools are invoked, return to the user
    if next_node == END:
        return END
    ai_message = state["messages"][-1]
    # This assumes single tool calls. To handle parallel tool calling, you'd want to
    # use an ANY condition
    first_tool_call = ai_message.tool_calls[0]
    if first_tool_call["name"] in sensitive_tool_names:
        return "sensitive_tools"
    return "safe_tools"

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant", route_tools, ["safe_tools", "sensitive_tools", END]
)
builder.add_edge("safe_tools", "assistant")
builder.add_edge("sensitive_tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory, interrupt_before=["sensitive_tools"])

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id,
    }
}


_printed = set()
def stream_graph_update(user_input):
    events = part_1_graph.stream(
        {"messages": ("user", user_input)}, config, stream_mode="values"
    ) 
    for event in events:
        _print_event(event, _printed)
    snapshot = part_1_graph.get_state(config)
    while snapshot.next:
        # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
        # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
        # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
        try:
            user_input = input(
                "Do you approve of the above actions? Type 'y' to continue;"
                " otherwise, explain your requested changed.\n\n"
            )
        except:
            user_input = "y"
        if user_input.strip() == "y":
            # Just continue
            result = part_1_graph.invoke(
                None,
                config,
            )
        else:
            # Satisfy the tool invocation by
            # providing instructions on the requested changes / change of mind
            result = part_1_graph.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                            content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
                config,
            )
        snapshot = part_1_graph.get_state(config)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_update(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_update(user_input)
        break