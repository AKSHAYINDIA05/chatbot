from langchain_openai import AzureChatOpenAI
from langgraph.graph import START, END, StateGraph
from langchain_core.tools import tool
import re
from langgraph.graph.message import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from typing import Optional
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

memory = MemorySaver()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
    model="gpt-4o",
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

client = MongoClient(os.getenv("MONGO_URL"))
db = client["sttribe"]
events_collection = db["events"]
bookings_collection = db["bookings"]
internships_collection = db["internships"]
applications_collection = db["applications"]

@tool
def search_events(location:Optional[str] = None, category:Optional[str] = None) -> list[dict]:
    """
    Search for Technical events based on location and category.
    Args: 
        - location = The location of the event where it is taking place. It defaults to None.
        - category = The category of the event that is taking place. It defaults to None.
    Returns:
        - list[dict] = A list of events dictionary matching the search criteria.
    """
    query_terms = []
    paths = []

    if location:
        query_terms.append(location)
        paths.append("location")
    
    if category:
        query_terms.append(category)
        paths.append("category")
    
    if not query_terms:
        return list(events_collection.find({}))
    
    pipeline = [
        {
            "$search":{
                "index":"eventSearchIndex",
                "text":{
                    "query":query_terms,
                    "path":paths
                },
            }
        }
    ]
    results = list(events_collection.aggregate(pipeline))
    return results

# print(search_events.invoke({"location":"Bangalore", "category":"Machine Learning"}))

@tool
def search_internships(role: Optional[str] = None, location: Optional[str] = None, company: Optional[str] = None) -> list[dict]:
    """
    Search for internships based on role, location, or company.

    Args:
        - role (str, optional): The role of the internship (e.g., "Frontend Developer Intern").
        - location (str, optional): The city where the internship is located (e.g., "Mumbai").
        - company (str, optional): The name of the company offering the internship (e.g., "ABC Solutions").

    Returns:
        - list[dict]: A list of internship dictionaries matching the search criteria.
    """
    query_terms = []
    paths = []

    if role:
        query_terms.append(role)
        paths.append("role")
    
    if location:
        query_terms.append(location)
        paths.append("location")

    if company:
        query_terms.append(company)
        paths.append("company")

    if not query_terms:
        return list(internships_collection.find({}))
    pipeline = [
        {
            "$search": {
                "index": "internshipSearchIndex",
                "text": {
                    "query": query_terms,
                    "path": paths,
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
        "booking_time": datetime.now()
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
        "cancellation_time": datetime.now()
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
        "applied_on": datetime.now()
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
        "cancellation_time": datetime.now()
    }

tools = [search_events, search_internships, book_event, cancel_event, apply_for_internship, cancel_internship_application]

llm_with_tools = llm.bind_tools(tools=tools)

class MessageState(MessagesState):
    pass

sys_msg = SystemMessage(
    content="You are helpful assistant for a community called Student Tribe."
            "You help students book for technical events and apply for internships, by collecting their details such as name, email, and other details as required."
            "Use the provided tools to search for events and internships, book events and internships, and cancel events and internships based on the user/student's requirement."
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
)
 
def assistant(state:MessageState):
    return {"messages":llm_with_tools.invoke([sys_msg] + state['messages'])}

from langgraph.prebuilt import tools_condition, ToolNode

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools=tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition
)
builder.add_edge("tools", "assistant")
graph = builder.compile(checkpointer=memory)

# config = {
#     "configurable":{
#         "thread_id":1
#     }
# }

# def stream_graph_events(user_input):
#     response = graph.invoke({"messages":[HumanMessage(content=user_input)]}, config)
#     print(f"Assistant : {response['messages'][-1].content}")

# while True:
#     try:
#         user_input = input("User:")
#         stream_graph_events(user_input)
#     except:
#         user_input = "Hello"
#         print(f"User:{user_input}")
#         stream_graph_events(user_input)
#         break

app = FastAPI()

class ChatInput(BaseModel):
    messages : list[str]
    thread_id : str

@app.post("/chat")
async def chat(input:ChatInput):
    config = {
        "configurable":{
            "thread_id":input.thread_id
        }
    }
    response = await graph.ainvoke(
        {
            "messages":input.messages
        },
        config=config
    )
    return response['messages'][-1].content

