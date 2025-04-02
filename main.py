from pydantic import BaseModel
from pymongo import MongoClient
from langchain_core.tools import tool

client = MongoClient("mongodb+srv://akshaymww:PE2YYynnqxZhW5qK@chatbotcluster.tavncb1.mongodb.net/?retryWrites=true&w=majority&appName=ChatBotCluster")
db = client["sttribe"]
events_collection = db["events"]

@tool
def search_events(location:str | None = None, category:str | None = None):
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

# matching_events = search_events.invoke({"location":"Bangalore", "category":"Machine Learning"})

# # Print results
# for event in matching_events:
#     print(event)

