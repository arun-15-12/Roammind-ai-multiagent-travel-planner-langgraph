import os
import certifi
from dotenv import load_dotenv
load_dotenv()

# Fix for SSL certificate issues across different platforms
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)

from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

# =========================
# Environment & Configurations
# =========================

def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )
    if "sslmode" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url += f"{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Please add your GROQ API Key to .env"
    )

# =========================
# LLM Initialization
# =========================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
)

# =========================
# Graph State Definition
# =========================

class TravelState(TypedDict):
    messages: Annotated[list, operator.add]  # Reducer appends messages automatically
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int  

# =========================
# Node Functions (Agents)
# =========================

def flight_agent(state: TravelState):
    query = state['user_query']
    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flights results fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }  

def hotel_agent(state: TravelState):
    query = state['user_query']
    hotel_data = tavily_search(f"hotels in {query}")

    return {
        "hotel_results": hotel_data,
        "messages": [
            AIMessage(content="Hotel results fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Make the itinerary practical, budget aware, and easy to follow.
"""  
    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner"),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
  
def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user 

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Day-by-Day Itinerary
4. Estimated Budget
5. Hotel Suggestions
6. Final Recommendations

Important:
Be clear and practical in your suggestions and recommendations.
Mention that live flight API may not provide ticket price if pricing is unavailable.
Keep the response useful for real travel planning.
"""

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant"),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# =========================
# Build & Compile Graph 
# =========================

graph = StateGraph(TravelState)

# Add Nodes
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

# Add Edges (Fixed Workflow Sequence)
graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

# PostgreSQL Checkpointing Setup
DATABASE_URL = get_database_url()
_conn = psycopg.connect(DATABASE_URL, row_factory=dict_row, autocommit=True)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)

# =========================
# Execution Entry Point for FastAPI
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    # 1. Generate unique thread_id if new conversation
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # 2. Invoke the compiled LangGraph workflow
    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    # 3. Extract the final generated response message
    final_answer = result["messages"][-1].content 

    # 4. Return structured JSON payload
    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }