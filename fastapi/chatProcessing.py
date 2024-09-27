import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

class ChatRequest(BaseModel):
    user_message: str
    role: str  

class ChatResponse(BaseModel):
    bot_reply: str

def load_event_details(file_path: str) -> str:
    with open(file_path, 'r') as file:
        data = json.load(file)
        events = data['events']
        event_details = "Upcoming Campus Events:\n\n"
        for i, event in enumerate(events, 1):
            event_details += f"{i}. **{event['name']}**\n"
            event_details += f"- **Date:** {event['date']}\n"
            event_details += f"- **Location:** {event['location']}\n"
            event_details += f"- **Contact:** {event['contact']}\n"
            event_details += f"- **Description:** {event['description']}\n\n"
        return event_details

EVENT_DETAILS = load_event_details('data/event_details.json')

# Initialize the LLM
llm = CTransformers(
    model='models\llama-2-7b-chat.Q6_K.gguf',
    model_type='llama',
    config={'max_new_tokens': 256, 'temperature': 0.7}
)

# Define a prompt template to guide the LLM's responses
prompt_template = PromptTemplate(
    input_variables=["user_message", "role"],
    template=f"""
You are a helpful assistant for a college social media platform. Use the following event details to answer the user's queries accurately.Give Answer in not more than 3 lines and never include that you have taken the data from the event details data in the response. And anyways you can give a general response if the query is not related to the Event Details

Event Details:
{EVENT_DETAILS}

User Role: {{role}}
User Message: {{user_message}}

Assistant:
"""
)