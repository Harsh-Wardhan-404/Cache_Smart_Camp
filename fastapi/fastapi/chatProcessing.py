# import json
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain.prompts import PromptTemplate
# from langchain.llms import CTransformers
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import RedirectResponse

# class ChatRequest(BaseModel):
#     user_message: str
#     role: str  

# class ChatResponse(BaseModel):
#     bot_reply: str

# def load_event_details(file_path: str) -> str:
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#         events = data['events']
#         event_details = "Upcoming Campus Events:\n\n"
#         for i, event in enumerate(events, 1):
#             event_details += f"{i}. **{event['name']}**\n"
#             event_details += f"- **Date:** {event['date']}\n"
#             event_details += f"- **Location:** {event['location']}\n"
#             event_details += f"- **Contact:** {event['contact']}\n"
#             event_details += f"- **Description:** {event['description']}\n\n"
#         return event_details

# EVENT_DETAILS = load_event_details('data/event_details.json')
# # FEES_DETAILS = load_event_details('data/fees_details.json')
# # Initialize the LLM
# llm = CTransformers(
#     model='models\llama-2-7b-chat.Q6_K.gguf',
#     model_type='llama',
#     config={'max_new_tokens': 256, 'temperature': 0.7}
# )

# # Define a prompt template to guide the LLM's responses
# prompt_template = PromptTemplate(
#     input_variables=["user_message", "role"],
#     template=f"""
# You are a helpful assistant for a college social media platform. Use the following event details to answer the user's queries accurately.Give Answer in not more than 3. If the question is related to fees, use the Fees_Details and if the quesyion is related to events, use the Event_Details.



# Event Details:
# {EVENT_DETAILS}




# User Role: {{role}}
# User Message: {{user_message}}

# Assistant:
# """
# )






















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

def load_event_details(file_path: str, key: str) -> str:
    with open(file_path, 'r') as file:
        data = json.load(file)
        items = data.get(key, [])
        details = f"Upcoming Campus {key.capitalize()}:\n\n"
        for i, item in enumerate(items, 1):
            details += f"{i}. **{item['name']}**\n"
            details += f"- **Date:** {item['date']}\n"
            details += f"- **Location:** {item['location']}\n"
            details += f"- **Contact:** {item['contact']}\n"
            details += f"- **Description:** {item['description']}\n\n"
        return details

def load_fee_details(file_path: str) -> str:
    with open(file_path, 'r') as file:
        data = json.load(file)
        fee_structure = data.get('fee_structure', {})
        categories = fee_structure.get('categories', [])
        notes = fee_structure.get('notes', [])
        
        fee_details = f"Fee Structure for Academic Year {fee_structure.get('academic_year', 'N/A')}:\n\n"
        for i, category in enumerate(categories, 1):
            fee_details += f"{i}. **{category.get('category', 'N/A')}**\n"
            fee_details += f"- **Tuition Fees:** {category.get('tuition_fees', 'N/A')}\n"
            fee_details += f"- **Development Fees:** {category.get('development_fees', 'N/A')}\n"
            fee_details += f"- **Exam Fees:** {category.get('exam_fees', 'N/A')}\n"
            fee_details += f"- **Misc Fees:** {category.get('misc_fees', 'N/A')}\n"
            fee_details += f"- **Total Fees:** {category.get('total_fees', 'N/A')}\n\n"
        
        fee_details += "Notes:\n"
        for note in notes:
            fee_details += f"- {note}\n"
        
        return fee_details

def determine_query_type(query):
    fees_keywords = ["fee", "fees", "tuition", "cost", "price", "payment", "scholarship"]
    events_keywords = ["event", "events", "orientation", "seminar", "workshop", "conference", "meeting"]

    query_lower = query.lower()

    if any(keyword in query_lower for keyword in fees_keywords):
        return "fees"
    elif any(keyword in query_lower for keyword in events_keywords):
        return "events"
    else:
        return "unknown"

EVENT_DETAILS = load_event_details('data/event_details.json', 'events')
FEES_DETAILS = load_fee_details('data/fees.json')

# Initialize the LLM
llm = CTransformers(
    model='models\\llama-2-7b-chat.Q6_K.gguf',
    model_type='llama',
    config={'max_new_tokens': 256, 'temperature': 0.7}
)

# Define a prompt template to guide the LLM's responses
prompt_template = PromptTemplate(
    input_variables=["user_message", "role", "details"],
    template=f"""
You are a helpful assistant for a college social media platform. Use the following details to answer the user's queries accurately. Give the response in only 1-3 sentences.

Details:
{{details}}

User Role: {{role}}
User Message: {{user_message}}

Assistant:
"""
)

app = FastAPI(title="API")

@app.post("/getEventsDescriptions")
def getEventsDescriptions():
    return {"message": "Events descriptions"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Determine the query type
        query_type = determine_query_type(request.user_message)
        
        # Select the appropriate details based on the query type
        if query_type == "fees":
            details = FEES_DETAILS
        elif query_type == "events":
            details = EVENT_DETAILS
        else:
            details = "I'm sorry, I couldn't determine the context of your query."

        # Format the prompt with user input and relevant details
        prompt = prompt_template.format(
            user_message=request.user_message,
            role=request.role,
            details=details
        )
        
        # Get the response from the LLM
        response = llm(prompt)
        
        return ChatResponse(bot_reply=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "College Campus Chatbot API is up and running."}