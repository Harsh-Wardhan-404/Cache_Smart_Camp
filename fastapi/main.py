
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Initialize FastAPI app
app = FastAPI(title="College Campus Chatbot API")

# Mount the static directory for favicon
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request schema
class ChatRequest(BaseModel):
    user_message: str
    role: str  # 'user' or 'admin'

# Define the response schema
class ChatResponse(BaseModel):
    bot_reply: str

# Hardcoded Event Details
EVENT_DETAILS = """
Upcoming Campus Events:

1. **Hackathon**
   - **Date:** 29-Sept
   - **Location:** Seminar Hall
   - **Contact:** 9945894687
   - **Description:** A 24-hour coding marathon where students collaborate to solve real-world problems.

2. **Cultural Fest**
   - **Date:** 15-Oct
   - **Location:** Main Auditorium
   - **Contact:** 9123456789
   - **Description:** A day filled with cultural performances, food stalls, and fun activities.

3. **Tech Talk Series**
   - **Date:** Every Friday in October
   - **Location:** Conference Room A
   - **Contact:** 9876543210
   - **Description:** Weekly sessions featuring industry experts discussing the latest in technology and innovation.
"""

# Initialize the LLM
llm = CTransformers(
    model='models/luna-ai-llama2-uncensored.ggmlv3.q4_K_S.bin',
    model_type='llama',
    config={'max_new_tokens': 256, 'temperature': 0.7}
)

# Define a prompt template to guide the LLM's responses
prompt_template = PromptTemplate(
    input_variables=["user_message", "role"],
    template=f"""
You are a helpful assistant for a college social media platform. Use the following event details to answer the user's queries accurately.

Event Details:
{EVENT_DETAILS}

User Role: {{role}}
User Message: {{user_message}}

Assistant:
"""
)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Format the prompt with user input
        prompt = prompt_template.format(
            user_message=request.user_message,
            role=request.role
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

# Route for favicon.ico
@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse(url="/static/favicon.ico")
