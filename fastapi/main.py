
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

from chatProcessing import ChatRequest, ChatResponse, prompt_template, llm

# Initialize FastAPI app
app = FastAPI(title="College Campus Chatbot API")

# Mount the static directory for favicon
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    # allow_or      igins=["http://localhost:5000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/getEventsDescriptions")
def getEventsDescriptions():
    return {"message": "Events descriptions"}


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





if __name__ == "__main__":
  
    uvicorn.run(app, port=os.getenv("PORT"))
