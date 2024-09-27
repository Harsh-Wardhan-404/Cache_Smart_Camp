from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

from chatProcessing import ChatRequest, ChatResponse, process_chat_request


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
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
        response = process_chat_request(request)
        return ChatResponse(bot_reply=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "College Campus Chatbot API is up and running."}

if __name__ == "__main__":
    uvicorn.run(app, port=int(os.getenv("PORT", 8000)))