
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
import os


load_dotenv()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}



if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, port=port)