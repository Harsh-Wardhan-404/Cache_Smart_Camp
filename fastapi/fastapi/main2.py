from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from transformers import GPT2TokenizerFast
import chromadb
import os
import logging
from langchain_community.llms import Ollama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up cache directory for transformers
cache_dir = r'C:\Users\arya2\.cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

os.environ['TRANSFORMERS_CACHE'] = cache_dir

# Initialize FastAPI app
app = FastAPI(title="College Campus Chatbot API")

# Initialize the ChromaDB client
client = chromadb.Client()
collection = client.create_collection("document_embeddings")

# Load a pre-trained model for embeddings with a cache directory
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)

# Tokenizer for counting tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Initialize the Ollama model
ollama = Ollama(model='qwen2', base_url="http://localhost:11434")

# Load and process the text file
def load_and_process_text(file_path: str):
    loader = TextLoader(file_path)
    docs = loader.load()

    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Generate embeddings for each chunk
    embeddings = embedding_model.encode([chunk.page_content for chunk in chunks])

    # Store the chunks and their embeddings in ChromaDB
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[str(idx)],  # Unique ID for each chunk
            documents=[chunk.page_content],  # The chunk of text
            embeddings=[embedding.tolist()]  # The embedding as a list
        )

    logger.info("Document chunks and embeddings stored in ChromaDB.")

# Function to handle queries
def ask_question(query: str):
    # Embed the query
    query_embedding = embedding_model.encode([query])

    # Perform similarity search
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3  # Number of similar documents to retrieve
    )

    # Extract the relevant chunks (ensure all are strings)
    retrieved_chunks = results['documents']  # Get the text of the most relevant chunks

    # Ensure each retrieved document is a string and not a list
    flat_chunks = []
    for doc in retrieved_chunks:
        if isinstance(doc, list):
            flat_chunks.append(" ".join(doc))  # Join the inner list elements into a single string
        else:
            flat_chunks.append(doc)

    # Combine the flat chunks into a single string for context
    context = " ".join(flat_chunks)

    # Check token count and limit the context to fit the model's token limit
    tokenized_context = tokenizer(context)
    if len(tokenized_context["input_ids"]) > 512:
        # Truncate context to fit within 512 tokens
        context = tokenizer.decode(tokenized_context["input_ids"][:512])

    # Prepare the prompt for the LLM
    prompt = f"Based on the following context: {context}, answer the question: {query}"

    # Generate a response using the Ollama model
    response = ollama.invoke(input=prompt)

    return response

# Load the knowledge base text file
knowledge_base_path = 'data/knowledge_base.txt'
if not os.path.exists(knowledge_base_path):
    raise FileNotFoundError(f"The knowledge base file was not found at path: {knowledge_base_path}")

# Ensure the directory exists
os.makedirs(os.path.dirname(knowledge_base_path), exist_ok=True)

load_and_process_text(knowledge_base_path)

class ChatRequest(BaseModel):
    user_message: str
    role: str  

class ChatResponse(BaseModel):
    bot_reply: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Handle the query
        response = ask_question(request.user_message)
        return ChatResponse(bot_reply=response)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "College Campus Chatbot API is up and running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)