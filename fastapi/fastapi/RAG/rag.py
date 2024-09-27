from sentence_transformers import SentenceTransformer
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from transformers import GPT2TokenizerFast 
import os

# Set the TRANSFORMERS_CACHE environment variable
os.environ['TRANSFORMERS_CACHE'] = 'C:\\Users\\arya2\\huggingface_cache'

 # Tokenizer to count tokens

# Data ingestion
from langchain_community.document_loaders import PyPDFLoader

# Step 1: Load the PDF document
loader = PyPDFLoader('money.pdf')
docs = loader.load()

# Load a pre-trained model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose any suitable model
llm = CTransformers(model='models\llama-2-7b-chat.Q6_K.gguf',
                    model_type='llama',
                    config={'max_new_tokens': 256, 'temperature': 0.7})

# Extract text from the documents
inputs = [doc.page_content for doc in docs]

# Step 2: Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = text_splitter.split_text(" ".join(inputs))  # Combine all text, then split into chunks

# Step 3: Generate embeddings for each chunk
embeddings = embedding_model.encode(chunks)

# Initialize the ChromaDB client
client = chromadb.Client()
collection = client.create_collection("document_embeddings")  # Create a collection

# Step 4: Store the chunks and their embeddings in ChromaDB
for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    collection.add(
        ids=[str(idx)],  # Unique ID for each chunk
        documents=[chunk],  # The chunk of text
        embeddings=[embedding.tolist()]  # The embedding as a list
    )

print("Document chunks and embeddings stored in ChromaDB.")

# Tokenizer for counting tokens (using GPT-2 tokenizer as an example)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")  # Choose the appropriate tokenizer

# Function to handle queries
def ask_question(query):
    # Step 5: Embed the query
    query_embedding = embedding_model.encode([query])

    # Step 6: Perform similarity search
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3  # Number of similar documents to retrieve
    )

    # Step 7: Extract the relevant chunks (ensure all are strings)
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

    # Step 8: Check token count and limit the context to fit the model's token limit
    tokenized_context = tokenizer(context)
    if len(tokenized_context["input_ids"]) > 512:
        # Truncate context to fit within 512 tokens
        truncated_input_ids = tokenized_context["input_ids"][:512]
        context = tokenizer.decode(truncated_input_ids, skip_special_tokens=True)

    # Prepare the prompt for the LLM
    prompt = f"Based on the following context: {context}, answer the question: {query}"

    # Generate a response using the LLM
    response = llm.invoke(prompt)

    return response

# Example usage: Ask a question
query = "What is the fee for the first year of B.Tech?"
response = ask_question(query)
print("LLM Response:", response)