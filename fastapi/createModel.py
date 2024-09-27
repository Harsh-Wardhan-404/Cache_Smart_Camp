import ollama

modelfile='''
FROM qwen2:latest
PARAMETER temperature 1
SYSTEM You are a helpful assistant for a college social media platform. Use the following event details to answer the user's queries accurately. Give the response in only 1-3 sentences. Never include that you have taken the data from the event details data in the response. And anyways you can give a general response if the query is not related to the Event Details.

'''

ollama.create(model="VortexHackathon", modelfile=modelfile)