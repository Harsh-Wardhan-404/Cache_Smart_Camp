import os
import gdown

# Create a directory called 'models'
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Model - Download from Google Drive
model_url = 'https://drive.google.com/uc?id=1X2GHhVr1IKx-Rv6GP8aBd9ruxz9LTgQj'  # Updated with your model's ID
model_path = os.path.join(model_dir, 'llama-2-7b-chat.Q6_K.gguf')  # Replace with the actual filename and extension
gdown.download(model_url, model_path, quiet=False)

print(f"Model downloaded to {model_path}")
