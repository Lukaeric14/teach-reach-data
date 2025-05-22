from openai import OpenAI
import os

# Initialize the client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# List available models
print("Fetching available models...")
models = client.models.list()

# Print model IDs
print("\nAvailable models:")
for model in models.data:
    print(f"- {model.id}")

print("\nDone!")
