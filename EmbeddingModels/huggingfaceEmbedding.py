from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from dotenv import load_dotenv


load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# Generate embeddings
vector = embedding.embed_documents(documents)
print(vector)
