# embed.py

import os
import time
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env if available

# Updated imports using langchain_community
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    # Read required API keys and region from environment variables.
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_region = os.environ.get("PINECONE_ENV")  # e.g., "us-west-2"

    if not all([openai_api_key, pinecone_api_key, pinecone_region]):
        raise ValueError("Missing API keys or environment variables. Make sure to set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_REGION.")

    # Create an instance of the new Pinecone client.
    pc = PineconeClient(api_key=pinecone_api_key)

    index_name = "test-2"

    pc.create_index(
        name=index_name,
        dimension=1536, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )


    # Load the PDF file.
    pdf_path = "Micturition-RAG.pdf"  # Replace with your actual PDF file path.
    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the document into manageable chunks.
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Create embeddings using OpenAI.
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Store the embeddings in the Pinecone index via LangChain's integration.
    print("Storing embeddings in Pinecone...")
    Pinecone.from_documents(docs, embeddings, index_name=index_name)

    print("PDF successfully embedded and stored in Pinecone index.")

if __name__ == "__main__":
    main()
