from langchain.schema import Document
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import os
import time
from langchain_ollama import OllamaEmbeddings
from langchain_cohere import CohereEmbeddings

def create_chroma_vectorstore():
    print('Creating Chroma vector store.')
    print('Loading latest dataset.')
    datasets = pd.read_csv(r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\transformed_data_2567.csv')

    docs = [Document(page_content=doc[0], metadata={'source': doc[1]}) for doc in datasets.values]

    # Initialize embeddings model
    print('Load Embedding Model.')
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    print('Embedding Model Loaded.')

    # Create vector store
    vectorstore = Chroma.from_documents(documents = docs,
                                        persist_directory = r"C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\vectorstore\chroma_index",
                                        embedding = embedding_model)
    
    os.system('cls')
    print('Vector Store Created.')
    time.sleep(1)
    
    return vectorstore


def create_faiss_vectorstore():
    print('Creating FAISS vector store.')
    print('Loading latest dataset..')
    datasets = pd.read_csv(r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\transformed_data_2567.csv')

    docs = [Document(page_content=doc[0]) for doc in datasets.values]

    # Initialize embeddings model
    print('Load Embedding Model..')
    embedding_model = OllamaEmbeddings(model="bge-m3:latest")
    print('Embedding Model Loaded.')

    # Create vector store
    vectorstore = FAISS.from_documents(documents = docs, embedding = embedding_model)
    vectorstore.save_local(r"C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\vectorstore\faiss_index")

    os.system('cls')
    print('Vector Store Created.')
    time.sleep(1)

    return vectorstore

def create_faiss_vectorstore_from_embedding():
    print('Creating FAISS vector store.')
    datasets = pd.read_csv(r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\transformed_data_2567.csv', nrows=100)

    print('Load Embedding Model..')
    embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0",
                                      cohere_api_key=cohere_api_key)
    print('Embedding Model Loaded.')

    # Extract the text content
    texts = [doc[0] for doc in datasets.values]
    
    # Create documents
    docs = [Document(page_content=text) for text in texts]
    
    # Process in batches with token-based rate limiting
    batch_size = 5
    all_docs = []
    
    def estimate_tokens(text):
        return len(text)
    
    for i in range(0, len(docs), batch_size):
        print(f"Processing batch {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1}")
        batch = docs[i:i+batch_size]
        
        # Calculate total tokens in this batch
        total_batch_tokens = sum(estimate_tokens(doc.page_content) for doc in batch)
        
        # Calculate sleep time based on rate limit
        tokens_per_minute_limit = 100000
        minutes_needed = total_batch_tokens / tokens_per_minute_limit
        seconds_needed = minutes_needed * 60
        
        # Add buffer for safety (20%)
        sleep_time = seconds_needed * 1.2
        
        all_docs.extend(batch)
        
        # Sleep based on token count (only if not the last batch)
        if i + batch_size < len(docs):
            print(f"Batch size: {len(batch)} docs, ~{total_batch_tokens} tokens")
            print(f"Rate limiting pause: {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
    
    # Create vector store
    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(all_docs, embedding_model)
    
    vectorstore.save_local(r"C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\vectorstore\faiss_index_from_embedding")

    os.system('cls')
    print('Vector Store Created.')
    time.sleep(1)

    return vectorstore

if __name__ == "__main__":
    load_dotenv(dotenv_path = r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\.env')
    cohere_api_key = os.getenv('COHERE_API_KEY')
    vector = create_faiss_vectorstore_from_embedding()