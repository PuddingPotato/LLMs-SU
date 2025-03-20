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
    datasets = pd.read_csv(r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\transformed_data_2567.csv')

    # Display total number of documents
    total_docs = len(datasets)
    print(f'Total documents to process: {total_docs}')

    print('Load Embedding Model..')
    embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0",
                                      cohere_api_key=cohere_api_key)
    print('Embedding Model Loaded.')

    # Extract the text content
    texts = [doc[0] for doc in datasets.values]
    
    # Create documents
    docs = [Document(page_content=text) for text in texts]
    
    # Track document count
    docs_processed = 0
    
    # Create an empty FAISS index first
    print("Creating initial empty FAISS index...")
    # Take a small batch to initialize
    initial_batch = docs[:5]
    vectorstore = FAISS.from_documents(initial_batch, embedding_model)
    docs_processed += len(initial_batch)
    print(f"Processed {docs_processed}/{total_docs} documents")
    
    # Now add the rest in batches
    remaining_docs = docs[5:]
    batch_size = 30 
    
    for i in range(0, len(remaining_docs), batch_size):
        batch = remaining_docs[i:i+batch_size]
        current_batch_size = len(batch)
        print(f"Processing batch {(i//batch_size) + 2}/{(len(remaining_docs)-1)//batch_size + 2}")
        
        # Add documents to existing vectorstore
        vectorstore.add_documents(batch)
        docs_processed += current_batch_size
        
        # Display progress
        print(f"Processed {docs_processed}/{total_docs} documents ({(docs_processed/total_docs)*100:.1f}%)")
        
        # Apply rate limiting between batches
        if i + batch_size < len(remaining_docs):
            # Calculate total tokens in this batch (Thai estimate)
            total_batch_tokens = sum(len(doc.page_content) for doc in batch)  # 1 token per char for Thai
            
            # Calculate sleep time
            tokens_per_minute_limit = 100000
            minutes_needed = total_batch_tokens / tokens_per_minute_limit
            seconds_needed = minutes_needed * 60
            
            # Add buffer (50% for more safety)
            sleep_time = seconds_needed * 1.5
            
            print(f"Batch size: {current_batch_size} docs, ~{total_batch_tokens} tokens")
            print(f"Rate limiting pause: {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
    
    vectorstore.save_local(r"C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\vectorstore\faiss_index_from_embedding")

    print('Vector Store Created.')
    print(f'Total documents processed: {docs_processed}')
    print(f'Total documents in vectorstore: {vectorstore.index.ntotal}')
    time.sleep(1)

    return vectorstore


if __name__ == "__main__":
    load_dotenv(dotenv_path = r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\.env')
    cohere_api_key = os.getenv('COHERE_API_KEY')
    vector = create_faiss_vectorstore_from_embedding()