from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import os
import time

def create_chroma_vectorstore():
    print('Creating Chroma vector store.')
    print('Loading latest dataset.')
    datasets = pd.read_csv(r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\transformed_data_2567.csv')

    docs = [Document(page_content=doc[0], metadata={'source': doc[1]}) for doc in datasets.values]

    # Initialize embeddings model
    print('Load Embedding Model.')
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print('Embedding Model Loaded.')

    # Create vector store
    vectorstore = Chroma.from_documents(documents = docs,
                                        persist_directory = r"C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\vectorstore\chroma_vector",
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
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print('Embedding Model Loaded.')

    # Create vector store
    vectorstore = FAISS.from_documents(documents = docs, embedding = embedding_model)
    vectorstore.save_local(r"C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\vectorstore\faiss_vector")

    os.system('cls')
    print('Vector Store Created.')
    time.sleep(1)

    return vectorstore

if __name__ == "__main__":
    vector = create_faiss_vectorstore()