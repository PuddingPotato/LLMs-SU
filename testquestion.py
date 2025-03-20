from dotenv import load_dotenv
import os

from pythainlp.util import normalize

from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate


load_dotenv(dotenv_path = r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\.env')
nvidia_api_key = os.getenv('NVIDIA_API_KEY')
typhoon_ai_api_key = os.getenv('TYPHOON_AI_API_KEY')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Junior, a friendly and knowledgeable male AI assistant.\nYour role is to assist users by providing academic advice, explanations, and helpful insights based on the given context.\nAlways ensure your responses are clear, concise, and easy to understand.\nUse only the provided context to answer questions, and do not rely on external knowledge.\nIf the context lacks sufficient information, politely ask the user for clarification.\nContext: {context}"),
    ('human', "{input}")
    ])

embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0",
                                           cohere_api_key = cohere_api_key)

vectorstore = FAISS.load_local(r"C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\vectorstore\faiss_index_cohere", embedding_model, allow_dangerous_deserialization = True)

nvidiallama33_70 = ChatNVIDIA(model="meta/llama-3.3-70b-instruct",
                                api_key = nvidia_api_key)

query = embedding_model.embed_query("มีวิชาชื่อ PRECALCULUS ไหม")
results = vectorstore.similarity_search_with_score_by_vector(query)

for doc, score in results:
    print(f'Score: {score}')
    print(f'Content: {doc}\n')
