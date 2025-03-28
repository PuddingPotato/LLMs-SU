from dotenv import load_dotenv
import os

from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from retriever_chain import build_retriever_chain


load_dotenv(dotenv_path = r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\.env')
os.environ['LANGSMITH_PROJECT'] = 'Cohere - LLaMA3.3 70B'
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING'] = 'true'

cohere_embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0",
                                          cohere_api_key=os.getenv('COHERE_API_KEY'))

cohere_vectorstore = FAISS.load_local(r"C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\vectorstore\faiss_index_cohere",
                                      cohere_embedding_model,
                                      allow_dangerous_deserialization=True)

nvidiallama33_70 = ChatNVIDIA(model="meta/llama-3.3-70b-instruct",
                              api_key=os.getenv('NVIDIA_API_KEY'))

retriever_chain = build_retriever_chain(nvidiallama33_70, cohere_vectorstore)


with open(r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\Evaluation\test_question.txt', 'r', encoding="utf-8") as file:
    test_questions = file.readlines()
test_questions = [question.strip() for question in test_questions]

for i, question in enumerate(test_questions):
    try:
        print(f'Processing Question {i+1}...')
        results = retriever_chain.invoke({'input': question})
        print(f'Question-Answering Test {i+1}:')
        print(f'  Question: {results["input"]}')
        print(f'  Answer: {results["answer"]}')
    except Exception as e:
        print(f'Error on question {i+1}: {str(e)}')
    print('-----------------------------------------------')