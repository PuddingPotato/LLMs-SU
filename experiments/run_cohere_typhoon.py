from dotenv import load_dotenv
import os
import time

from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_openai import ChatOpenAI
from retriever_chain import build_retriever_chain


load_dotenv(dotenv_path = r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\.env')
os.environ['LANGSMITH_PROJECT'] = 'Cohere - TYPHOON AI 2 70B'
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING'] = 'true'

cohere_embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0",
                                          cohere_api_key=os.getenv('COHERE_API_KEY'))

cohere_vectorstore = FAISS.load_local(r"C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\vectorstore\faiss_index_cohere",
                                      cohere_embedding_model,
                                      allow_dangerous_deserialization=True)

openaityphoon2_70 = ChatOpenAI(base_url = 'https://api.opentyphoon.ai/v1',
                               model = 'typhoon-v2-70b-instruct',
                               api_key = os.getenv('TYPHOON_AI_API_KEY'))

retriever_chain = build_retriever_chain(openaityphoon2_70, cohere_vectorstore)

with open(r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\Evaluation\test_question.txt', 'r', encoding="utf-8") as file:
    test_questions = file.readlines()
test_questions = [question.strip() for question in test_questions]

request_count_per_minute = 0
for i, question in enumerate(test_questions):
    request_count_per_minute += 1
    if request_count_per_minute >= 45:
        elapsed_time = time.time() - minute_start_time
        if elapsed_time < 60:

            time_to_wait = 60 - elapsed_time
            print(f"Approaching minute limit. Waiting {time_to_wait:.2f} seconds...")
            time.sleep(time_to_wait)
        
        minute_start_time = time.time()
        request_count_per_minute = 0
        
    try:
        print(f'Processing Question {i+1}...')
        results = retriever_chain.invoke({'input': question})
        print(f'Question-Answering Test {i+1}:')
        print(f'  Question: {results["input"]}')
        print(f'  Answer: {results["answer"]}')
    except Exception as e:
        print(f'Error on question {i+1}: {str(e)}')
        if 'rate limit' in str(e).lower():
            wait_time = 20
            print(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            try:
                results = retriever_chain.invoke({'input': question})
                print(f'Question-Answering Test {i+1} (retry):')
                print(f'  Question: {results["input"]}')
                print(f'  Answer: {results["answer"]}')
            except Exception as retry_error:
                print(f'Retry failed: {str(retry_error)}')
    
    print('-----------------------------------------------')
    time.sleep(5)