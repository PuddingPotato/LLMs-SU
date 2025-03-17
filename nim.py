import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv(dotenv_path = r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\.env')
nvidia_api_key = os.getenv('NVIDIA_API_KEY')


llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct",
                 api_key = nvidia_api_key)
result = llm.invoke("คุณชื่ออะไร")
print(result.content)