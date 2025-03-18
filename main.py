from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import gradio as gr
import time
import qrcode
from pythainlp.util import normalize
import os
from sentence_transformers import SentenceTransformer
from preprocess.create_vectorstores import create_faiss_vectorstore

load_dotenv(dotenv_path = r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\.env')
nvidia_api_key = os.getenv('NVIDIA_API_KEY')
typhoon_ai_api_key = os.getenv('TYPHOON_AI_API_KEY')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')

os.environ['LANGCHAIN_API_KEY'] = langsmith_api_key
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'Junior-Onboarding'

def build_retriever_chain(model):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Junior, a friendly and knowledgeable male AI assistant.\nYour role is to assist users by providing academic advice, explanations, and helpful insights based on the given context.\nAlways ensure your responses are clear, concise, and easy to understand.\nUse only the provided context to answer questions, and do not rely on external knowledge.\nIf the context lacks sufficient information, politely ask the user for clarification.\nContext: {context}"),
        ('human', "{input}")
    ])

    chain = create_stuff_documents_chain(
        llm = model,
        prompt = prompt
    )

    retriever = vectorstore.as_retriever(search_kwargs = {'k': 10})

    retriever_chain = create_retrieval_chain(
        retriever = retriever,
        combine_docs_chain = chain
    )

    return retriever_chain


def ask_question(model, query): # history_awareness, username, 
    starttime = time.time()
    normalized_query = normalize(query)
    # print("Questioner:", username)
    if model == 'LLaMA 3.3 70B':
        print('Selected LLaMA 3.3 70B')
        model = nvidiallama33_70

    elif model == 'TYPHOON AI 2 70B':
        print('Selected TYPHOON 2 70B')
        model = openaityphoon2_70

    else:
        print("Please select the model.")

    retriever_chain = build_retriever_chain(model)
    response = retriever_chain.invoke({"input": normalized_query})

    print('Normalized_Question:', normalized_query)
    print("Answer:", response["answer"])

    endtime = time.time()
    timespend = endtime - starttime
    print("Time Spend: %.2f secs." % timespend)
    print('\n')
    # print(f'responses: {list(response['context'])[0].page_content}\n')
    related_docs = [doc.page_content for doc in list(response['context'])]
    print(f'related_docs: {related_docs}', sep = '\n')
    str_related_docs = ''
    for index, doc in enumerate(related_docs):
        str_related_docs += f'document {index + 1}: {doc}\n'
    str_related_docs = f'Total Related Documents: {len(related_docs)}\n' + str_related_docs
    print('\nResponse:', response)
    return response["answer"], str_related_docs


## --------------------------------------------------------------


if __name__ == "__main__":
    # Load Embedding Model
    model_name = "kornwtp/simcse-model-phayathaibert"# "sentence-transformers/all-MiniLM-L6-v2"
    local_path = r"C:\Users\User\Desktop\Project LLMs\Embedding Models"
    if len(os.listdir(local_path)) == 0:
        model = SentenceTransformer(model_name)
        model.save(local_path)

        print("Model downloaded and saved to:", local_path)
        embedding_model = HuggingFaceEmbeddings(model_name = model_name)
        print('Embedding Model Loaded.')
        
    else:
        embedding_model = HuggingFaceEmbeddings(model_name = local_path)
        print('Embedding Model Loaded.')
        

    # Load Vector store
    print('Searching for vectors..')
    time.sleep(1)

    if len(os.listdir(path = r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\vectorstore')) == 0:
        print('No vector found. Creating new one..')
        vectorstore = create_faiss_vectorstore()
    vectorstore = FAISS.load_local(r"C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\vectorstore\faiss_index", embedding_model, allow_dangerous_deserialization = True)
    print('Vector Loaded.')

    # Load LLMs
    print('Loading Models.')
    nvidiallama33_70 = ChatNVIDIA(model="meta/llama-3.3-70b-instruct",
                                  api_key = nvidia_api_key)
    print('LLaMA 3.3 70B Loaded.')

    openaityphoon2_70 = ChatOpenAI(base_url = 'https://api.opentyphoon.ai/v1',
                        model = 'typhoon-v2-70b-instruct',
                        api_key = typhoon_ai_api_key)
    print('TYPHOON AI 2 70B Loaded.')

    time.sleep(5)
    os.system('cls')

    gr_interface = gr.Interface(
        fn=ask_question,
        inputs=[
            gr.Dropdown(['LLaMA 3.3 70B', 'TYPHOON AI 2 70B'], label = "Model", allow_custom_value = False),
            gr.Textbox(label = "Input", info = 'Question')
        ],
        outputs=[
            gr.Textbox(label = "Response"),
            gr.Textbox(label = "Related Documents", max_lines = 100)
        ],
        flagging_mode = 'auto',
    )

    gr_interface.launch(server_port=1234, share=True, prevent_thread_lock = True)

    if gr_interface.share_url:
        img = qrcode.make(gr_interface.share_url)
        img.save(r'.\data\QRCode.png')
        print(f"Shareable URL: {gr_interface.share_url}")
        print('QR code generated.')

while True:
    pass