from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr
import time, json
import qrcode
from pythainlp.util import normalize
import os

from chat_history import ChatHistory
from preprocess.create_vectorstores import create_faiss_vectorstore

def build_retriever_chain(model):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following contexts to answer the question. \nIgnore your own knowledge. \nContext:{context}"),
        ('human', "{input}")
    ])
    
    chain = create_stuff_documents_chain(
        llm = model,
        prompt = prompt
    )
    
    retriever = vectorstore.as_retriever()

    retriever_chain = create_retrieval_chain(
        retriever = retriever,
        combine_docs_chain = chain
    )
    return retriever_chain


def build_history_aware_retriever_chain(model):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following contexts to answer the question. \nIgnore your own knowledge. \nContext:{context}"),
        MessagesPlaceholder(variable_name = "chat_history"),
        ('human', "{input}")
    ])

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"),
    ])
    
    chain = create_stuff_documents_chain(
        llm = model,
        prompt = prompt
    )

    retriever = vectorstore.as_retriever()

    history_aware_retriever = create_history_aware_retriever(
        llm = model,
        retriever = retriever,
        prompt = retriever_prompt,
    )

    history_aware_retriever_chain = create_retrieval_chain(
        retriever = history_aware_retriever,
        combine_docs_chain = chain,
    )

    return history_aware_retriever_chain


def load_chat_history():
    with open(r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\user_chat_history.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def save_chat_history(chat_data):
    with open(r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\user_chat_history.json', 'w', encoding='utf-8') as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=4)


def manage_history(username, humanmessage, aimessage):
    global history
    chat_data = load_chat_history()
    history = ChatHistory(chat_data)

    if username in history.chat_history:
        print('Adding new history.')
        history.add_history(username=username, new_history=[humanmessage, aimessage])
    else:
        print(f"Creating new user data name {username}.")
        history.create_user(username=username, history=[humanmessage, aimessage])
    
    save_chat_history(history.chat_history)


def ask_question(history_awareness, username, model, query):
    starttime = time.time()
    normalized_query = normalize(query)
    print("Questioner:", username)
    if model == 'LLaMA 3.2 3B':
        print('Selected LLaMA 3.2 3B')
        model = ollamallama32_3

    elif model == 'TYPHOON AI 2 3B':
        print('Selected TYPHOON 2 3B')
        model = ollamatyphoon2_3

    elif model == 'TYPHOON AI 2 70B':
        print('Selected TYPHOON 2 70B')
        model = openaityphoon2_70

    else:
        print("Please select the model.")


    if history_awareness:
        if username not in history.chat_history.keys(): # New user who want to use the model with chat history
            retriever_chain = build_retriever_chain(model)
            response = retriever_chain.invoke({"input": normalized_query})
            manage_history(username = username, humanmessage = query, aimessage = response['answer'])

        else:
            cleaned_history = [HumanMessage(text) if i%2 == 0 else AIMessage(text) for i, text in enumerate(history.get_history(username))]
            retriever_chain = build_history_aware_retriever_chain(model)
            response = retriever_chain.invoke({"input": normalized_query,
                                               "chat_history": cleaned_history})
            manage_history(username = username, humanmessage = query, aimessage = response['answer'])


    else:
        print('Asking without history.')
        retriever_chain = build_retriever_chain(model)
        response = retriever_chain.invoke({"input": normalized_query})

    print('Normalized_Question:', normalized_query)
    print("Answer:", response["answer"])
    print(f"{len(response['answer'])} tokens used.")

    docsim = vectorstore.similarity_search(query)
    related_doc = "-----Related Document-----"

    for i, doc in enumerate(docsim):
        related_doc = related_doc + f"\nDocument {i+1}: " + doc.page_content + '\n' + '#------------------------------------------------#'

    endtime = time.time()
    timespend = endtime - starttime
    print("Time Spend: %.2f secs." % timespend)
    print('\n')

    return response["answer"], related_doc


## --------------------------------------------------------------


if __name__ == "__main__":
    
    #Chat History Data
    with open(r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\user_chat_history.json', 'r', encoding='utf-8') as f:
        chat_data = json.load(f)

    history = ChatHistory(chat_data)

    # Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    print('Embedding Model Loaded.')

    # Vector store
    vectorstore = create_faiss_vectorstore()
    vectorstore = FAISS.load_local(r"C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\vectorstore\faiss_vector", embedding_model, allow_dangerous_deserialization = True)
    print('Vector Loaded.')

    # LLMs
    ollamallama32_3 = OllamaLLM(model = "llama3.2:3b")
    ollamatyphoon2_3 = OllamaLLM(model = 'Sawatastic/llama3.2-typhoon2-3b-instruct:latest')
    print('LLaMA 3.2 3B Loaded.')

    openaityphoon2_70 = ChatOpenAI(base_url = 'https://api.opentyphoon.ai/v1',
                        model = 'typhoon-v2-70b-instruct',
                        api_key = "sk-8RMyxzbGRz3SdlvgrIcYYikDfooLrRLljWPTUrUex4YnZr8C")
    print('Typhoon 2 Loaded.')

    time.sleep(2)
    os.system('cls')
    
    gr_interface = gr.Interface(
        fn=ask_question,
        inputs=[
            gr.Checkbox(label = "Chat History Awareness"),
            gr.Dropdown([name for name in history.chat_history.keys()], allow_custom_value=True),
            gr.Dropdown(['LLaMA 3.2 3B', 'TYPHOON AI 2 3B', 'TYPHOON AI 2 70B']),
            gr.Textbox(info = 'Question')
        ],
        outputs=[
            gr.Textbox(label="Response"),
            gr.Textbox(label="Related Documents")
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