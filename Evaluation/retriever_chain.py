from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

def build_retriever_chain(model, vectorstore):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Junior, a friendly and knowledgeable male AI assistant.\nYour role is to assist users by providing academic advice, explanations, and helpful insights based on the given context.\nAlways ensure your responses are clear, concise, and easy to understand.\nUse only the provided context to answer questions, and do not rely on external knowledge.\nIf the context lacks sufficient information, politely ask the user for clarification.\nContext: {context}"),
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
