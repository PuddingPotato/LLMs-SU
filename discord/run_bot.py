import os
from dotenv import load_dotenv
from discord import Intents, Client
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from responses import get_response
from langchain_cohere import CohereEmbeddings

# Load Token
load_dotenv(dotenv_path = r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\.env')
discord_token = os.getenv('DISCORD_TOKEN')
typhoon_ai_api_key = os.getenv('TYPHOON_AI_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')

# Bot Setup
intents = Intents.default()
intents.message_content = True
client = Client(intents=intents)

# Message
async def send_message(message, user_message) -> None:
    if not user_message:
        print("Message was empty because intents were not enabled probably")
        return
    
    if is_private := user_message[0] == "?":
        user_message = user_message[1:]

    try:
        response = get_response(user_message, vectorstore, llmopenai)
        await message.author.send(response) if is_private else await message.channel.send(response)
        if message.author == client.user:
            return
        
        user_message = message.content
        channel = str(message.channel)

        print(f'[{channel}] {client.user}: "{response}')

    except Exception as e:
        print("Error",  e)

# Startup
@client.event
async def on_ready() -> None:
    print(f"{client.user} is now running!")


# Incoming messages
@client.event
async def on_message(message) -> None:
    if message.author == client.user:
        return

    username = str(message.author)
    user_message = message.content
    channel = str(message.channel)

    print(f'[{channel}] {username}: "{user_message}')
    if channel.lower() == "chat-with-sugpt":
        await send_message(message, user_message)
    else:
        print("Send from other text channel, skipped...")


# Main Entry Point
def main() -> None:
    client.run(token = discord_token)


if __name__ == "__main__":

    print('Load Embedding Model.')
    embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0",
                                           cohere_api_key = cohere_api_key)

    vectorstore = FAISS.load_local(r"C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\vectorstore\faiss_index_cohere", embedding_model, allow_dangerous_deserialization = True)
    print('Vector Loaded.')

    llmopenai = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                        model='typhoon-v2-70b-instruct',
                        api_key= typhoon_ai_api_key)
    
    main()