import os
import sys
import json

# Add the root directory to sys.path to allow importing from 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from src.prompt import *

# Pre-initialize (Note: In serverless, these might re-run, but Netlify sometimes reuses containers)
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot" 
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
chatModel = ChatGroq(model="openai/gpt-oss-safeguard-20b") 

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Note: In-memory history will not persist reliably across calls in serverless
chat_histories = {}

def handler(event, context):
    print(f"Received event: {event['httpMethod']}")
    
    # Handle preflight OPTIONS request
    if event['httpMethod'] == 'OPTIONS':
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST, OPTIONS"
            },
            "body": ""
        }

    # Only allow POST
    if event['httpMethod'] != 'POST':
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Method not allowed"})
        }

    try:
        body = json.loads(event['body'])
        msg = body.get("msg")
    except Exception:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid JSON or missing 'msg'"})
        }
    
    if not msg:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No message provided"})
        }

    session_id = "default_user"
    if session_id not in chat_histories:
        chat_histories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    memory = chat_histories[session_id]
    chain_input = {
        "input": msg,
        "chat_history": memory.chat_memory.messages
    }

    response = rag_chain.invoke(chain_input)
    memory.save_context({"input": msg}, {"output": response["answer"]})

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*", # Enable CORS
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST, OPTIONS"
        },
        "body": json.dumps({"answer": str(response["answer"])})
    }
