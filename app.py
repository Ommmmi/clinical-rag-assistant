from flask import Flask, jsonify, request
from flask_cors import CORS
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.prompt import *
import os



app = Flask(__name__)
CORS(app)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into Pinecone index.
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




# In-memory store for chat histories
chat_histories = {}

@app.route("/get", methods=["GET", "POST"])
def chat():
    # For simplicity, using a fixed session_id. In a real app, you'd get this from the request.
    session_id = "default_user"

    if request.is_json:
        msg = request.json.get("msg")
    else:
        msg = request.form.get("msg")
    
    if not msg:
        return jsonify({"error": "No message provided"}), 400
        
    print(f"User message: {msg}")

    # Get or create chat history for the session
    if session_id not in chat_histories:
        chat_histories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    memory = chat_histories[session_id]
    
    # Prepare the chain input
    chain_input = {
        "input": msg,
        "chat_history": memory.chat_memory.messages
    }

    response = rag_chain.invoke(chain_input)
    
    # Save the current interaction to memory
    memory.save_context({"input": msg}, {"output": response["answer"]})

    print("Response : ", response["answer"])
    return jsonify({"answer": str(response["answer"])})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)