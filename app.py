from flask import Flask, jsonify, request
from flask_cors import CORS
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.prompt import *
import os

app = Flask(__name__)
CORS(app)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Lazy-loaded — NOT initialized at startup to stay under 512MB
rag_chain = None
chat_histories = {}


def get_rag_chain():
    global rag_chain
    if rag_chain is not None:
        return rag_chain

    print("Initializing embeddings and RAG chain...")
    embeddings = download_hugging_face_embeddings()

    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot",
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    chatModel = ChatGroq(model="llama-3.1-8b-instant")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("RAG chain ready.")
    return rag_chain


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/get", methods=["GET", "POST"])
def chat():
    session_id = "default_user"

    if request.is_json:
        msg = request.json.get("msg")
    else:
        msg = request.form.get("msg")

    if not msg:
        return jsonify({"error": "No message provided"}), 400

    print(f"User message: {msg}")

    chain = get_rag_chain()

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    history = chat_histories[session_id]

    response = chain.invoke({
        "input": msg,
        "chat_history": history
    })

    answer = response["answer"]
    history.append(HumanMessage(content=msg))
    history.append(AIMessage(content=answer))

    print("Response:", answer)
    return jsonify({"answer": answer})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)