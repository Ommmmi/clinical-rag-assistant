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
from src.prompt import system_prompt
import os

# Load env vars first
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY

# Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Lazy globals
rag_chain = None
chat_histories = {}


def get_rag_chain():
    global rag_chain
    if rag_chain is not None:
        return rag_chain

    print("Loading embeddings...")
    embeddings = download_hugging_face_embeddings()

    print("Connecting to Pinecone...")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot",
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    print("Loading LLM...")
    llm = ChatGroq(model="llama-3.1-8b-instant")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("RAG chain ready!")
    return rag_chain


@app.route("/")
def home():
    return "Clinical RAG Assistant is running!"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/get", methods=["GET", "POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if request.is_json:
        msg = request.json.get("msg")
        session_id = request.json.get("session_id", "default_user")
    else:
        msg = request.form.get("msg")
        session_id = "default_user"

    if not msg:
        return jsonify({"error": "No message provided"}), 400

    print(f"[MSG] {msg}")

    try:
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

        print(f"[ANS] {answer[:80]}...")
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
