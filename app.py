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

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ FIX 1: Valid Groq model name
chatModel = ChatGroq(model="llama-3.1-8b-instant")


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ✅ FIX 2: Use plain list of LangChain messages (ConversationBufferMemory is deprecated)
chat_histories = {}


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

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    history = chat_histories[session_id]

    response = rag_chain.invoke({
        "input": msg,
        "chat_history": history
    })

    answer = response["answer"]

    # ✅ FIX 3: Append proper message objects, not using deprecated memory
    history.append(HumanMessage(content=msg))
    history.append(AIMessage(content=answer))

    print("Response:", answer)
    return jsonify({"answer": answer})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)