# Medical AI Chatbot

An intelligent, full-stack medical assistant powered by **Retrieval-Augmented Generation (RAG)**. This chatbot uses LangChain and Pinecone to provide accurate health information based on specialized medical literature.

![MediCare AI UI](https://i.ibb.co/d5b84Xw/Untitled-design.png)

## 🚀 Features

- **RAG Architecture**: Fetches relevant information from medical handbooks to provide evidence-based answers.
- **Modern React UI**: A sleek, animated interface built with React, Vite, and Tailwind CSS.
- **Fast Inference**: Uses **Groq** for lightning-fast LLM responses.
- **Vector Database**: Efficient retrieval using **Pinecone** vector store.
- **Embedded Knowledge**: HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`) for semantic search.
- **Real-time Interaction**: Streaming-like experience with typing indicators and auto-scrolling.

## 🛠️ Tech Stack

### **Backend**
- **Python / Flask**: API server.
- **LangChain**: Orchestrating the RAG chain.
- **Pinecone**: High-performance vector database.
- **Groq / OpenAI**: Powering the language model logic.
- **HuggingFace**: Local embeddings generation.

### **Frontend**
- **React / TypeScript**: Modern UI components.
- **Vite**: Ultra-fast build tool and dev server.
- **Tailwind CSS**: Professional styling and responsiveness.
- **Lucide Icons**: Clean, intuitive iconography.

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Medical-Chatbot.git
cd Medical-Chatbot
```

### 2. Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
# Optional: OPENAI_API_KEY=your_openai_key
```

### 3. Backend Setup
Activate your environment (e.g., Conda) and install dependencies:
```bash
conda activate chatbot  # Or create a new one
pip install -r requirements.txt
```

### 4. Frontend Setup
Navigate to the `UI` folder and install dependencies:
```bash
cd UI
npm install
```

---

## 🏃 Running the Project

### Step 1: Data Ingestion (First time only)
Process your medical PDFs and store them in the vector database:
```bash
# Ensure your PDFs are in the /data folder
python store_index.py
```

### Step 2: Start the Backend
```bash
python app.py
```
*Backend runs on `http://localhost:8080`*

### Step 3: Start the Frontend
In a new terminal:
```bash
cd UI
npm run dev
```
*Frontend runs on `http://localhost:5173`*

---

## 📂 Project Structure

- `app.py`: Flask API entry point.
- `src/helper.py`: Utility functions for PDF loading and embeddings.
- `src/prompt.py`: Custom RAG prompt templates.
- `store_index.py`: Script to index documents into Pinecone.
- `UI/`: React frontend application.
- `data/`: Folder containing medical PDF source files.
- `research/`: Jupyter notebooks for prototyping and trials.

---

## ⚠️ Disclaimer
*This chatbot provides general health information based on medical literature. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health providers with any questions you may have regarding a medical condition.*
