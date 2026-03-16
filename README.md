# 📚 LangGraph RAG Chatbot

An intelligent **Retrieval-Augmented Generation (RAG) chatbot** that answers questions from a PDF document using **LangChain, LangGraph, and Google's Gemini models**.

The system loads a document, converts it into embeddings, stores them in a **Chroma vector database**, and uses an **AI agent with tool-calling** to retrieve relevant information before generating accurate answers.

The application also includes a **Streamlit chat interface** with a dark mode UI.

---

# Features

-  **PDF Document Understanding**
-  **Retrieval-Augmented Generation (RAG)**
-  **Semantic Search using Embeddings**
-  **AI Agent powered by LangGraph**
-  **Tool Calling for Retrieval**
-  **Chat Interface using Streamlit**
-  **Dark Mode UI**
-  **Fast Vector Search with ChromaDB**

---

# Architecture

User Question
↓
LangGraph Agent
↓
LLM decides whether to call retrieval tool
↓
Chroma Vector Database
↓
Relevant Document Chunks Retrieved
↓
LLM Generates Final Answer

---

# Technologies Used

- **LangChain** – LLM framework for building applications
- **LangGraph** – Agent orchestration and tool execution
- **Google Gemini** – Large Language Model
- **ChromaDB** – Vector database for document retrieval
- **Streamlit** – Web interface for the chatbot
- **Python** – Core programming language

---

# Project Structure

Chatbot-RAG/
│
├── app.py # Streamlit web interface
├── RAG_Agent.py # LangGraph RAG agent logic
├── Stock_Market_Performance_2024.pdf
├── chroma_db/ # Vector database
├── .env.example # API keys
├── Screenshots # Results of App
└── README.md

---

# Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/waleed-222/Chatbot-RAG.git
cd Chatbot-RAG
```
---
### 2️⃣ Create environment

```bash
conda create -n rag python=3.10
conda activate rag
```
---
### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```
---
### 4️⃣ Set API key

```bash
cp .env.example  .env
```
then put your own `api key`

---

### 5️⃣ Change Location

Don't forget to change location of persist_directory

---

# Running the Application

Run the RAG agent (CLI)
```bash
python RAG_Agent.py
```
---
Run the Streamlit interface
```bash
streamlit run app.py
```
Open in your browser:
```bash
http://localhost:8501
```
---

# How the RAG System Works

1- The PDF is loaded using LangChain Document Loaders

2- The document is split into chunks

3- Each chunk is converted to vector embeddings

4- Embeddings are stored in ChromaDB

5- The LangGraph agent receives a user question

6- The agent calls a retrieval tool

7- Relevant chunks are retrieved

8- The LLM generates a final answer using the retrieved context

---