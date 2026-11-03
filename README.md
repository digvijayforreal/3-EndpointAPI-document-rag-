# 📚 3-Endpoint API — Document Q&A with RAG

An AI-powered **Document Question Answering System** built using Retrieval-Augmented Generation (RAG). Upload PDF or DOCX documents, ask questions about their content, and receive context-aware answers supported by relevant source citations.

The application extracts document text, splits it into smaller chunks, generates vector embeddings, and retrieves relevant passages to generate accurate, document-grounded answers using an LLM.

## ✨ Features

- **PDF & DOCX Support:** Upload and process PDF and Word documents.
- **Text Extraction:** Extract readable content from uploaded documents.
- **Text Chunking:** Split documents into smaller, manageable chunks for efficient retrieval.
- **Vector Embeddings:** Convert text chunks into numerical vector representations.
- **Semantic Search:** Retrieve relevant passages based on the meaning of user questions.
- **AI-Powered Answers:** Generate contextual responses using retrieved document information.
- **Source Citations:** Display supporting document references alongside generated answers.
- **FastAPI Backend:** Dedicated API layer for document processing and question answering.
- **Web Interface:** Interact with the system through a separate frontend.
- **Streamlit Interface:** An additional interface available through `app.py`.

## 🏗️ Project Architecture

The repository contains two interfaces that use the document Q&A functionality:

### 1. FastAPI Backend + Web Frontend

The newer implementation consists of:

- `backend_v2/main.py` — FastAPI backend that handles document processing and question-answering operations.
- `frontend/` — Web interface for interacting with the backend.

### 2. Streamlit Application

- `app.py` — A separate Streamlit-based interface for document question answering.

## 🔌 Three Main API Endpoints

The FastAPI backend provides three main workflow endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/upload` | POST | Accepts PDF or DOCX documents and processes their content for retrieval. |
| `/ask` | POST | Accepts a question and retrieves relevant document chunks to generate an answer. |
| `/health` | GET | Checks whether the API service is running. |

### Endpoint Workflow

**1. POST `/upload`**

- Accepts PDF or DOCX files.
- Extracts document text.
- Splits extracted text into chunks.
- Generates embeddings for the chunks.
- Prepares document content for semantic retrieval.

**2. POST `/ask`**

- Accepts a user's question.
- Converts the question into an embedding.
- Searches for relevant document chunks.
- Sends retrieved context to the LLM.
- Returns an answer with source citations.

**3. GET `/health`**

- Checks API availability.
- Returns the service health status.

## 🔄 How the RAG Pipeline Works

```text
       PDF / DOCX Upload
               |
               v
        Text Extraction
               |
               v
          Text Chunking
               |
               v
       Embedding Generation
               |
               v
       Vector Storage
               |
               |
        User Question
               |
               v
       Question Embedding
               |
               v
        Semantic Search
               |
               v
      Relevant Text Chunks
               |
               v
        LLM Generation
               |
               v
       Answer + Citations
```

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| FastAPI | REST API backend |
| Streamlit | Alternative application interface |
| HTML, CSS, JavaScript | Web frontend |
| MongoDB Atlas | Database and vector storage |
| Jina AI | Text embedding generation |
| Groq | LLM inference and answer generation |
| PyPDF | PDF text extraction |
| DOCX processing | Word document text extraction |

## 📂 Project Structure

```text
3-EndpointAPI-document-rag/
│
├── backend_v2/
│   └── main.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── ingestion/
│
├── documents/
│
├── test/
│
├── backup/
│
├── app.py
├── multi_pdf.py
├── rag_chat.py
├── store_embeddings.py
│
├── requirements.txt
└── README.md
```

*Note: The structure above highlights the main project components. Individual files may vary depending on the repository version.*

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/digvijayforreal/3-EndpointAPI-document-rag-.git

cd 3-EndpointAPI-document-rag-
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and add the required credentials:

```env
MONGODB_URI=your_mongodb_connection_string
JINA_API_KEY=your_jina_api_key
GROQ_API_KEY=your_groq_api_key
```

Make sure the variable names match those used in your application code.

### 5. Run the FastAPI Backend

From the project root:

```bash
uvicorn backend_v2.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 6. Explore API Documentation

FastAPI automatically provides interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Use Swagger UI to explore and test the available endpoints.

### 7. Run the Streamlit Application

To launch the separate Streamlit interface:

```bash
streamlit run app.py
```

## 🔐 Environment & Security

- Store API keys and database credentials in environment variables.
- Never commit `.env` files.
- Keep database credentials private.
- Configure the MongoDB vector search index required by the retrieval pipeline.

## 🎯 Project Objective

The objective of this project is to build a practical document intelligence system that makes information retrieval easier through AI.

By combining document processing, vector embeddings, semantic search, and LLM-based generation, the application allows users to interact with their documents naturally while receiving answers supported by source references.

---

**Built with Python, FastAPI, Streamlit, MongoDB Atlas, Jina AI, and Groq.**