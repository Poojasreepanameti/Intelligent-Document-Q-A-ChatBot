# Intelligent Document Q&A Chatbot using RAG

A Retrieval-Augmented Generation (RAG) system that answers user questions from uploaded documents. Built with Python, LangChain, FAISS, Hugging Face and Streamlit.

## Features

- Upload one or more PDF or TXT documents
- Text extraction, chunking and embedding generation
- Semantic similarity search using a FAISS vector store
- Retrieved chunks are passed as context to an LLM to generate grounded answers
- Streamlit chat interface with conversation history
- Shows the context used for each answer

## Tech Stack

| Component       | Tool                                                     |
| --------------- | -------------------------------------------------------- |
| Language        | Python                                                   |
| Framework       | LangChain                                                |
| Vector database | FAISS                                                    |
| Embeddings      | Hugging Face `sentence-transformers/all-MiniLM-L6-v2`    |
| LLM             | Hugging Face `Qwen/Qwen2.5-0.5B-Instruct` (runs locally) |
| PDF reading     | pypdf                                                    |
| Interface       | Streamlit                                                |

## How It Works

**Indexing (when documents are processed)**

1. Extract text from the uploaded PDF/TXT files
2. Split the text into chunks (500 characters, 50 overlap)
3. Convert each chunk into an embedding vector
4. Store the vectors in a FAISS index

**Querying (when a question is asked)**

1. Convert the question into an embedding
2. Retrieve the top 3 most similar chunks from FAISS
3. Build a prompt with the retrieved chunks and the question
4. The LLM generates an answer using only that context

## Project Structure

```
.
├── app.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

1. Upload PDF or TXT files in the sidebar
2. Click **Process Documents**
3. Type a question in the chat box

The first run downloads the Hugging Face models, so an internet connection is needed.

## Limitations

- Scanned PDFs are not supported (no OCR)
- Each question is answered independently; chat history is displayed but not used for follow-up questions
- The index is kept in memory and is rebuilt each time documents are processed
- The LLM is a small model (0.5B parameters), so answers can be less accurate than larger models
