# RAG-based LLM Question Answering System

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents or website URLs and ask questions based on their content.

---

## Features

- Upload PDF documents
- Ingest website URLs
- Semantic search using embeddings
- Retrieval-Augmented Generation (RAG)
- Fast responses using Groq LLM
- Interactive chat interface
- Source citation for answers

---

## System Architecture

User Query  
↓  
Retriever (Chroma Vector DB)  
↓  
Relevant Document Chunks  
↓  
LLM (Groq API)  
↓  
Generated Answer

---

## Tech Stack

Languages  
- Python

Frameworks & Libraries  
- Streamlit  
- LangChain  
- HuggingFace Embeddings  

Vector Database  
- ChromaDB

LLM  
- Groq API

---

## Project Structure

RAG-LLM-APP
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

---

## Installation

Clone the repository

git clone https://github.com/Chirag514/RAG-Based-LLM.git  
cd RAG-Based-LLM

Install dependencies

pip install -r requirements.txt  

Add API key

Create a `.env` file and add:

GROQ_API_KEY=your_api_key_here

Run the application

streamlit run app.py

---
