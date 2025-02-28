# **Retrieved Augmented Generation (RAG) System with Metadata**

This repository contains **code and instructions** for building a **Retrieval-Augmented Generation (RAG) system**.

## **📌 Overview**
- It processes **text files** and creates a **vector database with metadata**, allowing users to trace the **source of generated answers**.
- Given a **user query**, the system retrieves relevant information **along with its source and page number** for improved transparency.
- This approach ensures that responses are **grounded in factual sources** rather than relying solely on model-generated content.

## **⚙️ Setup**
- To use this system, you need to **define the parameters** of your **language model (LLM)**.
- **Note:** The `config_llm` script is **not included** in this repository. You must configure your **own LLM settings** before running the system.

## **📖 Features**
✔️ **Vector Database Creation** – Stores documents as embeddings with metadata.  
✔️ **Source Attribution** – Retrieves answers with citations (link to the source & page number). 

✔️ **Scalable & Extendable** – Can be adapted for a large folder of text files.  



🚀 **Happy Coding!** 😊
