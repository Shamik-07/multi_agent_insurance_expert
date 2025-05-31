#  A Multi-Agent, VLM-Powered Framework for Health Insurance Document Comprehension

A multi-agent framework, powered by Vision-Language Models (VLMs) and Retrieval Augmented Generation (RAG), designed to answer health insurance queries based on insurance policies and brochures.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
  - [1. Data Ingestion & VLM-RAG Pipeline](#1-data-ingestion--vlm-rag-pipeline)
  - [2. Multi-Agent System](#2-multi-agent-system)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation (Vector Database Setup)](#data-preparation-vector-database-setup)
- [Running the Application](#running-the-application)
  - [Using the Gradio Web UI](#using-the-gradio-web-ui)
- [Configuration](#configuration)
- [Resources](#resources)



## Overview

Navigating the complexities of health insurance policies can be daunting. `multi_agent_insurance_expert` aims to simplify this by providing an intelligent system that can "read" and understand insurance documents (PDFs) to answer your specific questions. It uses a novel approach where PDF pages are treated as images, allowing Vision-Language Models to interpret layouts, tables, and text visually. This is combined with a multi-agent architecture that can also leverage web search and Wikipedia for broader queries.

## Features

*   **VLM-Powered RAG:** Processes PDF documents by converting pages to images, generating multi-modal embeddings, and using VLMs to answer questions based on these visual document pages.
*   **Multi-Agent Framework:** Utilizes `smolagents` to create a hierarchical system with a manager agent orchestrating specialized agents for:
    *   Insurance Document Q&A (via VLM-RAG)
    *   Web Search (DuckDuckGo)
    *   Wikipedia Search
*   **Efficient Semantic Search:** Employs `Milvus` as a vector database for storing and retrieving relevant document page embeddings.
*   **Advanced Embedding Models:** Uses `colpali-engine` (`vidore/colqwen2.5-v0.2`) for generating rich image embeddings.
*   **Powerful LLMs/VLMs:** Leverages state-of-the-art Qwen models (e.g., `Qwen/Qwen2.5-VL-72B-Instruct`, `Qwen/Qwen3-235B-A22B`) via Hugging Face Inference Endpoints.
*   **Interactive Web UI:** A `Gradio` interface allows users to ask questions, view agent interactions, and inspect the source PDF documents.
*   **Open Source:** MIT Licensed.

## How It Works

The system operates in two main phases: data ingestion for the RAG pipeline and query processing via the multi-agent system.

### 1. Data Ingestion & VLM-RAG Pipeline

This process builds the knowledge base from your insurance PDF documents:

1.  **PDF to Image Conversion:** Input PDF documents (e.g., policy wordings) are processed. Each page is converted into a PNG image using `pdf2image` (which relies on `poppler-utils`).
2.  **Multi-Modal Embedding:** The `Colpali-Engine` (`vidore/colqwen2.5-v0.2`) generates dense vector embeddings for each page image. These embeddings capture both visual and semantic information.
3.  **Vector Storage:** The image embeddings, along with metadata (filepath, document ID), are stored in a `Milvus` vector database.
4.  **Indexing:** Milvus creates an index (e.g., FLAT or more advanced like HNSW if configured) for efficient similarity search.

### 2. Multi-Agent System & Query Processing

When a user submits a query:

1.  **Manager Agent Receives Task:** The main `manager_agent` (e.g., powered by `Qwen/Qwen3-235B-A22B`) receives the user's query.
2.  **Task Delegation:**
    *   If the query is related to health insurance (based on its prompt engineering), the `manager_agent` delegates it to the specialized `insurance_agent`.
    *   For general knowledge, it might delegate to the `web_search_agent` or `wikipedia_agent`.
3.  **Insurance Query Handling (VLM-RAG in action by `insurance_agent`):**
    *   **Query Embedding:** The user's query is embedded using `Colpali-Engine`.
    *   **Semantic Search:** The `Milvus` database is searched for the most similar page image embeddings. A re-ranking logic sums scores for multi-vector representations of pages to improve retrieval.
    *   **VLM Question Answering:** The retrieved page images (as base64) and the original query are passed to a VLM (`Qwen/Qwen2.5-VL-72B-Instruct`).
    *   **Answer Generation:** The VLM "reads" the images and formulates an answer, citing the source documents.
4.  **Response to User:** The final answer, potentially along with intermediate steps from the agents, is presented to the user via the Gradio UI.

## Technology Stack

*   **Agent Framework:** `smolagents`
*   **LLMs/VLMs:** Qwen series models via Hugging Face Inference Endpoints (e.g., `Qwen/Qwen2.5-VL-72B-Instruct`, `Qwen/Qwen3-235B-A22B`, `Qwen/Qwen3-30B-A3B`)
*   **Multi-Modal Embeddings:** `colpali-engine` (`vidore/colqwen2.5-v0.2`)
*   **Vector Database:** `Milvus` (`pymilvus` client)
*   **PDF Processing:** `pdf2image`, `poppler-utils`
*   **Web Interaction & Parsing:** `DuckDuckGoSearchTool`, `VisitWebpageTool` (from `smolagents`), `BeautifulSoup4`,  `markdownify`
*   **UI Framework:** `Gradio`, `gradio-pdf`
*   **Core Python & ML:** Python 3.12+, `PyTorch`, `Transformers`, `Hugging Face Hub`
*   **Development Tools:** `uv` (for package management, optional), `python-dotenv`, `pyprojroot`

## Prerequisites

*   **Operating System:** Linux (due to `poppler-utils` and general compatibility). macOS might work with Homebrew. Windows will require WSL.
*   **Python:** Version 3.12 or higher.
*   **Git:** For cloning the repository.
*   **Poppler Utilities:** Required by `pdf2image` for PDF processing.
    ```bash
    sudo apt update
    sudo apt install -y poppler-utils
    ```
*   **Hugging Face Hub Token:** To access models via Hugging Face Inference Endpoints. You'll need an account and an API token with payment methods set up if using paid inference endpoints like "hyperbolic" provider mentioned in notebooks.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/multi_agent_insurance_expert.git # Replace with your repo URL
    cd multi_agent_insurance_expert
    ```

2.  **Install dependencies:**
    You can use `uv` (if installed: `pip install uv`) or `pip`.
    ```bash
    # Using uv (faster)
    uv sync
    ```


3.  **Set up Environment Variables:**
    Create a `.env` file in the project root directory by copying the example (if you create one) or manually:
    ```env
    # .env
    HUGGING_FACE_HUB_TOKEN="hf_YOUR_HUGGING_FACE_TOKEN"
    # Add any other necessary environment variables, e.g., for Milvus connection if not local, etc.
    # HF_BILL_TO="VitalNest" # As seen in notebook, if your HF provider requires it
    # HF_PROVIDER="hyperbolic" # As seen in notebook
    ```
    Replace `hf_YOUR_HUGGING_FACE_TOKEN` with your actual Hugging Face API token.

4.  **Set up PYTHONPATH (if not using an editable install):**
    The `.envrc` file is provided for `direnv` users. If you don't use `direnv`, you might need to set `PYTHONPATH` manually, or ensure your project is installed in a way Python can find it (e.g., editable install).
    ```bash
    # For direnv users:
    # direnv allow

    # Manually (for current session):
    # export PYTHONPATH=$PWD:$PWD/src
    ```

## Data Preparation (Vector Database Setup)

Before you can query your insurance documents, you need to process them and populate the Milvus vector database.

1.  **Place PDF Documents:**
    Copy your health insurance PDF files into the `data/policy_wordings/` directory.

2.  **Create the Vector Database:**
    You'll need to run a script to ingest these PDFs. Currently, the `RAG` class in `src/multi_agent_insurance_expert/complex_rag.py` has a `create_vector_db` method.
    You can create a simple Python script in the project root, e.g., `ingest_data.py`:

    ```python
    # ingest_data.py
    from dotenv import load_dotenv
    from src.multi_agent_insurance_expert.complex_rag import RAG
    from src.multi_agent_insurance_expert.consts import PROJECT_ROOT_DIR
    import logging

    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    if __name__ == "__main__":
        rag_app = RAG()
        # This will process PDFs from PROJECT_ROOT_DIR / "data/policy_wordings"
        # and create/populate a Milvus DB named "policy_wordings"
        # The DB file will be located at src/multi_agent_insurance_expert/milvus_policy_wordings.db
        status_message = rag_app.create_vector_db(
            vectordb_id="policy_wordings",
            dir=PROJECT_ROOT_DIR / "data" # RAG expects a "policy_wordings" subdir here
        )
        print(status_message)
    ```
    Then run this script from the project root:
    ```bash
    python ingest_data.py
    ```
    This process can take some time depending on the number and size of your PDF documents. It will create a local Milvus database file (e.g., `src/multi_agent_insurance_expert/milvus_policy_wordings.db`).

    *Note: The `agents.py` file currently hardcodes `rag_app.vectordb_id = "policy_wordings"`. Ensure this matches the `vectordb_id` used during ingestion.*

## Running the Application

### Using the Gradio Web UI

The primary way to interact with `multi_agent_insurance_expert` is through its Gradio web interface.

1.  Ensure your virtual environment is activated and environment variables are set.
2.  Run the UI:
    ```bash
    python run_ui.py
    ```
    This will start the Gradio server, typically on `http://127.0.0.1:7860`. Open this URL in your web browser.



## Configuration

*   **Environment Variables (`.env` file):**
    *   `HUGGING_FACE_HUB_TOKEN`: Your Hugging Face API token (required).
    *   `HF_BILL_TO`: (Optional) Billing account for Hugging Face if using specific providers like "hyperbolic".
    *   `HF_PROVIDER`: (Optional) Specific Hugging Face provider.
*   **PDF Document Location:** Store PDFs in `data/policy_wordings/` for ingestion.
*   **Vector Database Name:** Hardcoded as `policy_wordings` in `agents.py` and typically used in ingestion. The Milvus DB file will be `src/multi_agent_insurance_expert/milvus_policy_wordings.db`.
*   **Agent Models & Prompts:** Configured within `src/multi_agent_insurance_expert/agents.py` and `src/multi_agent_insurance_expert/consts.py`.



## Resources
* [Slides](https://docs.google.com/presentation/d/1FqZReg4l_BtZv3Aq1IDV4HPApu_9-PvChDKTKSg1Sj0/edit?usp=sharing)
* [Demo](https://huggingface.co/spaces/Shamik/multi_agent_rag)
