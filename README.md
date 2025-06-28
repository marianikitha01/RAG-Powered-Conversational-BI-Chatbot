# RAG-Powered Conversational BI Chatbot
This is a Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on your custom tabular data using natural language. It leverages `LangChain`, `FAISS`, `sentence-transformers`, and `OpenAI` to create an intelligent Q&A system over BI datasets. It combines:

* **FAISS** for fast vector similarity search
* **LangChain** for retrieval and chaining
* **OpenAI GPT** for natural‑language responses
* **Pandas fallbacks** for precise BI calculations when API quota is exhausted
* **Docker** for reproducible, containerized deployment

---
## What it Does

- Reads CSV files from the `/data` directory
- Creates vector embeddings using HuggingFace models
- Stores them in a FAISS vector store
- Answers user queries using OpenAI GPT-3.5/4 and LangChain

---

## Features

* **Conversational BI**: Ask questions like “ Which product saw the biggest sales increase in December 2010 compared to November 2010?” in plain English.
* **Hybrid execution**: Uses OpenAI when available; if quota runs out or cloud errors occur, falls back to pandas‑based analytics.
* **Pandas‑based fallback**: Handles key BI queries (e.g. month‑over‑month sales increases) with guaranteed correctness.
* **Containerized**: One‑line Docker build and run for zero‑dependency demos.

---

## Quick Start

### 1. Clone this repository

```bash
git clone https://github.com/marianikitha01/RAG-Powered-Conversational-BI-Chatbot.git
```

### 2. (Local) Set up Python environment

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure credentials

1. Copy the placeholder file:

   ```bash
   # Windows
   copy .env.example 
   # macOS/Linux
   cp .env.example 
   ```
2. Open `.env` and fill in your:

   * `OPENAI_API_KEY`

### 4. Prepare your data

Place your `sales.csv` file (100K rows sampled) in the `data/` folder. If you have the original Excel, run:

```bash
python convert_data.py
```

### 5. Run the chatbot

```bash
python ask_bi.py
```

Type questions at the prompt and enjoy your BI assistant.

---

## Docker Deployment

1. Build the Docker image:

   ```bash
   ```

docker build -t rag-bi-chatbot\:latest .

````

2. Run the container (mount data and load `.env`):
```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  --env-file .env \
  rag-bi-chatbot:latest
````

> The chatbot will start inside the container—ask questions as usual.

---

