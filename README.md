# Multi-Agent RAGbot: ML Knowledge Assistant

This is a Streamlit-based chatbot that uses a Multi-Agent RAG (Retrieval-Augmented Generation) approach to answer questions about Machine Learning. It utilizes OpenAI for reasoning and Pinecone for vector storage.

## Setup and Installation

### 1. Prerequisites
- Python 3.8 or higher installed.
- API Keys for **OpenAI** and **Pinecone**.

### 2. Create and Activate Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies
Install the required Python packages:

```bash
pip -q install streamlit openai "pinecone>=5"
```

### 4. Configuration
You need to set up your API keys. You can do this by creating a `.streamlit/secrets.toml` file in the root directory:

**File:** `.streamlit/secrets.toml`
```toml
OPENAI_API_KEY = "your_openai_api_key"
PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_INDEX_NAME = "your_pinecone_index_name"
```

## Running the App

Once the environment is set up, run the application using Streamlit:

```bash
streamlit run app.py
```