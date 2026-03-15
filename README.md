# Multi-Agent RAGbot: ML Knowledge Assistant

This is a Streamlit-based chatbot that uses a Multi-Agent RAG (Retrieval-Augmented Generation) approach to answer questions about Machine Learning. It utilizes OpenAI for reasoning and Pinecone for vector storage.

---

## Setup and Installation (Local)

### 1. Prerequisites

* Python 3.8 or higher installed.
* API Keys for **OpenAI** and **Pinecone**.

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

Set up your API keys by creating a `.streamlit/secrets.toml` file in the root directory:

**File:** `.streamlit/secrets.toml`

```toml
OPENAI_API_KEY = "your_openai_api_key"
PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_INDEX_NAME = "your_pinecone_index_name"
```

### 5. Run Locally

Run the application using Streamlit:

```bash
streamlit run app.py
```

---

## Deploying via GitHub and Streamlit Cloud

You can also deploy your chatbot directly to the cloud using Streamlit Cloud. This allows anyone to access your bot without running it locally.

### 1. Push Your Code to GitHub

Make sure all required files are in a Git repository (excluding `.gitignore` entries like secrets, virtual environments, cache files, etc.):

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

### 2. Sign in to Streamlit Cloud

Go to [Streamlit Cloud](https://share.streamlit.io/) and log in with your GitHub account.

### 3. Create a New App

* Click **New app**.
* Select your repository and branch (e.g., `main`).
* Set **Main file path** to `app.py`.

### 4. Add Secrets

In the Streamlit Cloud dashboard:

* Go to **Settings → Secrets**.
* Add your keys:

```
OPENAI_API_KEY = your_openai_api_key
PINECONE_API_KEY = your_pinecone_api_key
PINECONE_INDEX_NAME = your_pinecone_index_name
```

### 5. Deploy

Click **Deploy**. Streamlit Cloud will build your environment, install dependencies from `requirements.txt` or your `pip install` setup, and launch your bot online.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
