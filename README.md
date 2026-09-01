# AI Assistant — Using LangChain

A collection of LangChain fundamentals and a deployable AI assistant, covering LCEL (LangChain Expression Language), vector retrieval, conversational chatbots, conversational RAG, and API deployment via LangServe.

---

## 📌 Overview

This repository demonstrates the progression from core LangChain building blocks to a deployable, served AI assistant application — covering chain composition, retrieval, conversation memory, and exposing the application as an API.

---

## 📓 Notebooks & Scripts

| File | Focus |
|---|---|
| `simplellmLCEL.ipynb` | LangChain Expression Language (LCEL) — composing chains declaratively |
| `vectorretriever.ipynb` | Vector store setup and retriever configuration |
| `chatbots.ipynb` | Conversational chatbot patterns with LangChain |
| `conversationqa.ipynb` | Conversational Question-Answering (RAG with memory) |
| `serve.py` | Deploying a LangChain chain as a REST API using **LangServe** |
| `app.py` | Application entry point / interface |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
LangChain Chain (LCEL)
    │
    ├──► Vector Retriever ──► Relevant Context
    │
    ▼
Conversation Memory
    │
    ▼
LLM
    │
    ▼
Response
    │
    ▼
Served via LangServe API (serve.py)
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core language |
| LangChain | LLM application framework |
| LangChain Expression Language (LCEL) | Declarative chain composition |
| LangServe | Deploying LangChain chains as REST APIs |
| Vector Store / Retriever | Semantic document retrieval |
| Jupyter Notebook | Interactive experimentation |

---

## 📁 Project Structure

```
AI-Assistant---Using-LangChain/
│
├── simplellmLCEL.ipynb
├── vectorretriever.ipynb
├── chatbots.ipynb
├── conversationqa.ipynb
├── serve.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Anipireddy-Pavan/AI-Assistant---Using-LangChain.git
cd AI-Assistant---Using-LangChain

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file for any required API keys:

```
OPENAI_API_KEY=your_key_here
# or the relevant provider key used in the notebooks
```

Never commit `.env` files or API keys to GitHub.

---

## ▶️ Running the Project

**Explore the notebooks:**
```bash
jupyter notebook
```

**Run the served API:**
```bash
python serve.py
```

**Run the application:**
```bash
python app.py
```

---

## 📚 Learning Outcomes

This repository demonstrates practical experience with:

- LangChain Expression Language (LCEL)
- Building and composing chains
- Vector stores and retrievers
- Conversational chatbots and memory
- Conversational RAG (Retrieval-Augmented Generation)
- Deploying LangChain applications as APIs with LangServe

---

## 👨‍💻 Author

**Pavan Anipireddy**
Data Science | Generative AI | LangChain | Python | SQL
GitHub: [Anipireddy-Pavan](https://github.com/Anipireddy-Pavan)

---

## 📄 License

This repository is intended for educational and portfolio purposes.
