import os
import bs4
import time
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

#########################################################
# PAGE CONFIG
#########################################################

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

#########################################################
# CUSTOM CSS
#########################################################

st.markdown(
    """
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
max-width:1200px;
}

.stChatMessage{
border-radius:15px;
padding:10px;
margin-bottom:10px;
}

code{
font-size:15px;
}

</style>
""",
    unsafe_allow_html=True,
)

#########################################################
# TITLE
#########################################################

st.title("🤖 AI Assistant")

st.caption(
    "Powered by Groq • LangChain • HuggingFace • ChromaDB"
)

#########################################################
# SIDEBAR
#########################################################

with st.sidebar:

    st.title("⚙️ Settings")

    st.success("🟢 Ready")

    st.divider()

    st.subheader("Model")

    st.write("**Groq**")

    st.write("Llama-3.3-70B")

    st.divider()

    st.subheader("Embedding")

    st.write("all-MiniLM-L6-v2")

    st.divider()

    st.subheader("Knowledge Base")

    st.write("Lilian Weng Blog")

    st.divider()

    if st.button("🗑 Clear Chat", use_container_width=True):

        st.session_state.messages=[]

        st.rerun()

#########################################################
# WELCOME MESSAGE
#########################################################

if "messages" not in st.session_state:

    st.session_state.messages=[]

if len(st.session_state.messages)==0:

    st.info(
        """
### 👋 Welcome

You can ask me questions about

✅ Python

✅ SQL

✅ Machine Learning

✅ Deep Learning

✅ LangChain

✅ LLMs

✅ RAG

✅ AI Agents

or anything from the uploaded website.
"""
    )

#########################################################
# DISPLAY CHAT
#########################################################

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])



#########################################################
# LOAD ENVIRONMENT VARIABLES
#########################################################

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    os.environ["HF_TOKEN"] = hf_token

#########################################################
# VALIDATE API KEY
#########################################################

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()



#########################################################
# BUILD RAG PIPELINE
#########################################################

@st.cache_resource(show_spinner="Loading AI Knowledge Base...")
def build_vectorstore():

    #################################################
    # LLM
    #################################################

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    #################################################
    # Embeddings
    #################################################

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    #################################################
    # Website Loader
    #################################################

    loader = WebBaseLoader(
        web_paths=(
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
        ),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )

    docs = loader.load()

    #################################################
    # Split Documents
    #################################################

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    splits = splitter.split_documents(docs)

    #################################################
    # Vector Database
    #################################################

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    return llm, retriever, len(splits)



#########################################################
# INITIALIZE
#########################################################

llm, retriever, total_chunks = build_vectorstore()



st.subheader("Knowledge Base")

st.write("📄 Lilian Weng Blog")

st.metric("Chunks", total_chunks)

st.metric("Embedding", "MiniLM")



#########################################################
# CONTEXTUALIZE USER QUESTION
#########################################################

contextualize_q_system_prompt = """
Given a chat history and the latest user question,
which might reference context in the chat history,
rewrite the question into a standalone question.

Do NOT answer the question.

If the question is already standalone,
return it unchanged.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)



#########################################################
# HISTORY AWARE RETRIEVER
#########################################################

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)



#########################################################
# SYSTEM PROMPT
#########################################################

system_prompt = """
You are an intelligent AI Assistant.

Your responsibilities are:

1. Search the retrieved documents first.

2. If the answer exists in the retrieved documents,
answer using ONLY the retrieved context.

3. If the retrieved documents do NOT contain the answer,
answer using your own knowledge.

When answering from your own knowledge,
mention briefly:

"Note: This answer is based on my general knowledge."

For programming questions:

• Write clean production-quality code.

• Explain the code.

• Use markdown code blocks.

For SQL:

• Format SQL properly.

• Explain each clause.

For Machine Learning:

• Explain concepts step-by-step.

• Give examples.

For LangChain:

• Explain with examples.

For interview questions:

• Keep answers simple and professional.

Keep responses concise but informative.

Retrieved Context:

{context}
"""


#########################################################
# QA PROMPT
#########################################################

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)



#########################################################
# DOCUMENT CHAIN
#########################################################

question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)



#########################################################
# RAG CHAIN
#########################################################

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain,
)



#########################################################
# SESSION MEMORY
#########################################################

store = {}


def get_session_history(session_id: str):

    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]



#########################################################
# CONVERSATIONAL RAG
#########################################################

conversational_rag_chain = RunnableWithMessageHistory(

    rag_chain,

    get_session_history,

    input_messages_key="input",

    history_messages_key="chat_history",

    output_messages_key="answer",
)



#########################################################
# CONTEXTUALIZE QUESTION
#########################################################

contextualize_q_system_prompt = """
Given the chat history and the latest user question,
rewrite the question into a standalone question that
can be understood without previous conversation.

Do not answer the question.

If the question is already standalone,
return it unchanged.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

#########################################################
# HISTORY AWARE RETRIEVER
#########################################################

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)

#########################################################
# MAIN SYSTEM PROMPT
#########################################################

system_prompt = """
You are an expert AI Assistant.

Your job is to answer every question as accurately as possible.

Rules:

1. First search the retrieved documents.

2. If the answer exists inside the retrieved documents,
use ONLY that information.

3. If the retrieved documents are not enough,
answer using your own knowledge.

When using your own knowledge, begin with:

"Note: This answer is based on my general knowledge."

-----------------------------

For Coding:

• Write clean Python code.
• Explain every important step.
• Use markdown code blocks.

-----------------------------

For SQL:

• Format SQL properly.
• Explain every clause.
• Give optimization tips whenever possible.

-----------------------------

For Machine Learning:

Explain

• Concept
• Working
• Mathematical intuition
• Real-world examples
• Interview tips

-----------------------------

For LangChain:

Explain

• Components
• Flow
• Example
• Best Practices

-----------------------------

For RAG:

Explain

• Retrieval
• Embeddings
• Chunking
• Vector Database
• Retriever
• Prompt
• LLM

-----------------------------

If the user asks interview questions,
answer professionally.

Keep answers detailed but easy to understand.

Retrieved Context:

{context}
"""

#########################################################
# QA PROMPT
#########################################################

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

#########################################################
# DOCUMENT CHAIN
#########################################################

question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)

#########################################################
# RAG CHAIN
#########################################################

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain,
)




#########################################################
# CHAT UI
#########################################################

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = rag_chain.invoke(
                {
                    "input": prompt,
                    "chat_history": []
                }
            )

            answer = response["answer"]

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    