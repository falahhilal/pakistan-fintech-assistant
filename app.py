import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Knowledge Chatbot")
st.caption("Ask me anything from the documents!")

# Load vectorstore
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )
    return vectorstore

# Load LLM
@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

vectorstore = load_vectorstore()
llm = load_llm()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat input
if prompt := st.chat_input("Ask something from the PDFs..."):

    # Show user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):

            # Retrieve relevant chunks from vectorstore
            docs = vectorstore.similarity_search(prompt, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Build messages with context injected
            messages = [
                SystemMessage(content=f"""You are a helpful assistant. 
Answer the user's question based ONLY on the context below.
If the answer is not in the context, say "I don't have information on that in my documents."

Context:
{context}""")
            ] + st.session_state.messages

            response = llm.invoke(messages)
            st.markdown(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))