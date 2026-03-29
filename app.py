import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

# Page config
st.set_page_config(page_title="Pakistan Fintech Knowledge Assistant", page_icon="🇵🇰")

# Sidebar
with st.sidebar:
    st.title("🇵🇰 About This App")
    st.markdown("This assistant answers questions strictly from official Pakistani financial documents. It will not guess or make up answers.")
    
    st.divider()
    
    st.markdown("**📄 Documents Loaded:**")
    st.markdown("- SBP Branchless Banking Statistics (Oct-Dec 2025)")
    st.markdown("- SBP Payment Systems Review (2024-25)")
    st.markdown("- Karandaaz Financial Inclusion Survey 2025")
    st.markdown("- Karandaaz Digital Deposits & Investment Report")
    st.markdown("- Federal Budget in Brief 2025-26")
    
    st.divider()
    
    st.markdown("**⚙️ System Info:**")
    st.markdown("- **LLM:** Llama 3.3 70b")
    st.markdown("- **Provider:** Groq")
    st.markdown("- **Vector DB:** ChromaDB")
    st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main UI
st.title("🇵🇰 Pakistan Fintech Knowledge Assistant")
st.caption("Ask questions about Pakistan's financial ecosystem — powered by official documents.")

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
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)

vectorstore = load_vectorstore()
llm = load_llm()

# Query rewriting for follow-up questions
def rewrite_query(question, chat_history):
    if not chat_history:
        return question
    history_text = "\n".join([
        f"User: {m.content}" if isinstance(m, HumanMessage)
        else f"Assistant: {m.content}"
        for m in chat_history[-4:]
    ])
    rewrite_messages = [
        SystemMessage(content="""Given conversation history and a follow-up question,
rewrite the follow-up as a complete standalone question.
Return ONLY the rewritten question, nothing else."""),
        HumanMessage(content=f"History:\n{history_text}\n\nFollow-up: {question}")
    ]
    result = llm.invoke(rewrite_messages)
    return result.content

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
if prompt := st.chat_input("Ask about Pakistan's fintech, payments, budget, or financial inclusion..."):

    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):

            # Rewrite query for better retrieval
            rewritten = rewrite_query(prompt, st.session_state.messages[:-1])

            # Retrieve relevant chunks
            docs = vectorstore.similarity_search(rewritten, k=6)
            
            # Build context with sources
            context_parts = []
            sources = []
            for doc in docs:
                context_parts.append(doc.page_content)
                source = doc.metadata.get('source', 'Unknown')
                # Clean up source path to just filename
                source = os.path.basename(source)
                sources.append(source)

            context = "\n\n".join(context_parts)
            unique_sources = list(set(sources))

            # Build messages
            messages = [
                SystemMessage(content=f"""You are a Pakistan fintech and financial data assistant.
Answer the user's question based ONLY on the context below.
The context may contain tables with numbers — interpret them carefully.
If the answer is not clearly in the context, say "I don't have information on that in my documents."
Be specific and mention actual numbers and figures when available.

Context:
{context}""")
            ] + st.session_state.messages

            response = llm.invoke(messages)
            st.markdown(response.content)

            # Show sources
            if unique_sources:
                st.divider()
                st.caption(f"📄 Sources: {', '.join(unique_sources)}")

    st.session_state.messages.append(AIMessage(content=response.content))