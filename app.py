import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

# Page config
st.set_page_config(page_title="Pakistan Fintech Knowledge Assistant", page_icon="🇵🇰")

# Auto-build vectorstore if it doesn't exist
if not os.path.exists("vectorstore"):
    with st.spinner("Building knowledge base... (first time only, takes ~2 mins)"):
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        loader = PyPDFDirectoryLoader("pdfs/")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        vectorstore_build = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory="vectorstore"
        )
        st.rerun()

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
    st.markdown("- **LLM:** Llama 3.1 8b Instant")
    st.markdown("- **Provider:** Groq")
    st.markdown("- **Vector DB:** ChromaDB")
    st.markdown("- **Embeddings:** BAAI/bge-base-en-v1.5")
    
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
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
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
if "suggested" not in st.session_state:
    st.session_state.suggested = None

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Suggested questions
if not st.session_state.messages:
    st.markdown("**💡 Try asking:**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📱 Branchless banking accounts in Pakistan?"):
            st.session_state.suggested = "What is the total number of branchless banking accounts in Pakistan?"
            st.rerun()
        if st.button("👩 Female financial inclusion rate?"):
            st.session_state.suggested = "What is the female financial inclusion rate in Pakistan?"
            st.rerun()
        if st.button("💰 Pakistan federal budget 2025-26?"):
            st.session_state.suggested = "What is the total federal budget outlay for 2025-26?"
            st.rerun()
    
    with col2:
        if st.button("📈 How has RAAST grown since launch?"):
            st.session_state.suggested = "How many transactions has RAAST processed since its launch?"
            st.rerun()
        if st.button("🚧 Barriers to digital savings?"):
            st.session_state.suggested = "Why do Pakistanis not use digital savings platforms?"
            st.rerun()
        if st.button("🏦 Digital payments growth in Pakistan?"):
            st.session_state.suggested = "How did digital payments grow from FY19 to FY25?"
            st.rerun()
    
    st.divider()

# Handle suggested question or typed input
user_input = None
if st.session_state.suggested:
    user_input = st.session_state.suggested
    st.session_state.suggested = None
elif prompt := st.chat_input("Ask about Pakistan's fintech, payments, budget, or financial inclusion..."):
    user_input = prompt

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            rewritten = rewrite_query(user_input, st.session_state.messages[:-1])
            docs = vectorstore.similarity_search(rewritten, k=6)
            
            context_parts = []
            sources = []
            for doc in docs:
                context_parts.append(doc.page_content)
                source = doc.metadata.get('source', 'Unknown')
                source = os.path.basename(source)
                sources.append(source)

            context = "\n\n".join(context_parts)
            unique_sources = list(set(sources))

            messages = [
                SystemMessage(content=f"""You are a Pakistan fintech and financial data assistant.
Answer the user's question based ONLY on the context below.
If the answer is not clearly in the context, say "I don't have information on that in my documents."
Be specific and mention actual numbers and figures when available.

Context:
{context}""")
            ] + st.session_state.messages[-6:]

            response = llm.invoke(messages)
            st.markdown(response.content)

            if unique_sources:
                st.divider()
                st.caption(f"📄 Sources: {', '.join(unique_sources)}")

    st.session_state.messages.append(AIMessage(content=response.content))

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
    st.markdown("- **LLM:** Llama 3.1 8b Instant")
    st.markdown("- **Provider:** Groq")
    st.markdown("- **Vector DB:** ChromaDB")
    st.markdown("- **Embeddings:** BAAI/bge-base-en-v1.5")
    
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
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
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
if "suggested" not in st.session_state:
    st.session_state.suggested = None

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Suggested questions
if not st.session_state.messages:
    st.markdown("**💡 Try asking:**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📱 Branchless banking accounts in Pakistan?"):
            st.session_state.suggested = "What is the total number of branchless banking accounts in Pakistan?"
            st.rerun()
        if st.button("👩 Female financial inclusion rate?"):
            st.session_state.suggested = "What is the female financial inclusion rate in Pakistan?"
            st.rerun()
        if st.button("💰 Pakistan federal budget 2025-26?"):
            st.session_state.suggested = "What is the total federal budget outlay for 2025-26?"
            st.rerun()
    
    with col2:
        if st.button("📈 How has RAAST grown since launch?"):
            st.session_state.suggested = "How many transactions has RAAST processed since its launch?"
            st.rerun()
        if st.button("🚧 Barriers to digital savings?"):
            st.session_state.suggested = "Why do Pakistanis not use digital savings platforms?"
            st.rerun()
        if st.button("🏦 Digital payments growth in Pakistan?"):
            st.session_state.suggested = "How did digital payments grow from FY19 to FY25?"
            st.rerun()
    
    st.divider()

# Handle suggested question or typed input
user_input = None
if st.session_state.suggested:
    user_input = st.session_state.suggested
    st.session_state.suggested = None
elif prompt := st.chat_input("Ask about Pakistan's fintech, payments, budget, or financial inclusion..."):
    user_input = prompt

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            rewritten = rewrite_query(user_input, st.session_state.messages[:-1])
            docs = vectorstore.similarity_search(rewritten, k=6)
            
            context_parts = []
            sources = []
            for doc in docs:
                context_parts.append(doc.page_content)
                source = doc.metadata.get('source', 'Unknown')
                source = os.path.basename(source)
                sources.append(source)

            context = "\n\n".join(context_parts)
            unique_sources = list(set(sources))

            messages = [
                SystemMessage(content=f"""You are a Pakistan fintech and financial data assistant.
Answer the user's question based ONLY on the context below.
If the answer is not clearly in the context, say "I don't have information on that in my documents."
Be specific and mention actual numbers and figures when available.

Context:
{context}""")
            ] + st.session_state.messages[-6:]

            response = llm.invoke(messages)
            st.markdown(response.content)

            if unique_sources:
                st.divider()
                st.caption(f"📄 Sources: {', '.join(unique_sources)}")

    st.session_state.messages.append(AIMessage(content=response.content))