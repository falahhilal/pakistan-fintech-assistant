from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load all PDFs from the pdfs folder
print("Loading PDFs...")
loader = PyPDFDirectoryLoader("pdfs/")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# 3. Embed and save to ChromaDB
print("Embedding and saving to ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="vectorstore"
)
print("Done! Vectorstore saved.")
