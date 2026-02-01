from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDF
reader = PdfReader("data/manual.pdf")

text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Chunks created:", len(chunks))

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector store
vectorstore = FAISS.from_texts(chunks, embeddings)

print("Vector database ready!")
