from langchain_community.document_loaders import PyPDFLoader
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma 
from dotenv import load_dotenv
load_dotenv()
data=PyPDFLoader("VectorStore/VectorStore.pdf").load()
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
chunks=splitter.split_documents(data)
embedding_model=OpenAIEmbeddings()
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chromadb"
)