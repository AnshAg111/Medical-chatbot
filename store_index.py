from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import os
from dotenv import load_dotenv
load_dotenv()
key=os.getenv('PINECONE_API_KEY')
# print(key)
extracted_data=load_pdf("data/")
text_chunks=text_split(extracted_data)
embeddings=download_hugging_face_embeddings()

pc=Pinecone(api_key=key)
index_name = "medical-chatbot"
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

documents=[]
for t in text_chunks:
    document=Document(
        page_content=t.page_content,
        metadata=t.metadata,
    )
    documents.append(document)
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)