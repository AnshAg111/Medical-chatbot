from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

app=Flask(__name__)

load_dotenv()
key=os.getenv('PINECONE_API_KEY')
embeddings=download_hugging_face_embeddings()
pc=Pinecone(api_key=key)
index_name = "medical-chatbot"
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}
llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8}
                  )
# question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
# chain = create_retrieval_chain(vector_store.as_retriever(search_kwargs={"k": 2}), question_answer_chain)
qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg=request.form["msg"]
    result=qa({"query":msg})
    return str(result["result"])

if __name__=="__main__":
    app.run(debug=True)