import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")

def vector_embedding():
    if "vectors" not in st.session_state:
        try:
            # Validate API key
            if not os.getenv("NVIDIA_API_KEY"):
                st.error("NVIDIA_API_KEY not found in environment variables")
                return False
                
            st.info("Initializing embeddings...")
            # Try with a specific embedding model
            try:
                st.session_state.embeddings = NVIDIAEmbeddings(
                    model="nvidia/nv-embedqa-e5-v5",
                    api_key=os.getenv("NVIDIA_API_KEY")
                )
            except Exception as embed_error:
                st.error(f"Failed to initialize NVIDIA embeddings: {embed_error}")
                st.info("Trying alternative embedding model...")
                # Fallback to a different model or provider
                st.session_state.embeddings = NVIDIAEmbeddings(
                    model="nvidia/nv-embed-v1",
                    api_key=os.getenv("NVIDIA_API_KEY")
                )
            
            st.info("Loading documents...")
            st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
            st.session_state.docs=st.session_state.loader.load() ## Document Loading
            
            if not st.session_state.docs:
                st.error("No documents found in ./us_census directory")
                return False
                
            st.info("Splitting documents...")
            st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) ## Chunk Creation
            st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) #splitting
            
            st.info(f"Creating vector embeddings for {len(st.session_state.final_documents)} document chunks...")
            # Process in smaller batches to avoid rate limits
            batch_size = 10
            documents = st.session_state.final_documents
            
            if len(documents) > batch_size:
                st.info(f"Processing in batches of {batch_size}...")
                first_batch = documents[:batch_size]
                st.session_state.vectors = FAISS.from_documents(first_batch, st.session_state.embeddings)
                
                # Add remaining documents in batches
                for i in range(batch_size, len(documents), batch_size):
                    batch = documents[i:i+batch_size]
                    batch_vectors = FAISS.from_documents(batch, st.session_state.embeddings)
                    st.session_state.vectors.merge_from(batch_vectors)
                    st.info(f"Processed {min(i+batch_size, len(documents))}/{len(documents)} documents")
            else:
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            
            return True
            
        except Exception as e:
            st.error(f"Error creating vector embeddings: {str(e)}")
            if "403" in str(e) or "Forbidden" in str(e):
                st.error("API key doesn't have permission for embeddings. Please check:")
                st.error("1. Your API key is valid and active")
                st.error("2. Your account has access to embeddings endpoints")
                st.error("3. You haven't exceeded rate limits")
            else:
                st.error("Please check your NVIDIA API key and internet connection")
            return False
    else:
        return True

st.title("Nvidia NIM Demo")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")


prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


prompt1=st.text_input("Enter Your Question From Doduments")


if st.button("Documents Embedding"):
    if vector_embedding():
        st.write("Vector Store DB Is Ready")
    else:
        st.write("Failed to create vector store. Please check the errors above.")

if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please create document embeddings first by clicking 'Documents Embedding' button")
    else:
        try:
            document_chain=create_stuff_documents_chain(llm,prompt)
            retriever=st.session_state.vectors.as_retriever()
            retrieval_chain=create_retrieval_chain(retriever,document_chain)
            start=time.process_time()
            response=retrieval_chain.invoke({'input':prompt1})
            print("Response time :",time.process_time()-start)
            st.write(response['answer'])

            # With a streamlit expander
            with st.expander("Document Similarity Search"):
                # Find the relevant chunks
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
