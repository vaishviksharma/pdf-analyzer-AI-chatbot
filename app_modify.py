import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_lottie import st_lottie
import requests
import io
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animations
lottie_document = load_lottieurl("https://lottie.host/166bd88a-d9d0-498b-8f86-774195b99454/MBWDTWyffi.json")
lottie_chat = load_lottieurl("https://lottie.host/df41fd21-d0a2-4904-8e86-6d266299bca2/m6FgBZPrax.json")

# PDF processing functions
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            
            # Process images in the PDF
            pdf.seek(0)  # Reset file pointer
            images = convert_from_bytes(pdf.read())
            for image in images:
                text += extract_text_from_image(image) + "\n"
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
            st.error("This file may not be a valid PDF or might be corrupted. Please check the file and try again.")
            return None
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

template = """
You are an AI legal advisor chatbot. Your task is to answer questions based solely on the content of the uploaded PDF document. 

Given the following extracted parts of a legal document and a question, create a final answer.

context: {context}

question: {question}

If the question is not related to the content of the PDF, politely decline to answer and explain that you can only provide information based on the uploaded document.

Answer:
"""

def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(input_variables=["question", "context"], template=template)
    chains = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chains

def user_input(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question})
    st.write("Answer:", response["output_text"])

def generate_summary(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = "Summarize the following legal document in a concise manner:\n\n" + text
    response = model.invoke(prompt)
    return response.content

# Page config and CSS remain the same

# Main content
st.title("🗂 DocWhisperer: Chat with Your Legal Documents")

# Display document animation before file upload
if lottie_document:
    st_lottie(lottie_document, height=300, key="document")

# File uploader
uploaded_files = st.file_uploader("Upload your legal document (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing your document..."):
        # Initialize progress bar and status message
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Read PDF and get text
        status_text.text("Reading PDF and extracting text...")
        raw_text = get_pdf_text(uploaded_files)
        progress_bar.progress(33)
        
        if raw_text is not None:
            # Generate and display summary
            status_text.text("Generating summary...")
            summary = generate_summary(raw_text)
            progress_bar.progress(66)
            
            st.subheader("PDF Summary")
            st.write(summary)

            # Create text chunks and vector store
            status_text.text("Creating text chunks and vector store...")
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            progress_bar.progress(100)

            status_text.text("Processing complete!")
            st.success("Document processed successfully! You can now ask questions about its content.")
            
            # Remove the progress bar and status text after completion
            progress_bar.empty()
            status_text.empty()

            # Chat interface
            if lottie_chat:
                st_lottie(lottie_chat, height=200, key="chat")

            user_question = st.text_input("Ask a question about your document:")
            if user_question:
                user_input(user_question, vector_store)
        else:
            st.error("Unable to process the document. Please upload a valid PDF file.")
            # Remove the progress bar and status text if there's an error
            progress_bar.empty()
            status_text.empty()

# Display warning
st.warning("Please note that DocWhisperer provides information based on the uploaded document. Always consult with a qualified legal professional for accurate legal advice.")

# Footer
st.markdown("""
---
<p style="text-align: center; color: #666666;">© 2024 Sahi Jawab - AI Legal Advisor. All rights reserved.</p>
""", unsafe_allow_html=True)
