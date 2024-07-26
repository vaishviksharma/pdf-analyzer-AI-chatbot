import os
from dotenv import load_dotenv
import streamlit as st

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

# Load GOOGLE_API_KEY from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Raise an error if the key is not found
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Template for the QA chain
template = """You are a chatbot having a conversation with a user based on the context.

Given the following extracted parts of a long document and a question, create a final answer. If you do not have the context, please do not give a wrong answer.

Context: \n {context}\n

Question: \n {question}\n
Answer:"""

# Function to read PDF and extract text
def get_pdf_reader(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(input_variables=["question", "context"], template=template)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to load FAISS index
def load_faiss_index(pickle_file):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    faiss_index = FAISS.load_local(pickle_file, embeddings=embeddings, allow_dangerous_deserialization=True)
    return faiss_index

# Function to process user input and display the response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    new_db = load_faiss_index("faiss_index")
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question})
    st.write("*Answer: *", response["output_text"])

# Streamlit app setup
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="auto"
)

with st.sidebar:
    st.title("PDF Upload")
    pdf_docs = st.file_uploader("Upload the file", accept_multiple_files=True, type=['pdf'])

    if st.button("Submit"):
        with st.spinner("Uploading..."):
            raw_text = get_pdf_reader(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("VectorDB upload finished")

def main():
    st.header("Chat with PDF")
    user_question = st.text_input("Ask your question")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
