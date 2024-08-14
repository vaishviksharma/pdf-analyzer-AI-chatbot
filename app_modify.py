import os
from dotenv import load_dotenv
import streamlit as st
from langdetect import detect
import unicodedata

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

# Templates for the QA chain (in both languages)
template_en = """You are a chatbot having a conversation with a user based on the context.

Given the following extracted parts of a long document and a question, create a final answer. If you do not have the context, please do not give a wrong answer.

Context: \n {context}\n

Question: \n {question}\n
Answer:"""

template_hi = """आप एक चैटबॉट हैं जो संदर्भ के आधार पर उपयोगकर्ता से बातचीत कर रहे हैं।

निम्नलिखित लंबे दस्तावेज़ के निकाले गए भागों और एक प्रश्न के आधार पर एक अंतिम उत्तर बनाएं। यदि आपके पास संदर्भ नहीं है, तो कृपया गलत उत्तर न दें।

संदर्भ: \n {context}\n

प्रश्न: \n {question}\n
उत्तर:"""

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  # Default to English if detection fails

# Function to read PDF and extract text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
    return text

# Function to split text into chunks
def get_text_chunks(text, chunk_size=4000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text could be extracted from the PDF. \nPlease check if the file is valid and not empty.")
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to create a conversational chain
def get_conversational_chain(lang):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(input_variables=["question", "context"], template=template_en if lang == 'en' else template_hi)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to generate summary for a chunk of text
def generate_chunk_summary(text, lang):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    
    summary_prompt_en = """
    You are an AI assistant tasked with summarizing documents. Given the following text from a part of a document, 
    create a concise summary that captures the main points and key information. The summary should be informative 
    and focus on the most important aspects.

    Text:
    {text}

    Please provide a summary in about 2-3 sentences.
    """
    
    summary_prompt_hi = """
    आप एक AI सहायक हैं जिसे दस्तावेज़ों का सारांश बनाने का काम सौंपा गया है। दस्तावेज़ के एक भाग से निम्नलिखित पाठ दिया गया है, 
    एक संक्षिप्त सारांश बनाएं जो मुख्य बिंदुओं और महत्वपूर्ण जानकारी को कैप्चर करता है। सारांश सूचनात्मक होना चाहिए 
    और सबसे महत्वपूर्ण पहलुओं पर ध्यान केंद्रित करना चाहिए।

    पाठ:
    {text}

    कृपया लगभग 2-3 वाक्यों में एक सारांश प्रदान करें।
    """
    
    prompt = summary_prompt_en if lang == 'en' else summary_prompt_hi
    response = model.invoke(prompt.format(text=text))
    return response.content

# Function to generate summary for the entire document
def generate_full_summary(text, lang):
    chunks = get_text_chunks(text, chunk_size=4000, chunk_overlap=200)
    chunk_summaries = []
    
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Summarizing part {i+1}/{len(chunks)}..." if lang == 'en' else f"भाग {i+1}/{len(chunks)} का सारांश बना रहे हैं..."):
            chunk_summary = generate_chunk_summary(chunk, lang)
            chunk_summaries.append(chunk_summary)
    
    # Combine chunk summaries
    combined_summary = " ".join(chunk_summaries)
    
    # Generate final summary
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    final_summary_prompt_en = """
    You are an AI assistant tasked with creating a comprehensive summary. Given the following combined summaries 
    of different parts of a document, create a coherent and concise final summary that captures the main points 
    and key information of the entire document.

    Combined Summaries:
    {combined_summary}

    Please provide a final summary in about 5-7 sentences, ensuring all major points are covered.
    Finally , strictly ensure to provide the summary in a clean formatted strucuture , headers ,sub-headers, bullet points and even symbols , icons to express beautifully.
    """
    
    final_summary_prompt_hi = """
    आप एक AI सहायक हैं जिसे एक व्यापक सारांश बनाने का काम सौंपा गया है। दस्तावेज़ के विभिन्न भागों के निम्नलिखित संयुक्त सारांशों के आधार पर, 
    एक सुसंगत और संक्षिप्त अंतिम सारांश बनाएं जो पूरे दस्तावेज़ के मुख्य बिंदुओं और महत्वपूर्ण जानकारी को कैप्चर करता है।

    संयुक्त सारांश:
    {combined_summary}

    कृपया लगभग 5-7 वाक्यों में एक अंतिम सारांश प्रदान करें, यह सुनिश्चित करते हुए कि सभी प्रमुख बिंदुओं को कवर किया गया है।
    अंत में, सुनिश्चित करें कि सारांश को एक साफ-सुथरे और संरचित तरीके से प्रस्तुत करें, हेडर, सब-हेडर, बुलेट पॉइंट्स और यहां तक कि प्रतीकों, आइकनों का उपयोग करके सुंदरता से व्यक्त करें।
    """
    
    prompt = final_summary_prompt_en if lang == 'en' else final_summary_prompt_hi
    response = model.invoke(prompt.format(combined_summary=combined_summary))
    return response.content

# Function to process user input and display the response
def user_input(user_question, lang):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(lang)
    response = chain({"input_documents": docs, "question": user_question})
    st.write("*Answer: *" if lang == 'en' else "*उत्तर: *", response["output_text"])

# Streamlit app setup
st.set_page_config(
    page_title="Document Summarizer",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="auto"
)

def main():
    st.header("🖇️🖇️ DocDigest 📑📜")

    # File uploader in the main content area
    pdf_docs = st.file_uploader("Upload your Document(.pdf)", accept_multiple_files=False, type=['pdf'])

    if pdf_docs:
        if st.button("Generate Document Summary"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text([pdf_docs])  # Pass as a list
                if not raw_text.strip():
                    st.error("No text could be extracted from the PDF. Please check if the file is valid and not empty.")
                    return

                lang = detect_language(raw_text)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                
                if vector_store is None:
                    return

                # Generate and display summary
               summary = generate_full_summary(raw_text, lang)
                st.session_state['summary'] = summary
                st.session_state['pdf_processed'] = True
                st.session_state['lang'] = lang

                st.subheader("Document Summary")
                st.write(summary)
                st.divider()

                st.success("Summary Generated Successfully. ")

    # Display summary if available
    if 'summary' in st.session_state:
        st.subheader("Document Summary ")
        st.write(st.session_state['summary'])
        st.divider()

    # Question input
    user_question = st.text_input("Ask your question about the document ")

    if user_question and st.session_state.get('pdf_processed', False):
        user_input(user_question, st.session_state['lang'])
    elif user_question:
        st.warning("Please process a PDF before asking questions. ")

if __name__ == "__main__":
    main()
