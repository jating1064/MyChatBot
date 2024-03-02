import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY="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"


#Upload pdf files

st.header("My first chatbot")

with st.sidebar:
    st.title("Your documents")

    file=st.file_uploader("Upload a pdf file & start aking qyestions", type="pdf")

#Extract text

if file is not None:
    #Read the file
    pdf_reader= PdfReader(file)
    text=""

    for page in pdf_reader.pages:
        text+=page.extract_text()

    #st.write(text)

    #break it into chunks

    text_splitter= RecursiveCharacterTextSplitter(
        separators= "\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    #st.write(chunks)

    #generating enbeddings
    embeddings=OpenAIEmbeddings(openai_api_key= OPENAI_API_KEY)


    #creating Vector Store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    #get user question
    user_question = st.text_input("Type your question here")
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)

        #define LLM
        llm=ChatOpenAI(
            openai_api_key= OPENAI_API_KEY,
            temperature=0.2,
            max_tokens= 1000,
            model_name="gpt-3.5-turbo"
        )

        #Output Results
        #chain -> take the question, get relevant documents, pass it to LLM, generate the output

        chain= load_qa_chain(llm, chain_type="stuff")
        response=chain.run(input_documents=match,question=user_question)
        st.write(response)