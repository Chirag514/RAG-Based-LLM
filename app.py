import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Chat with your Documents (PDFs & URLs)")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    if os.path.exists("./chroma_db"):
        try:
            st.session_state.vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        except Exception:
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key or groq_api_key == "your_groq_api_key_here":
    st.sidebar.warning("⚠️ Please add your GROQ_API_KEY to the .env file.")
    llm = None
else:
    llm = ChatGroq(api_key=groq_api_key, model_name="openai/gpt-oss-120b", temperature=0)

with st.sidebar:
    st.header("1. Insert Data")
    
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    web_url = st.text_input("Or enter a Website URL")
    
    if st.button("Process Documents"):
        if not llm:
            st.error("Please configure your GROQ API key first!")
            st.stop()
            
        with st.spinner("Processing documents... This might take a moment."):
            documents = []
            
            if uploaded_files:
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        loader = PyPDFLoader(tmp_path)
                        documents.extend(loader.load())
                    except Exception as e:
                        st.error(f"Error loading PDF: {e}")
                    finally:
                        os.remove(tmp_path)
                    
            if web_url:
                if not web_url.startswith("http"):
                    web_url = "https://" + web_url
                try:
                    loader = WebBaseLoader(web_url)
                    documents.extend(loader.load())
                except Exception as e:
                    st.error(f"Error loading URL: {e}")
                    
            if not documents:
                st.warning("No documents found. Please upload a PDF or enter a valid URL.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory="./chroma_db"
                    )
                else:
                    st.session_state.vector_store.add_documents(chunks)
                
                st.success(f"Successfully processed {len(documents)} documents into {len(chunks)} text chunks!")

st.header("2. Chat")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a question about your documents..."):
    if st.session_state.vector_store is None:
        st.error("Please process some documents first!")
    elif not llm:
        st.error("Please configure your GROQ API key first!")
    else:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.spinner("Thinking..."):
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
            
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know. "
                "Use concise and highly accurate answers."
                "\n\n"
                "Context: \n{context}"
            )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("View Sources"):
                    for i, doc in enumerate(response["context"]):
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "")
                        page_str = f" (Page {page})" if page else ""
                        st.write(f"**Source {i+1}:** {source}{page_str}")
                        st.caption(f'"{doc.page_content[:200]}..."')
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
