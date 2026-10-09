import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace

st.set_page_config(page_title="Document Q&A Chatbot")
st.title("Document Q&A Chatbot")


# ---------- Load models (cached so they load only once) ----------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_llm():
    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 200, "return_full_text": False},
    )
    return ChatHuggingFace(llm=llm)


# ---------- Step 1: Text extraction ----------
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file.read().decode("utf-8")  # .txt file


# ---------- Step 2-4: Chunking, embedding, FAISS vector store ----------
def build_vector_store(files):
    text = ""
    for f in files:
        text += extract_text(f) + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    return FAISS.from_texts(chunks, load_embeddings())


# ---------- Step 5-6: Retrieve chunks and generate answer ----------
def answer_question(question):
    docs = st.session_state.vector_store.similarity_search(question, k=3)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say \"I don't know\".\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return load_llm().invoke(prompt).content, context


# ---------- Streamlit UI ----------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: document upload
with st.sidebar:
    st.header("Upload Documents")
    files = st.file_uploader("PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Process Documents") and files:
        with st.spinner("Processing..."):
            st.session_state.vector_store = build_vector_store(files)
            st.session_state.messages = []
        st.success("Documents ready. Ask your questions!")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
question = st.chat_input("Ask a question about your documents")
if question:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, context = answer_question(question)
            st.write(answer)
            with st.expander("Context used"):
                st.write(context)

        st.session_state.messages.append({"role": "assistant", "content": answer})