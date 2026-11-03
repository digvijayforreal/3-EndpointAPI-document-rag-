"""Streamlit interface for multi-PDF, document-scoped cloud RAG."""
import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from multi_pdf import (MAX_BYTES, MAX_DOCS, answer_question, connect_db,
                       ingest_pdf, list_documents)
from rag_chat import MODEL, REFUSAL

load_dotenv()
st.set_page_config(page_title="Document RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Document RAG Assistant")
st.caption("Jina Embeddings · MongoDB Atlas Vector Search · Groq " + MODEL)

required = ("MONGODB_USERNAME", "MONGODB_PASSWORD", "MONGODB_HOST", "JINA_API_KEY", "GROQ_API_KEY")
missing = [k for k in required if not os.getenv(k)]
if missing:
    st.error("Missing .env variables: " + ", ".join(missing))
    st.stop()

@st.cache_resource(show_spinner=False)
def connections():
    return connect_db()

try:
    mongo, collection = connections()
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    docs = list_documents(collection)
except Exception:
    st.error("Could not connect to MongoDB. Check Atlas network access, credentials, and connection.")
    st.stop()

with st.sidebar:
    st.header("Documents")
    st.caption(f"PDF only · up to 8 MB each · up to {MAX_DOCS} documents in this demo")
    uploads = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Index uploaded PDFs", disabled=not uploads, use_container_width=True):
        for upload in uploads or []:
            try:
                data = upload.getvalue()
                if len(data) > MAX_BYTES:
                    st.warning(f"{upload.name}: larger than 8 MB; skipped")
                    continue
                with st.spinner(f"Indexing {upload.name} through Jina..."):
                    result = ingest_pdf(collection, upload.name, data)
                st.success(f"{upload.name}: {result['status']}")
            except ValueError as exc:
                st.warning(f"{upload.name}: {exc}")
            except Exception as exc:
                # Never display exception text that could include credentials or request headers.
                st.error(f"{upload.name}: indexing failed ({type(exc).__name__}); check APIs and quota")
        st.rerun()

    docs = list_documents(collection)
    if not docs:
        st.info("Upload a text-based PDF to get started.")
        st.stop()
    labels = {d["document_id"]: f"{d['filename']} ({d['chunks']} chunks)" for d in docs}
    doc_ids = list(labels)
    chosen = st.selectbox("Answer using only this document", doc_ids,
                          format_func=lambda key: labels[key])
    st.caption("The existing sample.pdf is available if it is already indexed.")
    if st.button("Clear chat for this document", use_container_width=True):
        st.session_state.pop(f"messages_{chosen}", None)
        st.rerun()

key = f"messages_{chosen}"
if key not in st.session_state:
    st.session_state[key] = []

st.subheader(labels[chosen])
st.caption("Answers require supporting PDF excerpts. Unsupported questions receive a refusal; verification is not infallible.")
for message in st.session_state[key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Retrieved evidence"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}: {source['filename']}, page {source['page']}** — score {source['score']:.3f}")
                    st.text(source["text"])

question = st.chat_input("Ask about this PDF...")
if question:
    st.session_state[key].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching and validating document evidence..."):
            try:
                answer, chunks = answer_question(collection, groq_client, chosen, question)
                st.markdown(answer)
                if chunks:
                    with st.expander("Retrieved evidence"):
                        for i, source in enumerate(chunks, 1):
                            st.markdown(f"**Source {i}: {source['filename']}, page {source['page']}** — score {source['score']:.3f}")
                            st.text(source["text"])
            except Exception as exc:
                st.error(f"Request failed ({type(exc).__name__}). Check API quotas, connection, and vector index.")
                st.stop()
    st.session_state[key].append({"role": "assistant", "content": answer, "sources": chunks})
