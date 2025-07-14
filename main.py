import streamlit as st
import os
import shutil

from pipeline import load_vectorstore, build_vectorstore, create_rag_chain
from documents import load_documents, save_manifest, are_documents_up_to_date
from config import MODELS, get_vectorstore_path

st.set_page_config(page_title="RAG Document QA", layout="wide")
st.title("📄🔍 Document RAG QA")

# Mostra documenti già caricati
st.subheader("📁 Documenti già presenti:")
if os.path.exists("docs"):
    files = os.listdir("docs")
    if files:
        st.write(files)
    else:
        st.info("Nessun documento presente nella cartella.")
else:
    os.makedirs("docs")
    st.info("La cartella 'docs' è stata creata, ma è vuota.")

# Upload facoltativo
st.subheader("⬆️ Caricamento nuovi documenti (opzionale)")
uploaded_files = st.file_uploader("Carica documenti PDF o TXT", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join("docs", file.name), "wb") as f:
            f.write(file.read())
    st.success("File caricati!")

# Selezione provider (ollama o huggingface)
provider_options = ["-- Seleziona un provider --"] + list(MODELS.keys())

if not 'provider' in st.session_state:
    provider = st.selectbox("Seleziona il provider del modello di embeddings", options=provider_options, index=0)
    selected_index = provider_options.index(provider)
    if selected_index == 0:
        st.warning("Devi selezionare un provider valido per procedere.")
        st.stop()
    else:
        st.session_state.provider = provider
        st.success(f"Provider selezionato: {provider}")
else:
    provider = st.session_state.provider
    st.success(f"Provider selezionato: {provider}")

vectorstore_base_path = get_vectorstore_path(provider)

# Controlla se indicizzare
if st.button("🔄 Controlla e indicizza documenti"):
    with st.spinner("Controllo e indicizzazione in corso..."):
        docs = load_documents("docs")
        print(f"Numero di documenti caricati: {len(docs)}")

        if not are_documents_up_to_date():
            print(f"Documenti non aggiornati, aggiorno il vectorstore")
            if os.path.exists(vectorstore_base_path):
                shutil.rmtree(vectorstore_base_path)
            vs = build_vectorstore(docs, vectorstore_base_path, provider)
            print(f"Vectorstore aggiornato. Numero di chunks nel vectorstore: {len(vs._collection.get()['documents'])}")
            save_manifest()
        else:
            print(f"Documenti aggiornati")
            if not os.path.exists(vectorstore_base_path):
                print(f"Vectorstore non presente, procedo alla creazione")
                vs = build_vectorstore(docs, vectorstore_base_path, provider)
            else:
                print(f"Vectorstore presente e aggiornato, procedo al caricamento")
                vs = load_vectorstore(vectorstore_base_path, provider)

        print(f"Numero di chunks nel vectorstore: {len(vs._collection.get()['documents'])}")
        rag_chain = create_rag_chain(vs)
        st.session_state["qa_chain"] = rag_chain
        st.success("Indicizzazione completata!")

# Carica il vectorstore se esistente e non già in sessione
elif os.path.exists(vectorstore_base_path) and are_documents_up_to_date() and "qa_chain" not in st.session_state:
    vs = load_vectorstore(vectorstore_base_path, provider)
    rag_chain = create_rag_chain(vs)
    st.session_state["qa_chain"] = rag_chain

if "qa_chain" in st.session_state:
    query = st.text_input("Fai una domanda sul contenuto dei documenti:")
    if query:
        print(f"Sto generando la risposta per il prompt: {query}")
        with st.spinner("Sto generando la risposta..."):
            answer = st.session_state["qa_chain"].invoke(query)
            st.markdown(f"### 🧠 Risposta:\n{answer['result']}")
            print(f"Risposta generata")
