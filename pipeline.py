import os

from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import is_ollama_provider, is_huggingface_provider, get_embeddings_model_name, get_llm_model_name

"""
RAG strategy options con modelli Ollama:

1. phi3:mini
   - Context window: 128k
   - RAM: <10 GB
   - Ideale per RAG:
        stuff: quando il totale dei chunk rientra nel limite dei 128k
        refine: per un output iterativo ma senza tante chiamate parallele
   - Note: ottimo trade-off tra performance e leggerezza

2. llama3:8b / mixtral
    - Context window:
        llama3:8b: 8k token
        mixtral: 32k token (grazie alla routing architecture)
    - RAM:
        llama3:8b: ~16–20 GB VRAM
        mixtral: ~24+ GB VRAM (modello MoE, carica parzialmente solo 2/8 esperti)
    - Ideale per RAG:
        refine: su chunk di dimensioni controllate (2k–3k token max per evitare overflow)
        map_reduce: per fare summarization robusta con alta qualità nei singoli chunk
    - Note: qualità di generazione superiore, adatti a output finali, report, Q&A complesse

3. mistral o llama2:7b
    - Context window:
        mistral: 8k token
        llama2:7b: 4k–8k token (a seconda della build)
    - RAM:
        mistral: ~12–14 GB VRAM
        llama2:7b: ~10–12 GB VRAM
    - Ideale per RAG:
        map_reduce: chunk piccoli (fino a 1k–1.5k token) con riduzione finale
        stuff: solo se i documenti sono molto brevi (es. FAQ, snippet)
    - Note: Leggeri, ma perdono precisione nei contesti lunghi
            Ottimi per prototipi, sviluppo e test, ma rischiano di "allucinare" su dati complessi
            
            
LangChain type:
1. stuff:
    Concatena tutti i documenti in un unico prompt e li "infila" direttamente nel modello LLM.
    Va bene per documenti brevi. Se il limite nel numero di token è superato, il prompt supera il context window e perde informazioni
    
2. map_reduce:
    Utilizza un processo in due fasi.
    In una prima fase, ogni chunk viene interrogato e genera una risposta parziale;
    In una seconda fase, tutte le risposte parziali vengono combinate in un output finale.
    È utile quando si lavora su documenti molto lunghi, perché permette di "riassumere" le informazioni pezzo per pezzo per poi combinarle.
    
3. refine:
    Inizia con una risposta iniziale ottenuta da un chunk e, iterativamente, “raffina” quella risposta man mano che viene processato il testo successivo.
    È utile se l’ordine dei chunk è importante, o per mantenere una coerenza progressiva, o se documenti hanno informazioni parziali distribuite.
"""

ollama_llm_model_name = "mistral"
langchain_type = "refine"

# Configurazione Chroma locale senza telemetria
settings = Settings(anonymized_telemetry=False)

def get_embeddings(provider):
    embeddings_model_name = get_embeddings_model_name(provider)
    print(f"Utilizzo provider embeddings: {provider}")

    if is_ollama_provider(provider):
        return OllamaEmbeddings(model=embeddings_model_name)
    elif is_huggingface_provider(provider):
        return  HuggingFaceEmbeddings(
            model_name=embeddings_model_name,
            model_kwargs={"device": "cpu"} # Forza l'uso della CPU
        )
    else:
        raise ValueError('Provider "{}" not recognized.'.format(provider))

def build_vectorstore(documents, persist_directory, provider):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        chunks,
        embedding=get_embeddings(provider),
        persist_directory=persist_directory,
        client_settings=settings,
        collection_name="document_collection"
    )

    path = os.path.abspath(persist_directory)
    if os.path.exists(path):
        print(f"Vectorstore salvato in: {os.path.abspath(persist_directory)}")
    else:
        print(f"Vectorstore NON presente in: {os.path.abspath(persist_directory)}")

    return vectorstore

def load_vectorstore(persist_directory, provider):
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embeddings(provider),
        client_settings=settings,
        collection_name="document_collection"
    )

def create_rag_chain(vectorstore, chain_type=langchain_type):
    retriever = vectorstore.as_retriever()
    llm = OllamaLLM(model=get_llm_model_name())

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type=chain_type
    )
    return qa_chain
