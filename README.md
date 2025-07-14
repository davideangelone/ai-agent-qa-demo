
# 🧠 Agente AI demo per Document RAG QA (LangChain + Streamlit + Ollama)

Proof of Concept di un agente AI accessibile da web app, per caricare documenti PDF/TXT e consentire interrogazioni in linguaggio naturale sul loro contenuto.
Si basa su una pipeline RAG (Retrieval-Augmented Generation) implementata con [LangChain](https://github.com/langchain-ai/langchain), [Ollama](https://ollama.com/), e [Streamlit](https://streamlit.io/).
Utilizza modelli locali di LLM ed embeddings.


## 🚀 Funzionalità principali

- Caricamento documenti (PDF, TXT)
- Indicizzazione automatica via embeddings e vector store (`Chroma`)
- Query in linguaggio naturale sui documenti indicizzati
- Q&A generativa tramite RAG con supporto a LangChain chain types: `refine`, `map_reduce`, `stuff`
- Supporto `Ollama` per LLM
- Supporto multi-provider per embeddings (`Ollama` e `HuggingFace`)
- Interfaccia user-friendly con Streamlit
- Modelli eseguiti in locale

---

## 🧰 Requisiti

- Python 3.10+
- [Ollama](https://ollama.com/) installato localmente
- Modelli compatibili con Ollama (es. `mistral`, `phi3`, `llama3`) scaricabili dal client Ollama
- Nota opzionale: GPU con VRAM ≥ 8GB consigliata per modelli LLM

---

## ⚙️ Setup ambiente

**Crea ambiente virtuale**:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux
.venv\Scripts\activate.bat     # Windows
```

**Installa le dipendenze**:

```bash
pip install -r requirements.txt
```

> **Nota**: Le dipendenze principali includono:
> - langchain
> - langchain-community
> - langchain-ollama
> - chromadb
> - streamlit
> - transformers
> - pypdf

**Installa e avvia Ollama**:

Scarica e installa Ollama da [ollama.com](https://ollama.com/download).

Per scaricare il modello LLM Ollama preconfigurato, esegui:

```bash
ollama pull mistral
```

Per scaricare il modello Ollama di embeddings preconfigurato:

```bash
ollama pull nomic-embed-text
```

Se si cambiano i modelli Ollama nel file `config.py`, sarà necessario scaricarli con:
```bash
ollama pull <nome_modello>
```

Il modello di embeddings HuggingFace verrà scaricato automaticamente.

Nota per Windows:
necessario abilitare la modalità sviluppatore

---

## ▶️ Avvio dell'app

```bash
streamlit run main.py
```

Accedi poi su `http://localhost:8501` per usare l'applicazione.

---

## 📂 Struttura del progetto

```text
.
├── main.py                 # Streamlit frontend
├── pipeline.py             # RAG chain, vectorstore, embeddings
├── documents.py            # Caricamento e hash dei documenti
└── config.py               # Configurazioni path e modelli embeddings/LLM
```

Una volta caricati i documenti, verranno posti nella folder `docs` e verrà generato il file `index_manifest.json`.
I vectorstore di embeddings verranno salvati nel folder `vectorstore` per un successivo recupero

---

## ✍️ Esempi d'uso

1. Carica uno o più files TXT o PDF
2. Clicca su **"Controlla e indicizza documenti"**
3. Fai una domanda come:
   > "Quali sono i prerequisiti per l'installazione?"

Il modello restituirà una risposta basata sui contenuti del documento.

---

## 📚 Note tecniche

Il vector store usa Chroma in modalità persistente.

E' possibile configurare il modello LLM Ollama da utilizzare.
Per i modelli embeddings, è possibile scegliere tra un modello Ollama o HuggingFace, anch'essi configurabili.

Per ogni provider, viene creata una directory separata (`/vectorstore/ollama/`, `/vectorstore/huggingface/`).

La logica di RAG è configurabile (chain type `refine` di default) ed è integrata tramite LangChain RetrievalQA.


**Esempi di modelli Ollama e casi d’uso**

| Modello     | RAM        | Contesto | Use-case ideale                     |
|-------------|------------|----------|-------------------------------------|
| `phi3:mini` | <10 GB     | 128k     | RAG refactor e low RAM              |
| `mistral`   | 12–14 GB   | 8k       | QA strutturato, documenti medi      |
| `llama3:8b` | 16+ GB     | 8k       | Analisi complesse                   |
| `mixtral`   | 24+ GB     | 32k      | RAG map-reduce avanzato             |

