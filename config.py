OLLAMA_PROVIDER = 'ollama'
HUGGINGFACE_PROVIDER = 'huggingface'
OLLAMA_LLM_MODEL = "mistral"

MODELS = {
    OLLAMA_PROVIDER: {
        "embeddings": "nomic-embed-text",
        "vectorstore_base_path": "./vectorstore/ollama/"
    },
    HUGGINGFACE_PROVIDER: {
        "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
        "vectorstore_base_path": "./vectorstore/huggingface/"
    }
}

def is_ollama_provider(provider):
    return provider == OLLAMA_PROVIDER

def is_huggingface_provider(provider):
    return provider == HUGGINGFACE_PROVIDER

def get_embeddings_model_name(provider):
    try:
        return MODELS[provider]["embeddings"]
    except KeyError:
        raise ValueError(f"Provider non supportato: {provider}")

def get_llm_model_name():
    return OLLAMA_LLM_MODEL

def get_vectorstore_path(provider):
    try:
        return MODELS[provider]["vectorstore_base_path"]
    except KeyError:
        raise ValueError(f"Provider non supportato: {provider}")

