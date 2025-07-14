import os
import hashlib
import json

from langchain_community.document_loaders import PyPDFLoader, TextLoader

def compute_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_docs_manifest(folder_path="docs"):
    manifest = {}
    for filename in os.listdir(folder_path):
        if filename.endswith((".pdf", ".txt")):
            filepath = os.path.join(folder_path, filename)
            manifest[filename] = compute_file_hash(filepath)
    return manifest

def are_documents_up_to_date(manifest_path="index_manifest.json", folder_path="docs"):
    current_manifest = get_docs_manifest(folder_path)
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, "r") as f:
            saved_manifest = json.load(f)
        updated = current_manifest == saved_manifest
        return updated
    except Exception:
        print(f"Eccezione {Exception} in {manifest_path}")
        return False

def save_manifest(manifest_path="index_manifest.json", folder_path="docs"):
    manifest = get_docs_manifest(folder_path)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

def load_documents(folder_path="docs"):
    docs = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs