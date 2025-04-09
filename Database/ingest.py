import chromadb
import json
import numpy as np
import ollama
import time
import os
import fitz  # PyMuPDF for PDF text extraction
from sentence_transformers import SentenceTransformer

# Initialize Chroma client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_or_create_collection(name="ds4300_notes")

EMBEDDING_VECTOR_DIMS = {
    "nomic": 768,
    "minilm": 384,
    "mpnet": 768
}

# Initialize embedding models
embedding_models = {
    "minilm": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    "mpnet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
    # "instructorxl": SentenceTransformer("hkunlp/instructor-xl")
}

def get_embedding(text: str, model_name: str) -> list:
    """Generate embeddings using the specified model."""
    model = embedding_models.get(model_name)

    start_time = time.time()

    if model_name == "nomic":
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        embedding = response["embedding"]
    else:
        model = embedding_models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found!")
        embedding = model.encode(text, normalize_embeddings=True).tolist()


    elapsed_time = time.time() - start_time
    return embedding, elapsed_time

def clear_chroma_store():
    """ used to clear the chroma db store """
    print("Clearing existing ChromaDb  store...")
    chroma_client.clear()
    print("ChromaDb store cleared.")

def store_embedding(collection, file, page, chunk, text, model_name):
    """Store embeddings for multiple models."""
    embedding, elapsed_time = get_embedding(text, model_name)
    
    id = f"{file}_page_{page}_chunk_{chunk}"
    
    collection.add(
        ids=[id],
        embeddings = [embedding],
        metadatas=[{
            "file": file,
            "page": page,
            "chunk": chunk,
            "text": text,
            "time": elapsed_time
        
        }]
    )

    print(f"Stored embedding for: {text}")



def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file by page."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def ingest_documents(collection, data_dir, embedding_model, chunk_size, overlap):
    """Process and ingest all PDFs in the specified directory."""
    start_time = time.time() 
    chunks_processed = 0

    for file_name in os.listdir(data_dir):
        pdf_path = os.path.join(data_dir, file_name)

        if file_name.endswith(".pdf"):
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                for chunk_index, chunk in enumerate(chunks):
                    store_embedding(collection, file_name, page_num, chunk_index, chunk, embedding_model)

        elif file_name.endswith(".json"): 
            with open(pdf_path, "r", encoding = "utf-8") as f: 
                recipes = json.load(f)
                
                max_items = 5
                for recipe_count, (recipe_id, details) in enumerate(list(recipes.items())[:max_items], 1):
                    recipe_details = json.dumps(details)

                    store_embedding(collection, file_name, recipe_id, recipe_count, recipe_details, embedding_model)
                    print(recipe_count)

            print(f"✅ Processed: {file_name}")   
    ingest_time = time.time() - start_time 

    return ingest_time

if __name__ == "__main__":
    print("\n--- Ingestion ---")
    # 
    # while True:
    #         embedding_model= input("Enter the embedding model to use for ingestion and querying (nomic/minilm/mpnet): ").strip().lower()
    #         if embedding_model in ['nomic','minilm', 'mpnet']:
    #             vector_dim = EMBEDDING_VECTOR_DIMS[embedding_model]
    #             break  # Exit the loop when a valid model is entered
    #         print("Invalid model. Please choose from: nomic, minilm, mpnet.")

    # chunk_size = int(input("Enter chunk size (default: 500): ") or 500)
    # overlap = int(input("Enter overlap size (default: 50): ") or 50)

    # Delete the collection if it exists
    try:
        chroma_client.delete_collection(name="ds4300_notes")
        print("Deleted existing collection 'ds4300_notes'.")
    except: 
        print("doesn't exist")
   
    # Create the collection with the correct dimensionality
    collection = chroma_client.create_collection(name="ds4300_notes")

    # Ingest documents 
    file_path = "/Users/vivianli/Documents/mktg_ai/project/rag-recipe-recommendation/Data"
    print("Ingesting PDFs")
    ingest_time = ingest_documents(collection, file_path, embedding_model='nomic', chunk_size=500, overlap=50)
    print("Document Ingestion Complete")

    print("\n--- Timing Data ---")
    print(f"Ingestion time: {ingest_time:.4f} sec")
    
