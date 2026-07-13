from google.cloud import storage
import pickle
import tempfile
import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from tenacity import retry, stop_after_attempt, wait_exponential

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=15))
def init_qdrant():
    print("[System] Connecting to Qdrant (Waking up cluster if paused...)")
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), timeout=60.0)
    # Ping the server to force wake up
    client.get_collections()
    return QdrantVectorStore(
        client=client,
        collection_name="perfumes",
        embedding=embeddings
    )

vectorstore = init_qdrant()

# We define BM25 logic below and trigger it dynamically on startup
bm25 = None
    
def hybrid_search_perfume(query: str, style_preference: str, notes_or_vibes: str, excluded_notes: str = "") -> str:
    """Perform hybrid search (or vector search fallback) for perfumes"""
    must_conditions = []
    must_not_conditions = []
    
    search_filter = None
    style = style_preference.lower()
    if style in ['male', 'masculine']:
        must_conditions.append(
            models.FieldCondition(key="metadata.gender", match=models.MatchAny(any=["male", "unisex"]))
        )
    elif style in ['female', 'feminine']:
        must_conditions.append(
            models.FieldCondition(key="metadata.gender", match=models.MatchAny(any=["female", "unisex"]))
        )
    qdrant_filter = None
    if must_conditions:
        qdrant_filter = models.Filter(
            must=must_conditions
        )
    combined_search_term = f"{query} {notes_or_vibes}".strip()
    
    search_kwargs = {"k": 20}
    if qdrant_filter:
        search_kwargs["filter"] = qdrant_filter

    filtered_vector_retriever = vectorstore.as_retriever(
        search_kwargs=search_kwargs
    )
    
    if bm25 is None:
        print("Warning: BM25 not found, falling back to Vector Search only.")
        raw_hybrid_results = filtered_vector_retriever.invoke(combined_search_term)
    else:
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25, filtered_vector_retriever],
            weights=[0.6, 0.4]
        )
        raw_hybrid_results = hybrid_retriever.invoke(combined_search_term)
    
    strict_filtered_results = []
    excluded_list = [n.strip().lower() for n in excluded_notes.split(',')] if excluded_notes else []
    
    for doc in raw_hybrid_results:
        doc_gender = str(doc.metadata.get('gender', '')).lower()
        if style in ['male', 'masculine'] and doc_gender not in ['male', 'unisex']:
            continue
        elif style in ['female', 'feminine'] and doc_gender not in ['female', 'unisex']:
            continue
            
        accords_dict = doc.metadata.get('accords', {})
        has_excluded = any(accords_dict.get(ex_note, 0) > 0 for ex_note in excluded_list if ex_note)
        
        if has_excluded:
            continue
            
        strict_filtered_results.append(doc)
   
    
    if not strict_filtered_results:
        return "No result found"
    else:
        print("Filtered results are available, returning top 5...")
        final_results = strict_filtered_results[:5]
        formatted_texts = []
        for i, doc in enumerate(final_results, 1):
            name = doc.metadata.get("name", "Unknown")
            brand = doc.metadata.get("brand", "Unknown")
            gender_meta = doc.metadata.get("gender", "Unknown")
            clean_content = doc.page_content.replace('\n', ' | ')
            
            formatted_texts.append(
                f"[{i}] Name: {name} | Brand: {brand} | Gender: {gender_meta}\n"
                f"Content: {clean_content}"
            )
            
        return "\n\n".join(formatted_texts)


def get_bm25_path():
    return os.path.join(tempfile.gettempdir(), "langchain_docs.pkl")

def load_bm25_from_disk():
    global bm25
    try:
        with open(get_bm25_path(), 'rb') as f:
            documents = pickle.load(f)
        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = 10
        print("BM25 Loaded Successfully from temp dir!")
    except FileNotFoundError:
        print("Error: langchain_docs.pkl not found in temp dir.")
        bm25 = None

def reload_bm25():
    try:
        bucket_name = os.getenv("GCP_BUCKET_NAME")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob("langchain_docs.pkl")
        blob.download_to_filename(get_bm25_path())
        print("Downloaded fresh BM25 from GCS to temp dir.")
        load_bm25_from_disk()
    except Exception as e:
        print(f"Failed to reload BM25 from GCS: {e}")


# Initialization: Always try to fetch the latest BM25 from GCS when server starts
reload_bm25()

# test
if __name__ == "__main__":
    result = hybrid_search_perfume(
        query="fresh scent", 
        style_preference="masculine", 
        notes_or_vibes="apple, woody",
        excluded_notes="vanilla"
    )
    print(result)
    if hasattr(vectorstore, 'client'):
        vectorstore.client.close()