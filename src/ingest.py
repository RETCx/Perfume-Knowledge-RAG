from google.cloud import storage
import pandas as pd
import re
import pickle
import os
import tempfile
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
import uuid

load_dotenv()

TEMP_DIR = tempfile.gettempdir()
CSV_PATH = os.path.join(TEMP_DIR, "perfume_details.csv")
DOCS_PICKLE_PATH = os.path.join(TEMP_DIR, "langchain_docs.pkl")

def parse_accords_to_dict(text):
    if pd.isna(text) or text == "": 
        return {}
    matches = re.findall(r'([a-zA-Z\s_]+):([\d.]+)%', str(text))
    return {name.strip().lower(): float(v) for name, v in matches}

def dynamic_intensity_accords(accords_dict):
    descriptions = []
    sorted_accords = sorted(accords_dict.items(), key=lambda x: x[1], reverse=True)
    for name, v in sorted_accords:
        if v >= 90.0: intensity = "dominantly"
        elif v >= 75.0: intensity = "intensely"
        elif v >= 50.0: intensity = "strongly"
        elif v >= 25.0: intensity = "moderately"
        elif v >= 10.0: intensity = "lightly"
        else: intensity = "a hint of"
        descriptions.append(f"{intensity} {name}")
    
    if not descriptions:
        return ""
    return "The scent profile is " + ", ".join(descriptions) + "."


def load_data():
    df = pd.read_csv(CSV_PATH)
    return df



def create_langchain_documents(df):
    documents = []
    doc_ids = []
    for _, row in df.iterrows():
        # Handle Notes
        notes = []
        if pd.notna(row.get('Top_Notes')): notes.append(f"Top: {row['Top_Notes']}")
        if pd.notna(row.get('Middle_Notes')): notes.append(f"Mid: {row['Middle_Notes']}")
        if pd.notna(row.get('Base_Notes')): notes.append(f"Base: {row['Base_Notes']}")
        notes_full = " | ".join(notes)
        
        # Handle Accords
        accords_text = str(row.get('Main_Accords', ''))
        
        # Handle Wear stats
        when_to_wear = str(row.get('WhenToWear', ''))
        seasons = [s for s in ['winter', 'spring', 'summer', 'fall'] if f"{s}:" in when_to_wear]
        times = [t for t in ['day', 'night'] if f"{t}:" in when_to_wear]
        
        page_content = (
            f"Name: {row.get('Name', 'Unknown')}\n"
            f"Brand: {row.get('Brand', 'Unknown')}\n"
            f"Intensity Accords: {dynamic_intensity_accords(parse_accords_to_dict(accords_text))}\n"
            f"Notes: {notes_full}\n"
            f"Description: {row.get('Description', '')}\n"
            f"Accords Text: {accords_text}\n"
        )

        metadata = {
            "name": row.get('Name', 'Unknown'),
            "brand": row.get('Brand', 'Unknown'),
            "year" : row.get('Year', 0),
            "gender": row.get('Gender', 'unisex'),
            "rating": float(row.get('Rating', 0)) if pd.notna(row.get('Rating')) and str(row.get('Rating')).replace('.','',1).isdigit() else 0.0,
            "seasons": seasons,       
            "times": times,           
            "longevity": str(row.get('Longevity', '')),
            "sillage" : str(row.get('Sillage', '')),
            "accords": parse_accords_to_dict(accords_text)
        }
        
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

        # Qdrant requires IDs to be Unsigned Integers or valid UUIDs.
        raw_id = row.get('ID')
        if pd.notna(raw_id) and str(raw_id).replace('.','',1).isdigit():
            unique_id = int(float(raw_id)) # Fragrantica ID is an integer
        else:
            # Fallback to deterministic UUID if ID is missing
            name_str = f"{row.get('Brand', '')}_{row.get('Name', '')}"
            unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, name_str))
            
        doc_ids.append(unique_id)
        
    return documents, doc_ids

def ingest_into_qdrant(
    documents,
    doc_ids,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60.0
    )
    
    try:
        qdrant_client.create_collection(
            collection_name="perfumes",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print("Collection 'perfumes' created on Cloud.")
    except Exception as e:
        print("Collection already exists, proceeding to add documents.")

    # Payload Index
    try:
        # qdrant_client.http.models is already imported at the top
        qdrant_client.create_payload_index(
            collection_name="perfumes",
            field_name="metadata.gender",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        qdrant_client.create_payload_index(
            collection_name="perfumes",
            field_name="metadata.brand",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print("Payload Index for 'metadata.gender' verified/created.")
    except Exception as e:
        print("Payload index might already exist or error:", e)

    print("Uploading vectors to Qdrant Cloud... (This may take a minute)")
    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name="perfumes",
        embedding=embeddings
    )
    
    # add document to qdrant cloud
    vectorstore.add_documents(documents, ids=doc_ids) 
    print(f"{len(documents)} documents ingested into Qdrant Cloud")
    return vectorstore

def save_docs_pickle(documents, filepath):
    """Save LangChain Document objects to pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(documents, f)
    print(f"Documents saved to {filepath}")

def sync_with_gcs(bucket_name, download=True):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    if download:
        # Downloading latest CSV from GCS.
        blob = bucket.blob("perfume_details_full_final.csv")
        blob.download_to_filename(CSV_PATH)
        print("Downloaded latest CSV from GCS.")
    else:
        # Uploading Pickle back to GCS (Scale to Zero)
        blob = bucket.blob("langchain_docs.pkl")
        blob.upload_from_filename(DOCS_PICKLE_PATH)
        print("Uploaded latest BM25 Pickle to GCS.")



def run_pipeline(bucket_name):
    print("1. Downloading latest data from GCS...")
    sync_with_gcs(bucket_name, download=True)
    
    print("2. Loading data...")
    df = load_data()
    
    print("3. Creating LangChain documents...")
    documents, doc_ids = create_langchain_documents(df)
    
    print("4. Saving BM25 Pickle & Uploading to GCS...")
    save_docs_pickle(documents, DOCS_PICKLE_PATH)
    sync_with_gcs(bucket_name, download=False)
    
    print("5. Upserting into Qdrant...")
    ingest_into_qdrant(documents=documents, doc_ids=doc_ids) 
    print("Pipeline Complete!")

if __name__ == "__main__":
    print("Starting manual ingestion pipeline...")
    # It is best practice to keep bucket names out of source code.
    run_pipeline(bucket_name=os.getenv("GCP_BUCKET_NAME"))
