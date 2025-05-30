import os
import json
import torch
import uuid
import numpy as np
import vecs
from dotenv import load_dotenv
from supabase import create_client, Client
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import logging
import shutil
import uuid


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DB_CONNECTION = os.getenv("DB_CONNECTION")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

vx = vecs.create_client(DB_CONNECTION)
vec_text = vx.get_or_create_collection(name="vec_text", dimension=768)
vec_table = vx.get_or_create_collection(name="vec_table", dimension=768)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
model = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True).to(device)

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

INPUT_DIR = TEMP_DIR / "embedding"

os.makedirs(INPUT_DIR, exist_ok=True)

def list_chunked_json_files_from_supabase(user_id: str, session_id: str) -> list:
    folder_path = f"user_{user_id}/{session_id}/chunking"
    response = supabase.storage.from_(SUPABASE_BUCKET).list(folder_path)
    
    return [f"{folder_path}/{f['name']}" for f in response if f['name'].endswith(".json")]



def download_file_from_supabase(file_name: str, current_user: str, session_id: str) -> Path | None:
    """Download file from Supabase Storage and save locally."""
    try:
        supabase_path = file_name
        file_bytes = supabase.storage.from_(SUPABASE_BUCKET).download(supabase_path)
        
        if not file_bytes:
            logger.error(f"Failed to download {file_name} from Supabase: Empty response")
            return None

        local_path = INPUT_DIR / file_name
        os.makedirs(local_path.parent, exist_ok=True)

        with open(local_path, "wb") as f:
            f.write(file_bytes)
        
        logger.info(f"File downloaded to {local_path}")
        return local_path

    except Exception as e:
        logger.error(f"Error downloading {file_name} from Supabase: {e}")
        return None

def get_embedding(text):
    """Generates an embedding vector from input text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()

def generate_table_description(table_data):
    """Generates a natural language description from a table's headers and rows."""
    headers = table_data["headers"]
    rows = table_data["rows"]
    description = [", ".join([f"{headers[i]}: {row[i]}" for i in range(len(headers))]) for row in rows]
    return " | ".join(description)

def convert_table_to_text(table_data, metadata):
    """Converts a table into a structured text format."""
    headers = ", ".join(table_data["headers"])
    rows = [" | ".join(row) for row in table_data["rows"]]
    table_title = metadata.get("table_title", "Unknown Table")
    table_text = f"{table_title}\nHeaders: {headers}\nRows:\n" + "\n".join(rows)
    table_description = generate_table_description(table_data)
    return table_text, table_description

def store_chunks_in_supabase(chunks: list[dict]):
    """Stores text and table chunks into Supabase."""
    document_entries, table_entries, text_records, table_records = [], [], [], []
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        metadata = chunk.get("metadata", {})
        user_id = metadata.get("user_id")  
        session_id = metadata.get("session_id") 

        if user_id and '@' in user_id:  
            user_id = get_user_uuid_by_email(user_id) 

        if "content" in chunk and chunk["content"]:
            embedding = get_embedding(chunk["content"])
            document_entries.append({
                "chunk_id": chunk_id,
                "content": chunk["content"],
                "metadata": metadata,
                "type": "text",
                "user_id": user_id, 
                "session_id": session_id 
            })
            text_records.append((chunk_id, embedding, {
                "user_id": user_id,
                "session_id": session_id
            }))

        if "table" in chunk and chunk["table"]:
            table_text, table_description = convert_table_to_text(chunk["table"], chunk.get("metadata", {}))
            table_embedding = get_embedding(table_text)
            table_entries.append({
                "chunk_id": chunk_id,
                "table_data": json.dumps(chunk["table"], ensure_ascii=False),
                "description": table_description,
                "metadata": metadata,
                "user_id": user_id,  
                "session_id": session_id  
            })
            table_records.append((chunk_id, table_embedding, {
                "user_id": user_id,
                "session_id": session_id
            }))
    
    if document_entries:
        supabase.table("documents_chunk").insert(document_entries).execute()
    if table_entries:
        supabase.table("tables_chunk").insert(table_entries).execute()
    vec_text.upsert(records=text_records)
    vec_table.upsert(records=table_records)


def get_user_uuid_by_email(email: str) -> str:
    """Function to get the UUID for a user by email from the users table in Supabase."""
    response = supabase.table("users").select("id").eq("email", email).execute()
    if response.data:
        return response.data[0]["id"]
    else:
        raise ValueError(f"User with email {email} not found.")

    

if __name__ == "__main__":

    # # BASE_DIR = Path(__file__).resolve().parent
    # # input_folder_e = BASE_DIR / "input_json"
    # for filename in os.listdir(input_folder_e):
    #     if filename.endswith(".json"):
    #         file_path = os.path.join(input_folder_e, filename)
    #         with open(file_path, "r", encoding="utf-8") as json_file:
    #             json_chunks = json.load(json_file)
    #         store_chunks_in_supabase(json_chunks)
    #         print(f"Processed and stored: {filename}")
    # print("All text and table embeddings stored successfully in Supabase!")

    store_chunks_in_supabase(chunks)