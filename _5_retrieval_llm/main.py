import os
import json
import torch
import uuid
import numpy as np
from supabase import create_client, Client
from transformers import AutoTokenizer, AutoModel
import ast
import re
import vecs
from dotenv import load_dotenv
import openai
from scipy.spatial.distance import cosine
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import psycopg2
import mlflow
import time
from pgvector.psycopg2 import register_vector
from pathlib import Path
from typing import List, Optional

from monitoring.decorator import monitor_rag_component
from monitoring.metrics import (
    measure_retrieval_metrics,
    measure_generation_metrics
)


nltk.download('all')
nltk.download('punkt')
nltk.download('stopwords')

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DB_CONNECTION = os.getenv("DB_CONNECTION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
openai.api_key = OPENAI_API_KEY

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

vx = vecs.create_client(DB_CONNECTION)
vec_text = vx.get_or_create_collection(name="vec_text", dimension=768)
vec_table = vx.get_or_create_collection(name="vec_table", dimension=768)

tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
model = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True).to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def get_embedding(text):
    """Generates an embedding vector from input text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()

def get_accessible_session_ids(supabase: Client, user_id: str):
    """Mengambil ID session yang dapat diakses oleh user berdasarkan role dan aturan session."""

    user_data = (
        supabase.table("users")
        .select("user_role, is_admin")
        .eq("id", user_id)
        .single()
        .execute()
    )

    if user_data.data is None:
        return []

    user_role = user_data.data["user_role"]
    is_admin = user_data.data["is_admin"]

    if is_admin:
        sessions = supabase.table("sessions").select("id").execute()
        if sessions.data is None:
            return []
        return [s["id"] for s in sessions.data]

    sessions = supabase.table("sessions").select("id, is_public, allowed_roles, created_by").execute()
    if sessions.data is None:
        return []
    
    accessible_ids = []

    for session in sessions.data:
        allowed_roles = session.get("allowed_roles", [])
        if isinstance(allowed_roles, str):
            allowed_roles = allowed_roles.split(",") 
        if (
            session["is_public"]
            or session["created_by"] == user_id
            or user_role in allowed_roles
        ):
            accessible_ids.append(session["id"])

    return accessible_ids


@monitor_rag_component(component_name="Retriever")
def query_supabase(user_query, user_id, session_ids=None):
    query_embedding = get_embedding(user_query)
    embedding_str = ','.join([str(x) for x in query_embedding])

    conn = psycopg2.connect(DB_CONNECTION)
    register_vector(conn)
    cur = conn.cursor()

    TOP_K = 20

    query_text = f"""
        SELECT id, 1 - (vec <=> ARRAY[{embedding_str}]::vector) AS similarity
        FROM vecs.vec_text
        ORDER BY vec <=> ARRAY[{embedding_str}]::vector
        LIMIT {TOP_K}
    """
    cur.execute(query_text)
    text_chunk_ids = cur.fetchall()

    text_results = []
    if text_chunk_ids:
        chunk_id_list = tuple([str(row[0]) for row in text_chunk_ids])
        cur.execute(f"""
            SELECT chunk_id, content, metadata, session_id
            FROM public.documents_chunk
            WHERE chunk_id IN %s;
        """, (chunk_id_list,))
        text_chunks = {row[0]: row[1:] for row in cur.fetchall()}

        for cid, sim in text_chunk_ids:
            if cid in text_chunks:
                chunk = text_chunks[cid]
                session_id = chunk[2]  
                if not session_ids or session_id in session_ids:
                    text_results.append((cid, "text", chunk[0], sim)) 

    query_table = f"""
        SELECT id, 1 - (vec <=> ARRAY[{embedding_str}]::vector) AS similarity
        FROM vecs.vec_table
        ORDER BY vec <=> ARRAY[{embedding_str}]::vector
        LIMIT {TOP_K}
    """
    cur.execute(query_table)
    table_chunk_ids = cur.fetchall()

    table_results = []
    if table_chunk_ids:
        chunk_id_list = tuple([str(row[0]) for row in table_chunk_ids])
        cur.execute(f"""
            SELECT chunk_id, description, metadata, session_id
            FROM public.tables_chunk
            WHERE chunk_id IN %s;
        """, (chunk_id_list,))
        table_chunks = {row[0]: row[1:] for row in cur.fetchall()}

        for cid, sim in table_chunk_ids:
            if cid in table_chunks:
                chunk = table_chunks[cid]
                session_id = chunk[2] 
                if not session_ids or session_id in session_ids:
                    table_results.append((cid, "table", chunk[0], sim)) 

    conn.close()

    combined_results = text_results + table_results
    combined_results.sort(key=lambda x: x[3], reverse=True)

    top5 = combined_results[:5]

    # Perfomance metrics
    if not top5:
        print("[DEBUG] No results found for the query.")
        measure_retrieval_metrics(top_item=[], embeddings=[])
        return []
    
    measure_retrieval_metrics(top_chunk=top5, get_embedding=get_embedding)

    return top5


@monitor_rag_component(component_name="LLM Generator")
def call_openai_llm(user_query, retrieved_chunks, chat_history=[]):
    """Send the query along with retrieved context and chat history to OpenAI API."""
    context_text = "\n\n".join([f"Chunk {i+1}: {chunk[2]}" for i, chunk in enumerate(retrieved_chunks)])
    messages = [
        {"role": "system", "content": "You are an intelligent assistant. Use the following retrieved information to answer the user's query."},
    ]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nUser's Question: {user_query}"})
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.7
    )
    answer = response.choices[0].message.content  
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": answer})

    # Performance metrics
    if not retrieved_chunks:
        print("[DEBUG] No context retrieved for the query.")
        return answer, chat_history

    measure_generation_metrics(
        user_query, 
        retrieved_chunks, 
        answer, 
        get_embedding
    )

    return answer, chat_history


def chat():
    """Handles continuous chat interaction with support for new chat and conversational context."""
    chat_history = [] 
    print("Welcome to the assistant! Type 'exit' to end the chat, 'new chat' to start over.")

    while True:
        user_query = input("User: ")

        if user_query.lower() in ["exit", "quit"]:
            print("Chat ended.")
            break
        
        if user_query.lower() == "new chat":
            chat_history = []  
            print("Starting a new chat...\n")
            continue  


        retrieved_chunks = query_supabase(user_query)

        answer, chat_history = call_openai_llm(user_query, retrieved_chunks, chat_history)
        
        print(f"Assistant: {answer}\n")



if __name__ == "__main__":
    chat()




