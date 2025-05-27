import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import uvicorn
import sys
import os
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging
from dotenv import load_dotenv
from supabase import create_client, Client
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from uuid import uuid4


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _5_retrieval_llm.main import query_supabase, call_openai_llm, get_accessible_session_ids
from auth.dependencies import get_current_user

# llm_app = FastAPI(title="Retrieval and LLM API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


router = APIRouter()

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DB_CONNECTION = os.getenv("DB_CONNECTION")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

def get_user_uuid_by_email(email: str) -> str:
    """Function to get the UUID for a user by email from the users table in Supabase."""
    response = supabase.table("users").select("id").eq("email", email).execute()
    if response.data:
        return response.data[0]["id"]
    else:
        raise ValueError(f"User with email {email} not found.")

def save_chat_message(user_id, session_id, role, message):
    supabase.table("chat_history").insert({
        "user_id": user_id,
        "session_id": session_id,
        "role": role,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()  
    }).execute()

def validate_user_access_to_session(user_uuid: str, session_id: str) -> None:
    session = supabase.table("sessions").select("*").eq("id", session_id).single().execute().data
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    is_owner = session["created_by"] == user_uuid
    is_public = session.get("is_public", False)
    allowed_roles = session.get("allowed_roles", [])
    user_role_resp = supabase.table("users").select("user_role").eq("id", user_uuid).single().execute()
    user_role = user_role_resp.data["user_role"] if user_role_resp.data else None

    if not (is_owner or is_public or user_role in allowed_roles):
        raise HTTPException(status_code=403, detail="You do not have access to this session's chat history")


def get_user_role_by_id(user_id: str) -> str:
    user = supabase.table("users").select("user_role").eq("id", user_id).single().execute()
    return user.data["user_role"] if user.data else None

def generate_session_name(context_session_ids, is_global_context):
    today_str = datetime.now().strftime("%d %b %Y %H:%M:%S")

    if is_global_context:
        return f"Global Context ({today_str})"

    elif context_session_ids:
        if len(context_session_ids) == 1:
            folder = supabase.table("sessions")\
                .select("session_name")\
                .eq("id", context_session_ids[0])\
                .limit(1)\
                .execute()
            folder_name = folder.data[0]["session_name"] if folder.data else "Unnamed"
            return f"Based on: {folder_name} ({today_str})"

        return f"Multi-Session ({today_str})"

    return f"Multi-Session ({today_str})"

def get_existing_virtual_context_session(user_id: str, context_session_ids: list[str]) -> Optional[str]:
    """
    Cek apakah sudah ada virtual context session dengan context_session_ids yang sama (urutan tidak masalah)
    """
    result = supabase.table("sessions")\
        .select("id, context_session_ids")\
        .eq("created_by", user_id)\
        .eq("is_global_context", False)\
        .execute()

    if not result.data:
        return None 

    for row in result.data:
        existing_ids = row.get("context_session_ids", [])
        if existing_ids is None:
            existing_ids = []
        if set(existing_ids) == set(context_session_ids):
            return row["id"]

    return None

def create_virtual_context_session(user_id: str, context_session_ids: list[str], force_new: bool = False) -> str:
    if not force_new:
        existing = get_existing_virtual_context_session(user_id, context_session_ids)
        if existing:
            return existing

    session_id = str(uuid4())
    now_str = datetime.utcnow().isoformat()
    session_name = generate_session_name(context_session_ids, is_global_context=False)

    supabase.table("sessions").insert({
        "user_id": user_id,
        "id": session_id,
        "session_name": session_name,
        "created_at": now_str,
        "last_used": now_str,
        "is_public": False,
        "allowed_roles": [],
        "created_by": user_id,
        "is_global_context": False,
        "context_session_ids": context_session_ids
    }).execute()

    return session_id

def get_or_create_global_context_session(user_id: str, force_new: bool = False) -> str:
    if not force_new:
        existing = supabase.table("sessions").select("id")\
            .eq("created_by", user_id)\
            .eq("is_global_context", True)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()

        if existing.data and len(existing.data) == 1:
            return existing.data[0]["id"]

    session_id = str(uuid4())
    now_str = datetime.utcnow().isoformat()

    session_name = generate_session_name([], is_global_context=True)


    supabase.table("sessions").insert({
        "user_id": user_id,  
        "id": session_id,
        "session_name": session_name,
        "created_at": now_str,
        "last_used": now_str,
        "is_public": False,
        "allowed_roles": [],
        "created_by": user_id,
        "is_global_context": True
    }).execute()

    return session_id

def is_valid_uuid(value):
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False

def are_valid_uuids(session_ids):
    try:
        return all(uuid.UUID(sid) for sid in session_ids)
    except:
        return False

def get_chat_history_by_session_id(session_id):
    resp = supabase.table("chat_history").select("role, message")\
        .eq("session_id", session_id).order("timestamp").execute()
    
    return [
        {"role": item["role"], "content": item["message"]}
        for item in resp.data
        if item.get("message") is not None and isinstance(item["message"], str)
    ]

class QueryRequest(BaseModel):
    user_query: str
    chat_history: list = []
    session_ids: Optional[List[str]] = None 
    force_new_global_session: Optional[bool] = False
    force_new_virtual_session: Optional[bool] = False


def sanitize(obj):
    if isinstance(obj, np.generic): 
        return obj.item()
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize(i) for i in obj)
    return obj


@router.get("/")
def root():
    return {"message": "Document Retrieval and LLM API is running."}

@router.post("/query")
def query_documents(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    email = current_user["email"]
    user_uuid = get_user_uuid_by_email(email)

    try:
        session_ids = request.session_ids or get_accessible_session_ids(supabase, user_uuid)

        retrieved_chunks = query_supabase(request.user_query, user_uuid, session_ids)
        sanitized_chunks = sanitize(retrieved_chunks)

        return {"retrieved_chunks": sanitized_chunks}

    except Exception as e:
        print("Error in /query:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# @router.post("/chat")
# def chat_with_llm(
#     request: QueryRequest,
#     current_user: dict = Depends(get_current_user)
# ):
#     email = current_user["email"]
#     user_uuid = get_user_uuid_by_email(email)

#     try:
#         session_ids = request.session_ids or get_accessible_session_ids(supabase, user_uuid)

#         retrieved_chunks = query_supabase(request.user_query, user_uuid, session_ids)

#         answer, chat_history = call_openai_llm(
#             request.user_query, retrieved_chunks, request.chat_history
#         )

#         if session_ids and len(session_ids) == 1 and is_valid_uuid(session_ids[0]):
#             save_chat_message(user_uuid, session_ids[0], "user", request.user_query)
#             save_chat_message(user_uuid, session_ids[0], "assistant", answer)

#         return {
#             "answer": answer,
#             "chat_history": chat_history
#         }

#     except Exception as e:
#         print("Error in /chat:", str(e))
#         raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/chat")
def chat_with_llm(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    email = current_user["email"]
    user_uuid = get_user_uuid_by_email(email)

    try:
        if request.session_ids is not None and len(request.session_ids) > 0:
            context_session_ids = request.session_ids

            if len(context_session_ids) == 1:
                session_id = context_session_ids[0]

                session_resp = supabase.table("sessions").select("created_by, is_public")\
                    .eq("id", session_id).execute()

                session_data = session_resp.data[0] if session_resp.data else None

                if session_data and session_data["created_by"] == user_uuid and not session_data.get("is_public", False):
                    save_session_id = session_id
                    session_type = "Single-session"
                else:
                    force_new = request.force_new_virtual_session if request.force_new_virtual_session else False
                    save_session_id = create_virtual_context_session(user_uuid, context_session_ids, force_new)
                    session_type = "Multi-session"
            else:
                force_new = request.force_new_virtual_session if request.force_new_virtual_session else False
                save_session_id = create_virtual_context_session(user_uuid, context_session_ids, force_new)
                session_type = "Multi-session"

        else:
            context_session_ids = get_accessible_session_ids(supabase, user_uuid)
            force_new = request.force_new_global_session if request.force_new_global_session else False
            save_session_id = get_or_create_global_context_session(user_uuid, force_new)
            session_type = "Global-context"

        if not context_session_ids:
            raise ValueError("context_session_ids is empty or None")
        
        if request.chat_history:
            recent_history = request.chat_history
        else:
            full_history = get_chat_history_by_session_id(save_session_id)
            recent_history = full_history[-6:] if len(full_history) >= 6 else full_history

        recent_history.append({"role": "user", "content": request.user_query})

        retrieved_chunks = query_supabase(request.user_query, user_uuid, context_session_ids)

        answer, _ = call_openai_llm(request.user_query, retrieved_chunks, recent_history)

        save_chat_message(user_uuid, save_session_id, "user", request.user_query)
        save_chat_message(user_uuid, save_session_id, "assistant", answer)

        return {
            "answer": answer,
            "chat_history": recent_history + [{"role": "assistant", "content": answer}],
            "used_session_id": save_session_id,
            "context_session_ids": context_session_ids,
            "session_type": session_type
        }


    except Exception as e:
        print("Error in /chat:", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/session/{session_id}/history")
def get_chat_history(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    user_uuid = get_user_uuid_by_email(current_user["email"])

    validate_user_access_to_session(user_uuid, session_id)

    try:
        response = supabase.table("chat_history") \
            .select("role, message, timestamp") \
            .eq("user_id", user_uuid) \
            .eq("session_id", session_id) \
            .order("timestamp", desc=False) \
            .execute()

        return {"chat_history": response.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")


# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8400)

