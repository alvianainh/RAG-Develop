from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import shutil
import json
import os
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from _1_parsing.main import convert_and_save, extract_nodes, download_file_from_supabase_parsing
from _2_image.main import process_markdown_files
from _3_chunking.main import process_markdown, list_markdown_files_from_supabase
from _4_embedding_store.main import store_chunks_in_supabase, list_chunked_json_files_from_supabase, download_file_from_supabase
from auth.dependencies import get_current_user
from auth.models import User
import logging
from dotenv import load_dotenv
from supabase import Client, create_client
from pydantic import BaseModel
from datetime import datetime
from collections import defaultdict
from fastapi import Body
from sqlalchemy.orm import Session
from database import get_db
import requests
from fastapi.responses import StreamingResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR_parsing = BASE_DIR.parent / "_1_parsing" / "temp"
os.makedirs(TEMP_DIR_parsing, exist_ok=True)

INPUT_DIR_parsing = TEMP_DIR_parsing / "input_pdfs"
OUTPUT_DIR_parsing = TEMP_DIR_parsing / "parsed"
os.makedirs(INPUT_DIR_parsing, exist_ok=True)
os.makedirs(OUTPUT_DIR_parsing, exist_ok=True)

TEMP_DIR_img = BASE_DIR.parent / "_2_image" / "temp"
os.makedirs(TEMP_DIR_img, exist_ok=True)
INPUT_DIR_img = TEMP_DIR_img / "parsed"
OUTPUT_DIR_img = TEMP_DIR_img / "img"
os.makedirs(INPUT_DIR_img, exist_ok=True)
os.makedirs(OUTPUT_DIR_img, exist_ok=True)

TEMP_DIR_chunking = BASE_DIR.parent / "_3_chunking" / "temp"
os.makedirs(TEMP_DIR_chunking, exist_ok=True)
INPUT_DIR_chunking = TEMP_DIR_chunking / "img"
OUTPUT_DIR_chunking = TEMP_DIR_chunking / "chunking"
os.makedirs(INPUT_DIR_chunking, exist_ok=True)
os.makedirs(OUTPUT_DIR_chunking, exist_ok=True)

TEMP_DIR_embedding = BASE_DIR.parent / "_4_embedding_store" / "temp"
os.makedirs(TEMP_DIR_embedding, exist_ok=True)
INPUT_DIR_embedding = TEMP_DIR_embedding / "embedding"
os.makedirs(INPUT_DIR_embedding, exist_ok=True)


def get_user_uuid_by_email(email: str) -> str:
    """Function to get the UUID for a user by email from the users table in Supabase."""
    response = supabase.table("users").select("id").eq("email", email).execute()
    if response.data:
        return response.data[0]["id"]
    else:
        raise ValueError(f"User with email {email} not found.")


def build_folder_tree(files, storage):
    tree = {}

    for file in files:
        parts = file['name'].split('/')
        current = tree

        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                current.setdefault('files', []).append({
                    "name": part,
                    "full_path": file['name'],
                    "download_url": storage.from_("docs").get_public_url(file['name'])
                })
            else:
                current = current.setdefault('folders', {}).setdefault(part, {})

    def format_tree(tree_dict):
        result = []
        for folder_name, content in tree_dict.get('folders', {}).items():
            result.append({
                "folder_name": folder_name,
                "contents": format_tree(content)
            })
        for file in tree_dict.get('files', []):
            result.append(file)
        return result

    return format_tree(tree)


def list_files_with_folder_structure(storage, prefix: str):
    all_items = storage.from_("docs").list(path="") 
    matching_items = [item for item in all_items if item["name"].startswith(prefix)]
    return build_folder_tree(matching_items, storage)


def list_files_recursively_by_prefix(storage, path=""):
    items = storage.from_("docs").list(path=path)
    all_files = []

    for item in items:
        item_path = f"{path}/{item['name']}" if path else item['name']

        if 'metadata' not in item or not item['metadata']:
            folder_contents = list_files_recursively_by_prefix(storage, item_path)
            if folder_contents:
                all_files.append({
                    "folder_name": item['name'],
                    "contents": folder_contents
                })
        else:
            file_url = storage.from_("docs").get_public_url(item_path)
            all_files.append({
                "name": item['name'],
                "full_path": item_path,
                "download_url": file_url
            })

    return all_files

def list_folder_structure(storage, path=""):
    """
    Fungsi untuk menelusuri folder dan file secara rekursif.
    """
    items = storage.from_("docs").list(path=path)
    result = []

    for item in items:
        item_name = item["name"]
        item_path = f"{path.rstrip('/')}/{item_name}"  

        if item_name.endswith("/"):
            folder_contents = list_folder_structure(storage, item_path)
            result.append({
                "folder_name": item_name.rstrip("/"),  
                "contents": folder_contents
            })
        else:
            result.append({
                "name": item_name,
                "full_path": item_path
            })

    return result

def recursive_list_files(bucket, path_prefix):
    all_files = []
    items = bucket.list(path=path_prefix)
    for item in items:
        item_name = item["name"]
        full_path = f"{path_prefix}/{item_name}"
        metadata = item.get("metadata")

        if not metadata or metadata.get("mimetype") is None:
            sub_files = recursive_list_files(bucket, full_path)
            all_files.extend(sub_files)
        else:
            all_files.append(full_path)

    return all_files


class SessionCreate(BaseModel):
    session_name: str | None = None 
    is_public: bool

# @router.get("/sessionadmin")
# def list_sessions(current_user: dict = Depends(get_current_user)):
#     """
#     Fungsi untuk mengembalikan daftar session untuk admin atau user.
#     """
#     user_id = current_user["email"]
#     role = current_user["role"]
#     storage = supabase.storage

#     all_sessions = []

#     if role == "admin":
#         # Ambil semua folder di root (untuk admin)
#         top_folders = storage.from_("docs").list(path="")

#         for folder in top_folders:
#             folder_name = folder["name"]
#             prefix = folder_name if folder_name.endswith("/") else f"{folder_name}/"
#             files = list_folder_structure(storage, prefix)
#             if files:
#                 all_sessions.append({
#                     "user_folder": folder_name,
#                     "contents": files
#                 })
#     else:
#         # Untuk user biasa, ambil folder khusus sesuai user_id
#         prefix = f"user_{user_id}/"
#         files = list_folder_structure(storage, prefix)
#         all_sessions.append({
#             "user_folder": prefix,
#             "contents": files
#         })

#     if not all_sessions:
#         raise HTTPException(status_code=404, detail="No sessions found.")

#     return all_sessions

@router.get("/admin/users")
def get_all_users(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Not authorized")

    users = db.query(User).all()
    
    
    result = []
    for user in users:
        result.append({
            "id": str(user.id),
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.user_role,
            "is_admin": user.is_admin,
            "created_at": user.created_at
        })

    return result

@router.post("/admin/set-role")
def set_user_role(
    email: str = Body(...), 
    new_role: str = Body(...),
    current_user: dict = Depends(get_current_user)
):
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    valid_roles = ["user", "sales", "editor", "manager"]
    if new_role not in valid_roles:
        raise HTTPException(status_code=400, detail="Invalid role")

    try:
        response = supabase.table("users").update({
            "user_role": new_role
        }).eq("email", email).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="User not found or no update made")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update role: {str(e)}")

    return {"message": f"Role for user {email} updated to {new_role}"}


@router.get("/allsessions")
def get_accessible_sessions(current_user: dict = Depends(get_current_user)):
    user_email = current_user["email"]
    
    user = supabase.table("users").select("id", "user_role").eq("email", user_email).single().execute()
    if not user.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_role = user.data["user_role"]
    user_id = user.data["id"]

    all_sessions = supabase.table("sessions").select("*").execute()
    accessible_sessions = []

    for session in all_sessions.data:
        is_public = session.get("is_public", False)
        allowed_roles = session.get("allowed_roles", [])
        created_by = session.get("created_by")
        is_global = session.get("is_global_context", False)
        context_session_ids = session.get("context_session_ids")

        if user_role == "admin":
            accessible_sessions.append(session)

        elif user_role == "sales":
            if is_public or user_role in allowed_roles:
                accessible_sessions.append(session)

        elif user_role == "user":
            if is_public or created_by == user_id:
                accessible_sessions.append(session)
        
        if context_session_ids and created_by == user_id:
            if session not in accessible_sessions:
                accessible_sessions.append(session)

        if is_global and created_by == user_id:
            if session not in accessible_sessions:
                accessible_sessions.append(session)

    return accessible_sessions

@router.get("/session/{session_id}")
def get_session_detail(session_id: str, current_user: dict = Depends(get_current_user)):
    user_email = current_user["email"]

    user = supabase.table("users").select("*").eq("email", user_email).single().execute()
    if not user.data:
        raise HTTPException(status_code=404, detail="User not found")

    user_id = user.data["id"]
    user_role = user.data["user_role"]

    session_res = supabase.table("sessions").select("*").eq("id", session_id).single().execute()
    if not session_res.data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = session_res.data
    is_owner = session_data["created_by"] == user_id
    is_public = session_data.get("is_public", False)
    allowed_roles = session_data.get("allowed_roles", [])

    if not (is_owner or is_public or user_role in allowed_roles or user.data["is_admin"]):
        raise HTTPException(status_code=403, detail="You do not have access to this session")

    documents = supabase.table("documents_track") \
        .select("id, file_name, file_path, uploaded_at") \
        .eq("session_id", session_id).execute().data


    chats = supabase.table("chat_history") \
        .select("id, role, message, timestamp") \
        .eq("session_id", session_id) \
        .eq("user_id", user_id) \
        .order("timestamp", desc=True).execute().data

    return {
        "session": {
            "id": session_data["id"],
            "name": session_data["session_name"],
            "created_by": session_data["created_by"],
            "created_at": session_data["created_at"],
            "is_public": session_data["is_public"],
            "allowed_roles": session_data["allowed_roles"]
        },
        "documents": documents,
        "chat_history": chats
    }

@router.get("/myfiles")
def get_user_uploaded_files(current_user: dict = Depends(get_current_user)):
    user_email = current_user["email"]


    user = supabase.table("users").select("id, user_role").eq("email", user_email).single().execute()
    if not user.data:
        raise HTTPException(status_code=404, detail="User not found")

    user_id = user.data["id"]
    role = user.data["user_role"]

    allowed_user_ids = []

    if role == "admin":
        all_users = supabase.table("users").select("id").execute()
        allowed_user_ids = [u["id"] for u in all_users.data]

    elif role == "sales":
        allowed_users = supabase.table("users") \
            .select("id") \
            .in_("user_role", ["sales", "admin"]) \
            .execute()
        allowed_user_ids = [u["id"] for u in allowed_users.data]

    elif role == "user":
        allowed_users = supabase.table("users") \
            .select("id") \
            .in_("user_role", ["admin"]) \
            .execute()
        allowed_user_ids = [u["id"] for u in allowed_users.data]
        allowed_user_ids.append(user_id)  

    else:
        raise HTTPException(status_code=403, detail="Unknown role")

    files_res = supabase.table("documents_track") \
        .select("id, file_name, file_path, user_id, session_id, uploaded_at") \
        .in_("user_id", allowed_user_ids) \
        .execute()

    if not files_res.data:
        return []

    bucket_name = "docs"
    file_list = []

    for file in files_res.data:
        file_path = file["file_path"]
        try:
            signed_url_res = supabase.storage.from_(bucket_name).create_signed_url(
                path=file_path, expires_in=3600
            )
            file_url = signed_url_res.get("signedURL") or signed_url_res.get("publicURL")
        except Exception:
            file_url = None

        file_list.append({
            "id": file["id"],
            "file_name": file["file_name"],
            "session_id": file.get("session_id"),
            "uploaded_at": file["uploaded_at"],
            "user_id": file["user_id"], 
            "file_url": file_url,
        })

    return file_list

@router.get("/download/{file_id}")
def download_file(file_id: str, current_user: dict = Depends(get_current_user)):
    user_email = current_user["email"]

    user = supabase.table("users").select("id, user_role").eq("email", user_email).single().execute()
    if not user.data:
        raise HTTPException(status_code=404, detail="User not found")

    user_id = user.data["id"]
    role = user.data["user_role"]

    file_res = supabase.table("documents_track") \
        .select("file_name, file_path, user_id, session_id") \
        .eq("id", file_id).single().execute()

    if not file_res.data:
        raise HTTPException(status_code=404, detail="File not found")

    file_owner_id = file_res.data["user_id"]
    file_path = file_res.data["file_path"]
    file_name = file_res.data["file_name"]
    session_id = file_res.data["session_id"]

    session_res = supabase.table("sessions") \
        .select("is_public, allowed_roles") \
        .eq("id", session_id).single().execute()

    if not session_res.data:
        raise HTTPException(status_code=404, detail="Session not found")

    is_public = session_res.data.get("is_public", False)
    allowed_roles = session_res.data.get("allowed_roles") or []

    is_authorized = False

    if is_public:
        is_authorized = True
    elif role in allowed_roles:
        is_authorized = True
    elif role == "admin":
        is_authorized = True
    elif role == "sales":
        allowed_users = supabase.table("users") \
            .select("id") \
            .in_("user_role", ["sales", "admin"]) \
            .execute()
        allowed_user_ids = [u["id"] for u in allowed_users.data]
        is_authorized = file_owner_id in allowed_user_ids
    elif role == "user":
        allowed_users = supabase.table("users") \
            .select("id") \
            .in_("user_role", ["admin"]) \
            .execute()
        allowed_user_ids = [u["id"] for u in allowed_users.data]
        allowed_user_ids.append(user_id)
        is_authorized = file_owner_id in allowed_user_ids

    if not is_authorized:
        raise HTTPException(status_code=403, detail="You do not have access to this file")

    signed_url_res = supabase.storage.from_("docs").create_signed_url(
        path=file_path,
        expires_in=3600
    )
    file_url = signed_url_res.get("signedURL")

    if not file_url:
        raise HTTPException(status_code=500, detail="Failed to get signed URL")

    response = requests.get(file_url, stream=True)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch file from storage")

    return StreamingResponse(
        response.raw,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'}
    )



@router.post("/create/session")
def create_session(data: SessionCreate, current_user: str = Depends(get_current_user)):

    user_email = current_user["email"]  
    
    user = supabase.table("users").select("id", "user_role").eq("email", user_email).single().execute()
    
    if not user.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_id = user.data["id"]
    user_role = user.data["user_role"]
    
    session_id = str(uuid4())
    session_name = data.session_name or f"Session-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    now_str = datetime.utcnow().isoformat()

    allowed_roles = []
    if user_role == "admin":
        is_public = True
        allowed_roles = []  
    elif user_role == "sales":
        is_public = False
        allowed_roles = ["admin", "sales"]
    elif user_role == "user":
        is_public = False
        allowed_roles = ["admin", "user"]

    else:
        raise HTTPException(status_code=403, detail="Invalid role")

    supabase.table("sessions").insert({
        "id": session_id,
        "user_id": user_id,
        "session_name": session_name,
        "created_at": now_str,
        "last_used": now_str,
        "is_public": is_public, 
        "allowed_roles": allowed_roles,
        "created_by": user_id
    }).execute()

    return {"session_id": session_id, "session_name": session_name}

@router.delete("/session/{session_id}")
def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    user_email = current_user["email"]

    root_folder_path = f"user_{user_email}/{session_id}"

    user = supabase.table("users").select("id", "user_role").eq("email", user_email).single().execute()
    if not user.data:
        raise HTTPException(status_code=404, detail="User not found")

    user_id = user.data["id"]

    session = supabase.table("sessions").select("*").eq("id", session_id).single().execute()
    if not session.data:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.data["created_by"] != user_id:
        raise HTTPException(status_code=403, detail="You are not allowed to delete this session")

    bucket = supabase.storage.from_(SUPABASE_BUCKET)
    file_paths_to_delete = recursive_list_files(bucket, root_folder_path)

    if file_paths_to_delete:
        bucket.remove(file_paths_to_delete)

    supabase.table("documents_chunk").delete().eq("session_id", session_id).execute()
    supabase.table("tables_chunk").delete().eq("session_id", session_id).execute()
    supabase.table("chat_history").delete().eq("session_id", session_id).execute()
    supabase.table("documents_track").delete().eq("session_id", session_id).execute()
    supabase.table("sessions").delete().eq("id", session_id).execute()

    # supabase.table("vecs.vec_table").delete().filter("metadata->>session_id", "eq", session_id).execute()
    # supabase.table("vecs.vec_text").delete().filter("metadata->>session_id", "eq", session_id).execute()


    return {"message": "Session deleted successfully"}

@router.post("/session/{session_id}/run_pipeline")
async def run_pipeline(
    session_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):

    chunking_results = []
    embedding_results = []

    user_id = current_user["email"]
    user_uuid = get_user_uuid_by_email(user_id) 
    try:
        file_location_parsing = f"user_{user_id}/{session_id}/input_pdfs/{file.filename}"
        file_content = await file.read()

        # Upload file for parsing
        supabase.storage.from_(SUPABASE_BUCKET).upload(file_location_parsing, file_content)
        logger.info(f"File uploaded to Supabase Storage at {file_location_parsing}")

        now_str = datetime.utcnow().isoformat()
        supabase.table("documents_track").insert({
            "session_id": session_id,
            "user_id": user_uuid,
            "file_name": file.filename,
            "file_path": file_location_parsing,
            "uploaded_at": now_str
        }).execute()

        file_name = file.filename
        file_path = f"input_pdfs/{file_name}"
        downloaded_file = download_file_from_supabase_parsing(file_path, current_user=user_id, session_id=session_id)

        if not downloaded_file:
            raise HTTPException(status_code=500, detail="File download failed.")

        convert_and_save(user_id=user_id, session_id=session_id)
        md_filename = Path(file.filename).with_suffix(".md")
        md_file_path = OUTPUT_DIR_parsing / f"{md_filename.stem}.md" 

        if md_file_path.exists():
            extract_nodes(md_file_path, user_id, session_id)

            # Upload markdown file
            with open(md_file_path, "r", encoding="utf-8") as md_file:
                md_content = md_file.read()
            md_file_location = f"user_{user_id}/{session_id}/parsed/{md_filename.name}"
            supabase.storage.from_(SUPABASE_BUCKET).upload(md_file_location, md_content.encode("utf-8"))

            # Upload artifacts
            artifact_dir = OUTPUT_DIR_parsing / f"{md_filename.stem}_artifacts"
            if artifact_dir.exists():
                for image_path in artifact_dir.glob("*"):
                    with open(image_path, "rb") as img_file:
                        img_content = img_file.read()
                    image_location = f"user_{user_id}/{session_id}/parsed/{md_filename.stem}_artifacts/{image_path.name}"
                    supabase.storage.from_(SUPABASE_BUCKET).upload(image_location, img_content)

            # Upload JSON nodes file
            json_file_path = OUTPUT_DIR_parsing / f"{md_filename.stem}_nodes.json"
            if json_file_path.exists():
                with open(json_file_path, "r", encoding="utf-8") as json_file:
                    json_content = json_file.read()
                json_location = f"user_{user_id}/{session_id}/parsed/{md_filename.stem}_nodes.json"
                supabase.storage.from_(SUPABASE_BUCKET).upload(json_location, json_content.encode("utf-8"))
            else:
                logger.error(f"JSON file not found: {json_file_path}")

        # Image processing
        image_processing_results = process_markdown_files(INPUT_DIR_img, OUTPUT_DIR_img, user_id, session_id)
        if "Error" in image_processing_results.get("message", ""):
            raise HTTPException(status_code=500, detail=image_processing_results["message"])

        skipped_files = [r for r in image_processing_results.get("results", []) if r["status"] == "skipped"]
        if skipped_files:
            logger.warning(f"Image processing skipped for some files: {skipped_files}")

        # Chunking
        files = list_markdown_files_from_supabase(user_id, session_id)
        if not files:
            raise HTTPException(status_code=404, detail="No markdown files found.")

        for file_name in files:
            url = process_markdown(file_name, user_id, session_id)
            chunking_results.append({"file_name": file_name, "json_url": url})

        # Embedding
        chunked_files = list_chunked_json_files_from_supabase(user_id, session_id)
        if not chunked_files:
            raise HTTPException(status_code=404, detail="No chunked JSON files found in Supabase.")


        results = []
        for file_name in chunked_files: 
            logger.info(f"Downloading: {file_name}")
            file_path_e = download_file_from_supabase(file_name, user_id, session_id)
            with open(file_path_e, "r", encoding="utf-8") as json_file:
                chunks = json.load(json_file)

            store_chunks_in_supabase(chunks)

            results.append({
                "file_name": file_name,
                "status": "stored"
            })

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for temp_dir in [TEMP_DIR_parsing, TEMP_DIR_embedding]:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Temporary directory {temp_dir} removed.")
                except Exception as e:
                    logger.warning(f"Failed to remove temp dir {temp_dir}: {e}")

    return {
        "message": "Pipeline completed successfully.",
        "chunking_results": chunking_results,
        "embedding_results": embedding_results,
        "image_processing_results": image_processing_results
    }
    
