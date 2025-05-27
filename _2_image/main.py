import os
import re
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

INPUT_DIR = TEMP_DIR / "parsed"
OUTPUT_DIR = TEMP_DIR / "img"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_internvl_model():
    path = 'OpenGVLab/InternVL2_5-1B'
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  
        low_cpu_mem_usage=True,
        use_flash_attn=True if torch.cuda.is_available() else False,
        trust_remote_code=True
    ).eval().to(device) 

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer, device

def download_file_from_supabase(supabase_path: str, local_path: Path, supabase: Client, session_id: str) -> Path:
    """Download a single file from Supabase Storage and save locally."""
    try:
        file_bytes = supabase.storage.from_(SUPABASE_BUCKET).download(supabase_path)

        if not file_bytes:
            logger.error(f"Failed to download {supabase_path}: Empty response")
            return None

        os.makedirs(local_path.parent, exist_ok=True)

        with open(local_path, "wb") as f:
            f.write(file_bytes)
        
        logger.info(f"Downloaded {supabase_path} to {local_path}")
        return local_path

    except Exception as e:
        logger.error(f"Error downloading {supabase_path}: {e}")
        return None


def download_all_files_from_folder(user_id: str, supabase: Client, session_id: str) -> dict:
    """Download all files (including folders) from Supabase folder user_{user_id}/parsed to local INPUT_DIR."""
    try:
        base_supabase_path = f"user_{user_id}/{session_id}/parsed"
        logger.info(f"Listing contents of: {base_supabase_path} in bucket {SUPABASE_BUCKET}")
        items = supabase.storage.from_(SUPABASE_BUCKET).list(base_supabase_path)

        if not items:
            logger.warning(f"No files found in Supabase path: {base_supabase_path}")
            return {"error": "No files found in the folder."}

        downloaded_files = []

        for item in items:
            name = item["name"]
            full_supabase_path = f"{base_supabase_path}/{name}"
            logger.info(f"Processing item: {name}")

            if "." not in name:
                subfolder_path = f"{base_supabase_path}/{name}"
                try:
                    subitems = supabase.storage.from_(SUPABASE_BUCKET).list(subfolder_path)
                except Exception as e:
                    logger.warning(f"Could not list subfolder {subfolder_path}: {e}")
                    continue

                if not subitems:
                    logger.warning(f"Subfolder {subfolder_path} is empty or does not exist. Skipping.")
                    continue

                logger.info(f"Found subitems in folder {name}: {subitems}")

                for subitem in subitems:
                    sub_name = subitem["name"]
                    sub_supabase_path = f"{subfolder_path}/{sub_name}"
                    local_file_path = INPUT_DIR / name / sub_name
                    result = download_file_from_supabase(sub_supabase_path, local_file_path, supabase, session_id)
                    if result:
                        downloaded_files.append(str(result))
            else:
                local_file_path = INPUT_DIR / name
                result = download_file_from_supabase(full_supabase_path, local_file_path, supabase, session_id)
                if result:
                    downloaded_files.append(str(result))

        return {"message": "All files downloaded successfully.", "files": downloaded_files}

    except Exception as e:
        logger.error(f"Error downloading files: {e}")
        return {"error": str(e)}

def upload_file_to_supabase(file_name: str, local_path: Path, current_user: str, session_id: str) -> bool:
    """Upload updated file back to Supabase Storage."""
    try:
        if isinstance(local_path, str):
            local_path = Path(local_path)

        supabase_path = f"user_{current_user}/{session_id}/img/{file_name}" 
        logger.debug(f"Uploading {file_name} to Supabase at {supabase_path}")

        with open(local_path, "rb") as f:
            data = f.read()

        content_type = (
            "text/markdown" if local_path.suffix == ".md"
            else "application/json" if local_path.suffix == ".json"
            else "image/jpeg"
        )

        response = supabase.storage.from_(SUPABASE_BUCKET).upload(supabase_path, data, {"content-type": content_type})

        logger.debug(f"Supabase upload response: {response}")

        if response.get("error"):
            logger.error(f"Failed to upload {file_name} to Supabase: {response['error']}")
            return False
        elif response.get("Key"):  
            logger.info(f"File uploaded to Supabase at {supabase_path}")
            return True

        logger.error(f"Failed to upload {file_name} to Supabase: {response}")
        return False

    except Exception as e:
        logger.error(f"Error uploading {file_name} to Supabase: {e}")
        return False


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def load_image(image_file, input_size=448):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16 if torch.cuda.is_available() else torch.float32).to(device)
    return pixel_values

def extract_images_and_context(markdown_path):
    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    image_data = []
    for i, line in enumerate(lines):
        match = re.search(r'!\[.*?\]\((.*?)\)', line)
        if match:
            img_path = match.group(1)
            context_before = " ".join(lines[max(0, i-2):i]).strip()
            context_after = " ".join(lines[i+1:min(len(lines), i+3)]).strip()
            image_data.append((img_path, context_before, context_after))
    return image_data, lines

def generate_caption(model, tokenizer, image_path, context_before, context_after):
    if not os.path.exists(image_path):
        print(f"Warning: Image not found - {image_path}")
        return "[Image description unavailable]"

    pixel_values = load_image(image_path)
    prompt = (
        "<image>\n"
        f"Context before the image:\n{context_before}\n"
        f"Context after the image:\n{context_after}\n\n"
        "Please describe the image shortly."
    )
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return response

def update_markdown(markdown_path, image_data, lines, output_folder): 
    new_lines = []
    for line in lines:
        new_lines.append(line)
        match = re.search(r'!\[.*?\]\((.*?)\)', line)
        if match:
            img_path = match.group(1)
            caption = next((desc for img, _, _, desc in image_data if img == img_path), "[Image description unavailable]")
            new_lines.append(f"\n*Image Description:* {caption}\n")


    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, os.path.basename(markdown_path))
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def process_markdown_files(input_folder, output_folder, current_user, session_id: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model, tokenizer, device = load_internvl_model()

    download_result = download_all_files_from_folder(user_id=current_user, supabase=supabase, session_id=session_id)

    if "error" in download_result:
        print(f"Error downloading files: {download_result['error']}")
        return {"message": "Error downloading files from Supabase."}
    
    downloaded_files = download_result["files"]
    results = []

    for file_path in downloaded_files:
        if file_path.endswith(".md"):
            filename = os.path.basename(file_path)
            markdown_path = file_path 

            filename_without_ext = os.path.splitext(filename)[0]
            image_folder = os.path.join(input_folder, f"{filename_without_ext}_artifacts")

            enriched_data = []
            lines = []

            if os.path.exists(image_folder):
                image_data, lines = extract_images_and_context(markdown_path)
                for img_path, context_before, context_after in image_data:
                    full_image_path = os.path.join(image_folder, os.path.basename(img_path))
                    caption = generate_caption(model, tokenizer, full_image_path, context_before, context_after)
                    enriched_data.append((img_path, context_before, context_after, caption))
            else:
                with open(markdown_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                print(f"Info: Image folder '{image_folder}' not found for '{filename}', processing markdown without image enrichment.")

            output_path = os.path.join(output_folder, filename)
            update_markdown(markdown_path, enriched_data, lines, output_folder)
            upload_file_to_supabase(filename, output_path, current_user, session_id)
            print(f"Processed and uploaded: {filename}")
            results.append({"file": filename, "status": "processed"})

    try:
        shutil.rmtree(TEMP_DIR) 
        print(f"Folder {TEMP_DIR} has been removed successfully.")
    except Exception as e:
        print(f"Error removing folder {TEMP_DIR}: {e}")
    
    return {
        "message": "Image processing completed.",
        "results": results
    }
    
    # temp_folder = TEMP_DIR
    # try:
    #     if temp_folder.exists() and temp_folder.is_dir():
    #         shutil.rmtree(temp_folder)
    #         logger.info(f"Temporary folder {temp_folder} has been removed.")
    #     else:
    #         logger.warning(f"Temporary folder {temp_folder} does not exist or is not a directory.")
    # except Exception as e:
    #     logger.error(f"Error removing temporary folder {temp_folder}: {e}")

    # return {"message": "All markdown files processed successfully."}

if __name__ == "__main__":
    
    # BASE_DIR = Path(__file__).resolve().parent
    # markdown_folder = INPUT_DIR
    # output_folder = OUTPUT_DIR

    process_markdown_files(markdown_folder, output_folder, current_user, session_id)
