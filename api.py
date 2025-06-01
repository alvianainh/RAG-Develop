import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Request, Depends, HTTPException, status
from fastapi.responses import RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
import importlib.util
import requests
import shutil
import os
import mlflow
import httpx
from pathlib import Path
from fastapi.openapi.utils import get_openapi
# from fastapi.security import OAuth2PasswordBearer
from monitoring.setup import configure_monitoring, get_lifespan

configure_monitoring()


def import_router(path): 
    spec = importlib.util.spec_from_file_location("api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.router


parsing_router = import_router(os.path.abspath("_1_parsing/api.py"))
image_router = import_router(os.path.abspath("_2_image/api.py"))
chunking_router = import_router(os.path.abspath("_3_chunking/api.py"))
embedding_router = import_router(os.path.abspath("_4_embedding_store/api.py"))
llm_router = import_router(os.path.abspath("_5_retrieval_llm/api.py"))
pipeline_router = import_router(os.path.abspath("pipeline_runner/pipeline.py"))
auth_router = import_router(os.path.abspath("auth/api.py"))


# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

app = FastAPI(title="Pipeline API", lifespan=get_lifespan())
# app.add_middleware(AdminOnlyMiddleware, protected_prefix="/mlflow")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# # <== Di bawah ini tempelkan custom_openapi
# def custom_openapi():
#     if app.openapi_schema:
#         return app.openapi_schema
#     openapi_schema = get_openapi(
#         title="Pipeline API",
#         version="1.0.0",
#         description="API untuk RAG Development",
#         routes=app.routes,
#     )
#     openapi_schema["components"]["securitySchemes"] = {
#         "BearerAuth": {
#             "type": "http",
#             "scheme": "bearer",
#             "bearerFormat": "JWT",
#         }
#     }
#     # Menambahkan security untuk semua path yang ada
#     for path in openapi_schema["paths"].values():
#         for method in path.values():
#             method.setdefault("security", [{"BearerAuth": []}])
#     app.openapi_schema = openapi_schema
#     return app.openapi_schema

# app.openapi = custom_openapi


app.include_router(auth_router, prefix="/auth", tags=["auth"])

app.include_router(pipeline_router, prefix="/pipeline", tags=["pipeline"])

# app.include_router(parsing_router, prefix="/upload")
# app.include_router(image_router, prefix="/img")
# app.include_router(chunking_router, prefix="/markdown")
# app.include_router(embedding_router, prefix="/store")
app.include_router(llm_router, prefix="/llm")

@app.get("/")
def read_root():
    return {"message": "Pipeline API is running!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
