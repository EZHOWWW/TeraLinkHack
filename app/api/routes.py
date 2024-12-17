import os

from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse

from app.core import Document, Application, APP_SETTINGS, create_app
from .config import settings

application: Application = create_app(APP_SETTINGS)
router = APIRouter(
    prefix=settings.prefix,
    tags=settings.tags,
)


@router.get("/download")
def download_result():
    return FileResponse(os.path.join(os.getcwd(), "Analyzed_doc.json"), filename='Analyzed_doc.json', media_type="application/json")


@router.post("/process_all")
def process_document(file: UploadFile) -> FileResponse:
    document = application.process_document(file)
    Document.to_json(document, "Analyzed_doc.json")
