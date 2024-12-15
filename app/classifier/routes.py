from fastapi import APIRouter

from app.ocr.create_model import model
from app.ocr.config import settings

router = APIRouter(tags=[settings.api.tag])


