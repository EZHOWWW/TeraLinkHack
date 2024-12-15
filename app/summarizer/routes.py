from fastapi import APIRouter

from app.summarizer.create_model import model
from app.summarizer.config import settings

router = APIRouter(tags=[settings.api.tag])


