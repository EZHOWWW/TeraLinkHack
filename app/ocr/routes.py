from fastapi import APIRouter

from app.classifier.create_model import model
from app.classifier.config import settings

router = APIRouter(tags=[settings.api.tag])


@router.get("/")
async def root():
    return {"Hello": "World"}
