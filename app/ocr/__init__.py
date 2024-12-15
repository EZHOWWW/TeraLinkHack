from fastapi import APIRouter

from app.ocr.routes import router as service_router
from app.ocr.config import settings

router = APIRouter(
    prefix=settings.api.prefix,
)

router.include_router(service_router)
