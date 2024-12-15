from fastapi import APIRouter

from app.summarizer.routes import router as service_router
from app.summarizer.config import settings

router = APIRouter(
    prefix=settings.api.prefix,
)

router.include_router(service_router)
