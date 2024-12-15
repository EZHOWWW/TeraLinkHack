from fastapi import APIRouter

from app.classifier.routes import router as service_router
from app.classifier.config import settings

router = APIRouter(
    prefix=settings.api.prefix,
)

router.include_router(service_router)
