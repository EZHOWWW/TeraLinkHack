from fastapi import APIRouter

from app.sap.routes import router as service_router
from app.sap.config import settings

router = APIRouter(
    prefix=settings.api.prefix,
)

router.include_router(service_router)
