from fastapi import APIRouter

from app.sap.create_model import model
from app.sap.config import settings

router = APIRouter(tags=[settings.api.tag])


