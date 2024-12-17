import logging

from app.api import app
from app.config import settings


logging.basicConfig(
    level=settings.logger.log_level,
    format=settings.logger.log_format,
)
