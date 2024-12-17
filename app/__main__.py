import logging

import uvicorn

from app.api import app
from app.config import settings


logging.basicConfig(
    level=settings.logger.log_level,
    format=settings.logger.log_format,
)


if __name__ == '__main__':
    uvicorn.run(
        "app:app",
        host=settings.run.host,
        port=settings.run.port,
        reload=True,
    )