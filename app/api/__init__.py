__all__ = (
    'app',
)

from fastapi import FastAPI

from .routes import router
from .config import settings

app = FastAPI()
app.include_router(router)
