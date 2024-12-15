import logging

import uvicorn
from fastapi import FastAPI

from app.classifier import router as classifier_router
from app.config import settings
from app.ocr import router as ocr_router
from app.sap import router as sap_router
from app.summarizer import router as summarizer_router

logging.basicConfig(
    # level=logging.INFO
    format=settings.logging.log_format,
)

app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(ocr_router)
app.include_router(sap_router)
app.include_router(classifier_router)
app.include_router(summarizer_router)

if __name__ == '__main__':
    uvicorn.run(
        "app:app",
        host=settings.run.host,
        port=settings.run.port,
        reload=True,
    )
