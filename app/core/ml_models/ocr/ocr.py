from app.core.ml_models.model import Model
from app.core.document import Document, EXAMPLE


class OCR(Model):
    def __init__(self, **kwargs):
        pass

    def from_file(self, file) -> Document:
        return EXAMPLE

    def process(self, data: Document) -> Document:
        return data
