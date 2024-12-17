from app.core.document import Document
from app.core.ml_models.model import Model


class Autocorrect(Model):
    def __init__(self, **kwargs):
        pass

    def process(self, data: Document) -> Document:
        return data
