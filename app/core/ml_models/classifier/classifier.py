from app.core.ml_models.model import Model
from app.core.document import Document


class Classifier(Model):
    def __init__(self, **kwargs):
        pass

    def process(self, data: Document) -> Document:
        return data
