from abc import ABC, abstractmethod

from app.core.document import Document


class Model(ABC):

    @abstractmethod
    def process(self, data: Document) -> Document:
        pass
