from abc import ABC, abstractmethod

from pydantic import BaseModel

from app.core.document import Document


class Model(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def process(self, data: Document) -> Document:
        pass

    @classmethod
    def create_model(cls, settings: BaseModel) -> "Model":
        """
        Общий метод создания мльных моделей

        :param settings: Настройки модели
        :return: Экземпляр предоставленной модели с соответствующими настройками
        """

        return Model(**settings.model_dump())
