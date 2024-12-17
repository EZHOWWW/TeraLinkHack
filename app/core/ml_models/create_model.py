from pydantic import BaseModel

from app.core.ml_models.model import Model


def create_model(model: type, settings: BaseModel) -> Model:
    """
    Общий метод создания мльных моделей

    :param model: Тип модельки, которую создаем - класс
    :param settings: Настройки модели
    :return: Экземпляр предоставленной модели с соответствующими настройками
    """

    return model(**settings.model_dump())
