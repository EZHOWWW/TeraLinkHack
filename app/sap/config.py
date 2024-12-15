from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings


LOG_DEFAULT_FORMAT = "[%(asctime)s.%(msecs)03d] %(module)10s:%(lineno)-3d %(levelname)-7s - %(message)s"


class LoggingConfig(BaseModel):
    log_level: Literal[
        'debug',
        'info',
        'warning',
        'error',
        'critical',
    ] = 'info'
    log_format: str = LOG_DEFAULT_FORMAT


class ModelConfig(BaseModel):
    pass


class Api(BaseModel):
    tag: str = "SAP"
    prefix: str = "/sap"


class Settings(BaseSettings):
    model_config = ModelConfig()
    logging: LoggingConfig = LoggingConfig()
    api: Api = Api()


settings = Settings()
