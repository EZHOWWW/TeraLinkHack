from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings

LOG_DEFAULT_FORMAT = "[%(asctime)s.%(msecs)03d] %(module)10s:%(lineno)-3d %(levelname)-7s - %(message)s"


class LoggingConfig(BaseModel):
    log_level: Literal[
        'DEBUG',
        'INFO',
        'WARNING',
        'ERROR',
        'CRITICAL',
    ] = 'INFO'
    log_format: str = LOG_DEFAULT_FORMAT


class RunConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class Settings(BaseSettings):
    run: RunConfig = RunConfig()
    logger: LoggingConfig = LoggingConfig()


settings = Settings()
