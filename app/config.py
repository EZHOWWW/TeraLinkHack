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


class RunConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080


class Settings(BaseSettings):
    run: RunConfig = RunConfig()
    logging: LoggingConfig = LoggingConfig()


settings = Settings()
