from pydantic import BaseModel


class AppConfig(BaseModel):
    pass


settings = AppConfig()
