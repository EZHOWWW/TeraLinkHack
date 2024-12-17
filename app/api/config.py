from pydantic import BaseModel


class Api(BaseModel):
    prefix: str = "/api"
    tags: list[str] = ["Api"]


settings = Api()
