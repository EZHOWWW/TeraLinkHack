import os.path

from pydantic import BaseModel

from dataclasses import dataclass, field, asdict
import json


@dataclass(order=False, eq=False, repr=False)
class Document():
    """
    class Document
    все модули меняют поле text,
    preprocessed_text нужен для классификатора (нужно хранить и то и то) (mb тоже в метадата ?),
    в metadata будем хранить некоторые теги для микросервисов (например тег : договор).
    """

    text: str
    name: str = ""
    preprocessed_text: str = ""
    sum: str = ""

    metadata: dict[str, str | int | list[str]] = field(default_factory=dict)

    def __eq__(self, other):
        return self.text == other.text

    def __repr__(self):
        return self.text

    @classmethod
    def parse_json(cls, file: str):
        with open(file, "r") as f:
            data = json.load(f)
        res = Document(**data)
        return res

    @classmethod
    def to_json(cls, doc, file: str):
        if doc.name == "":
            doc.name = file[:file.find(".")]
        with open(file, "w") as f:
            json.dump(asdict(doc), f)


EXAMPLE = Document.parse_json(os.getcwd() + "\\app\core\example.json")
