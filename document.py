from dataclasses import dataclass, field
import json
import types as tp

@dataclass(order=False, eq=False, repr=False)
class Document:
    '''
    class Document
    все модули меняют меняют поле text, 
    preprocessed_text нужен для классификатора (нужно хранить и то и то) (mb тоже в метадата ?), 
    в metadata будем хранить некотые теги для микросервисов (например тег : договор). 
    TODO: json
    '''
    text: str
    preprocessed_text: str = ''

    metadata : dict[str, str | int | list[str]] = field(default_factory=dict)

    def __eq__(self, other):
        return self.text == other.text
    def __repr__(self):
        return self.text
    
    @classmethod
    def parse_json(cls, file):
        pass
    @classmethod
    def to_json(cls, file):
        pass
