from document import Document

import transformers
from transformers import pipeline
import tensorflow as tf


pipe = pipeline("summarization", model="d0rj/rut5-base-summ")
pipe("lalalalala")

model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
)


class Summarizer:
    """
    summarize main data from document, using small llm
    """
    def __init__(self, model:str, tokenizer: str | None):
        self.pipe = transformers.pipeline('summarization', model=model)

    def sum(self, data=tp.List[Document], save_sum = False) -> tp.List[str]:
        res = []
        for i in data:
            s = self.pipe(i.text)
            if save_sum:
                i.sum = s
            res.append(s)
        return res

