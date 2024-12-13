from document import Document
import types as tp
import transformers

MODELS = ["d0rj/rut5-base-summ",]

class Summarizer:
    """
    summarize main data from document, using small llm
    """
    def __init__(self, model:str, tokenizer: str | None = None):
        self.pipe = transformers.pipeline('summarization', model=model)

    def sum(self, data:list[Document], save_sum = False)->list[Document]:
        res = []
        for i in data:
            s = self.pipe(i.text)
            if save_sum:
                i.sum = s
            res.append(s)
        return res
 