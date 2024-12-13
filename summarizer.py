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

    def sum_list(self, data:list[Document], save_sum = False)->list[str]:
        res = []
        for i in data:
            res.append(self.sum(i, save_sum))
        return res

    def sum(self, doc:Document, save_sum = False) -> str:
        s = self.pipe(doc.text)
        if save_sum:
            i.sum = s
        return s
 