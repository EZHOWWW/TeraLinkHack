from document import Document
import tf_keras

import transformers

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

