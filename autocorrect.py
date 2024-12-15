from document import Document
# from spellchecker import SpellChecker
# from sbert_punc_case_ru import SbertPuncCase
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
import torch

# Зададим название выбронной модели из хаба
MODEL_NAME = 'UrukHan/t5-russian-spell'
MAX_INPUT = 1024


class Autocorrct:
    def __init__(self, language='ru'):
        self.tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
        self.spell_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.device = torch.device('cpu')
        # self.spell = SpellChecker(language='ru')
        # self.punct_model = SbertPuncCase()

    def correct(self, data: Document, replase=True) -> str:
        input_sequences = [data.text]
        task_prefix = "Spell correct: "
        encoded = self.tokenizer(
            [task_prefix + sequence for sequence in input_sequences],
            padding="longest",
            max_length=MAX_INPUT,
            truncation=True,
            return_tensors="pt",
        )
        predicts = self.spell_model.generate(**encoded.to(self.device))
        res = self.tokenizer.batch_decode(
            predicts, skip_special_tokens=True)[0]
        if replase:
            data.text = res
        return res

    '''
    def correct_spell(self, data: str) -> str:
        text = data.split(' ')
        for i, w in enumerate(text):
            cor = self.spell.correction(w)
            if cor is not None and cor != w:
                text[i] = cor
        return ' '.join(text)

    def correct_punctuation(self, data: str) -> str:
        return self.punct_model.punctuate(data)

    def correct(self, data: Document, replase=True) -> str:
        text = data.text
        text = self.correct_spell(text)
        text = self.correct_punctuation(text)
        if replase:
            data.text = text
        return text

    '''


if __name__ == '__main__':
    a = Autocorrct()
    text = 'ывсем привет выныканалетоп армии который публикует новости и это двадцать пятый день спец операций на украине ет самый главной новости российские военные ракетами кинжалы калибр уничтожили крупную военную топливную базу украины ракетным ударом по населенному пункту под жетамиром уничтжены более стаукраинских военных в две тысячи двадцать втором году'
    doc = Document(text=text)
    a.correct(doc)
    print(doc)
