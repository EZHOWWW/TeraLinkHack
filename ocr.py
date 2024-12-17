from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import pathlib
import textract
from document import Document


IMAGE_PATH = './test1.jpg'
DOCUMENTS_PATH = pathlib.Path('./Documents')


class OCR:
    def __init__(self, languages=['ru', 'en'], documents_path=DOCUMENTS_PATH):
        self.langs = languages
        self.det_processor, self.det_model = load_det_processor(), load_det_model()
        self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()

    def predictions(self, dock_path: str):
        dock_path = pathlib.Path(dock_path)
        res = None
        if dock_path.suffix in ('.jpg', '.jpeg', '.png'):
            res = self.predictions_image(str(dock_path))
        else:
            res = textract.process(
                str(dock_path), language=self.langs[0]).decode('utf-8')
        return res

    def predictions_image(self, image_path: str):
        image = Image.open(image_path)
        predictions = run_ocr([image], [self.langs], self.det_model,
                              self.det_processor, self.rec_model, self.rec_processor)
        return predictions

    def get_document(self, dock_path: str) -> Document:
        pred = self.predictions(dock_path)
        text = ''
        if isinstance(pred, list):
            pred = pred[0]
            text = '\n'.join([line.text for line in pred.text_lines])
        if isinstance(pred, str):
            text = pred
        return Document(text=text, name=pathlib.Path(dock_path).name,
                        predict=pred)
