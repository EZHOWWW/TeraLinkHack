from pydantic import BaseModel

from app.core.ml_models import Model
from app.core.ml_models import OCR, OCR_SETTINGS
from app.core.ml_models import Autocorrect, AUTOCORRECT_SETTINGS
from app.core.ml_models import Classifier, CLASSIFIER_SETTINGS
from app.core.ml_models import Summarizer, SUMMARIZER_SETTINGS

from app.core.document import Document


class Application:

    ocr_model: OCR = None
    autocorrect: Autocorrect = None
    classifier: Classifier = None
    summarizer: Summarizer = None

    def __init__(
            self,
            ocr_settings: BaseModel = OCR_SETTINGS,
            autocorrect_settings: BaseModel = AUTOCORRECT_SETTINGS,
            classifier_settings: BaseModel = CLASSIFIER_SETTINGS,
            summarizer_settings: BaseModel = SUMMARIZER_SETTINGS,
    ):
        self.ocr_model = OCR.create_model(ocr_settings)
        self.autocorrect = Autocorrect.create_model(autocorrect_settings)
        self.classifier = Classifier.create_model(classifier_settings)
        self.summarizer = Summarizer.create_model(summarizer_settings)

    def process_document(self, file) -> Document:
        document = self.ocr_model.from_file(file)
        document = self.ocr_model.process(document)
        document = self.autocorrect.process(document)
        document = self.classifier.process(document)
        document = self.summarizer.process(document)
        return document


def create_app(settings: BaseModel) -> Application:
    return Application(**settings.model_dump())
