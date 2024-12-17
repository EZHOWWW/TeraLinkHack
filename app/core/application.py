from pydantic import BaseModel

from app.core.ml_models import Model
from app.core.ml_models import OCR, OCR_SETTINGS
from app.core.ml_models import Autocorrect, AUTOCORRECT_SETTINGS
from app.core.ml_models import Classifier, CLASSIFIER_SETTINGS
from app.core.ml_models import Summarizer, SUMMARIZER_SETTINGS
from app.core.ml_models import create_model

from app.core.document import Document


class Application:

    ocr_model: Model = None
    autocorrect: Model = None
    classifier: Model = None
    summarizer: Model = None

    def __init__(
            self,
            ocr_settings: BaseModel = OCR_SETTINGS,
            autocorrect_settings: BaseModel = AUTOCORRECT_SETTINGS,
            classifier_settings: BaseModel = CLASSIFIER_SETTINGS,
            summarizer_settings: BaseModel = SUMMARIZER_SETTINGS,
    ):
        self.ocr_model = create_model(OCR, ocr_settings)
        self.autocorrect = create_model(Autocorrect, autocorrect_settings)
        self.classifier = create_model(Classifier, classifier_settings)
        self.summarizer = create_model(Summarizer, summarizer_settings)

    def process_document(self, file) -> Document:
        document = self.ocr_model.from_file(file)
        document = self.ocr_model.process(document)
        document = self.autocorrect.process(document)
        document = self.classifier.process(document)
        document = self.summarizer.process(document)
        return document


def create_app(settings: BaseModel) -> Application:
    return Application(**settings.model_dump())
