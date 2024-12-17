__all__ = (
    'Model',
    'OCR',
    'OCR_SETTINGS',
    'Autocorrect',
    'AUTOCORRECT_SETTINGS',
    'Classifier',
    'CLASSIFIER_SETTINGS',
    'Summarizer',
    'SUMMARIZER_SETTINGS',
)

from .model import Model
from .ocr import OCR, settings as OCR_SETTINGS
from .autocorrect import Autocorrect, settings as AUTOCORRECT_SETTINGS
from .classifier import Classifier, settings as CLASSIFIER_SETTINGS
from .summarizer import Summarizer, settings as SUMMARIZER_SETTINGS
