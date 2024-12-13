import string
import typing as tp

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from document import Document

NGRAM_RANGE = (1, 1)
MAX_FEATURES = 128
NUM_CLASSTERS = 3
CLASSES = ['Отчеты', 'Договоры', 'Акты', 'Приказы']


class Clastorizer:
    def __init__(self, ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES, num_classters=NUM_CLASSTERS, classes=CLASSES, language="russian"):
        self.language = language
        self.tf_idf = TfidfVectorizer(
            ngram_range=ngram_range, max_features=max_features
        )
        self.data: tp.List[Document] = []

        self.num_classters = 3
        self.kmeans = KMeans(n_clusters=self.num_classters, random_state=52)

        self._init_nltk()

        self.classes = classes
        self.prep_classes = [self.stemmer.stem(i.lower())
                             for i in self.classes]

    def _init_nltk(self):
        nltk.download("stopwords")
        nltk.download("punkt")
        nltk.download("punkt_tab")
        self.stop_words = set(stopwords.words(self.language))
        self.punctuation = set(string.punctuation)
        self.stemmer = SnowballStemmer(self.language)

    def preprocessing(self, data: tp.List[Document]) -> tp.List[Document]:
        """
        preprocessing data
        удаление союзов предлогов и т.п, приведение к нижнему регистру
        """
        for i, v in enumerate(data):
            words = word_tokenize(
                v.text.lower()
            )  # Привести к нижнему регистру и токенизировать
            filtered_words = [
                word
                for word in words
                if word not in self.stop_words and word not in self.punctuation
            ]
            stemmed_words = [
                self.stemmer.stem(word)
                for word in word_tokenize(" ".join(filtered_words))
            ]
            res = stemmed_words
            data[i].preprocessed_text = " ".join(res)
        return data

    def fit_etf_idf(self, dataset=None):
        """
        fit tf_idf with dataset.
        обучаем tf_idf и далее используем
        """
        self.tf_idf.fit(dataset)

    def get_categories(
        self,
        data: tp.List[Document] | tp.Set[Document],
        refit=False,
        add_to_data=False,
        preprocess=True,
    ) -> tp.Tuple[tp.List[Document], tp.List[float]]:
        """
        input: get list of documents(List(Document))
            refit - Если хотим переобучить модель на self.data
            add_to_data - если хотим добавлять данные;
        return:
            features - names of features,
            scores - value of each feature;
        """
        texts = []
        if preprocess:
            self.preprocessing(data)
            texts = [i.preprocessed_text for i in data]
        else:
            texts = [i.text for i in data]
        if add_to_data:
            self.data += data
            data = self.data

        if refit:
            tf_idf_mat = self.tf_idf.fit_transform(texts)
        else:
            tf_idf_mat = self.tf_idf.transform(texts)

        features = self.tf_idf.get_feature_names_out()
        scores = tf_idf_mat.toarray()

        return features, scores

    def clastorize(
        self, data: tp.List[Document] | tp.Set[Document], refit=False, add_to_data=False
    ) -> tp.List[int]:
        """
        input: see get_categories
        output: для каждого документа номер класса к которому он относится
        """
        features, scores = self.get_categories(data, refit, add_to_data)
        res = self.kmeans.fit_transform(scores)
        return [np.argmax(r) for r in res]

    def get_docs_in_classter(self, classter: int) -> tp.List[Document]:
        return [
            self.data[j] for i in np.where(self.kmeans.labels_ == classter) for j in i
        ]

    def get_word_of_classter(self, classter: int) -> str:
        vec = TfidfVectorizer()
        mat = vec.fit_transform(
            [i.text for i in self.get_docs_in_classter(classter)]
        ).toarray()
        ind = mat.argmax() % len(vec.get_feature_names_out())

        return vec.get_feature_names_out()[ind]

    def _zip_scor_feat(self, scores, features):
        res = []
        for i in scores:
            res.append(sorted(zip(i.tolist(), features), reverse=True))
        return res

    def classify(self, data: tp.List[Document], refit=False,
                 add_to_data=False,
                 preprocess=True,
                 ) -> tp.List[str]:
        '''Классифицируем через tf-idf, просто ищем наибольшее значение среди self.classes'''
        features, scores = self.get_categories(
            data, refit=refit, add_to_data=add_to_data, preprocess=preprocess)
        res = []
        max_val = 0
        clas = ''
        for i in self._zip_scor_feat(scores, features):
            for s, f in i:
                if s >= max_val and f in self.prep_classes:
                    max_val, clas = s, f
            res.append(clas)
        return res


def to_docs(data):
    res = []
    for i in data:
        res.append(Document(text=i))
    return res


dada = to_docs(
    [
        "финансы деньги ляля.",
        "деньги мало бебе, хер.",
        "финансы очень плохо но похуй.",
        "окей и пох финансы хуй деньги.",
        "Деньгами финансам не поможешь",
        "Отчет не сделан да и хер с ним",
        "Надо сделать отчет по алгосам",
        "Отчет сука мне нужен отчет!",
        "Отчет по финансы ",
        "Хакатон делаем хакатон",
        "выйграем деньги на хакатон",
        "Отчет по хакатон хакатон",
        "хакатоны сасать америка",
    ]
)

dat2 = to_docs([
    'Отчет: лфывалдфываждофывждарфыждвпр фждыоварфыджов фдыпрждофырав фждыова рждфыоап дофыардр',
    'Отчет: фдыова дофыарв дрфывджа фыджоар вфоырва д',
    'договоры: фываолф джфывола ждлфыов афдылоа ',
    'договоры фыждвао джфылова ждфыова фыджвоа договры джфылоав',
    'акт 1 фджылав ждфылвоа фыжвд аложфдылво афдылвоа акт 2',
])

if __name__ == "__main__":
    cl = Clastorizer()
    print(cl.classify(dat2, True, True))
    '''
    z = cl._zip_scor_feat(*cl.get_categories(dada, True, True))
    claDocuments = cl.clastorize(dada, True)
    for i in range(cl.num_classters):
        print(cl.get_docs_in_classter(i), "\n")
        print(cl.get_word_of_classter(i), "\n")
    '''

    # cl.clastorize(pred_data + new_data)
