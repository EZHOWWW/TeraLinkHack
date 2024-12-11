from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import numpy as np
import typing as tp

NGRAM_RANGE = (1, 2)
MAX_FEATURES = 128


class clastorizer:
    def __init__(self):
        self.tf_idf = TfidfVectorizer(
            ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)
        self.pipeline = Pipeline(
            [self.tf_idf])
        self.data: tp.Set[str] = set() 

        self.num_classes = 4
        self.kmeans = KMeans(n_clusters=self.num_classes, random_state=52)
        self.df_idf_mat = None

    def get_categories(self, data: tp.List[str] | tp.Set[str]) -> tp.List[tp.Tuple[float, str]]:
        '''
        input: get list of documents(List(str))
        return: return list of scores for each word for each document
        TODO: сейчас плохо работает: каждый раз сохроняем все полученные документы вместо хранения IDF. 
            мейби нужно обучать на какой то обучающей выборке а потом уже использовать ?
        '''
        self.data |= set(data)
        self.tf_idf_mat = self.tf_idf.fit_transform(self.data)
        # нормолизация матрицы мб долгая операция (спросить у попкова)
        features = self.tf_idf.get_feature_names_out()
        scores = self.tf_idf_mat.toarray()

        return [self._get_scorc_feature(features, scores[-i-1]) for i in range(len(data)-1, -1, -1)]

    def clastorize(self, data):
        cat = self.get_categories(data)
        print(self.tf_idf.get_feature_names_out())
        self.kmeans.fit(self.tf_idf_mat)
        for i in range(self.num_classes):
            cl = np.where(self.kmeans.labels_ == i)[0]
            print(cl)
            print(f"Кластер {i + 1}:")
            for idx in cl:
                print(data[idx])

    def fit(self, data):
        self.tf_idf.fit(data)

    def _get_scorc_feature(self, features, scores):
        return sorted(zip((scores.tolist()), features), reverse=True)


if __name__ == "__main__":
    pred_data = [
        "Машинное обучение - это интересная область.",
        "Обучение с учителем - ключевой аспект машинного обучения.",
        "Область NLP также связана с машинным обучением.",
        "Ебля с tf-idf тоже связана с машинным обучением агентов."
    ]
    cl = clastorizer()
    # print(cl.get_categories(pred_data))
    new_data = ['трахать и дрочить очень хорошо',
                'дрочить плохо трахать неплохо',
                'Трахать очень плохо, а дрочить хорошо )',
                'И трахать и дрочить очень сильно плохо (']
    # n_cl = cl.get_categories(new_data)
    cl.clastorize(pred_data + new_data)
