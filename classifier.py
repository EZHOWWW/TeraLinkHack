from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_20newsgroups
import typing as tp


class Classifier:
    def __init__(self):
        self.tf_idf = TfidfVectorizer()
        self.data: tp.List[str] = []

    def get_classes(self, data: tp.List[str]) -> tp.List[tp.Tuple[float, str]]:
        '''
        input: get list of documents(List(str))
        return: return list of scores for each word for each document
        TODO: сейчас плохо работает: каждый раз сохроняем все полученные документы вместо хранения IDF. 
            мейби нужно обучать на какой то обучающей выборке а потом уже использовать ?
        '''
        self.data += data
        tf_idf_mat = self.tf_idf.fit_transform(self.data)
        features = self.tf_idf.get_feature_names_out()
        scores = tf_idf_mat.toarray()

        return [self.get_scorc_feature(features, scores[-i-1]) for i in range(len(data))][::-1]

    def get_scorc_feature(self, features, scores):
        return sorted(zip((scores.tolist()), features), reverse=True)


if __name__ == "__main__":
    pred_data = [
        "Машинное обучение - это интересная область.",
        "Обучение с учителем - ключевой аспект машинного обучения.",
        "Область NLP также связана с машинным обучением.",
        "Ебля с tf-idf тоже связана с машинным обучением агентов."
    ]
    cl = Classifier()
    print(cl.get_classes(pred_data))
    new_data = ['трахать и дрочить очень хорошо',
                'дрочить плохо трахать неплохо',
                'Трахать очень плохо, а дрочить хорошо )',
                'И трахать и дрочить очень сильно плохо (']
    n_cl = cl.get_classes(new_data)
    for i in n_cl:
        print('\n', i, '\n')
