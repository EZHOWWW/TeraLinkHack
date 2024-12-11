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
    def fit_tf_idf(self, dataset = None):
        '''
        fit tf_idf with dataset.
        обучаем tf_idf и далее используем
        '''
        self.tf_idf.fit(dataset)
        

   

    def get_categories(self, data: tp.List[str] | tp.Set[str], refit = False, add_to_data = False) -> tp.Tuple[tp.List[str], tp.List[float]]:
        '''
        input: get list of documents(List(str))
            refit - Если хотим переобучить модель на self.data
            add_to_data - если хотим добавлять данные;
        return: 
            features - names of features,
            scores - value of each feature;
        '''
        if add_to_data:
            self.data |= set(data)
            data = self.data

        if refit:
            tf_idf_mat = self.tf_idf.fit_transform(data)
        else:
            tf_idf_mat = self.tf_idf.transform(data)


        features = self.tf_idf.get_feature_names_out()
        scores = tf_idf_mat.toarray()

        return features, scores


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


if __name__ == "__main__":
    pred_data = [
        "Машинное обучение - это интересная область.",
        "Обучение с учителем - ключевой аспект машинного обучения.",
        "Область NLP также связана с машинным обучением.",
        "Ебля с tf-idf тоже связана с машинным обучением агентов."
    ]
    cl = clastorizer()
    print(cl.get_categories(pred_data, True, True))
    new_data = ['трахать и дрочить очень хорошо',
                'дрочить плохо трахать неплохо',
                'Трахать очень плохо, а дрочить хорошо )',
                'И трахать и дрочить очень сильно плохо (']
    print(cl.get_categories(pred_data, True, True))
    #cl.clastorize(pred_data + new_data)
