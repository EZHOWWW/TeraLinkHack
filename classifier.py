import string
import typing as tp

#import nltk
import numpy as np
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

NGRAM_RANGE = (1, 1)
MAX_FEATURES = 128


Document = str 

class clastorizer:
    def __init__(self):
        self.tf_idf = TfidfVectorizer(
            ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)
        self.data: tp.List[Document] = [] 

        self.num_classes = 3
        self.kmeans = KMeans(n_clusters=self.num_classes, random_state=52)


    def _init_nltk(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stop_words = set(stopwords.words('russian'))
        self.punctuation = set(string.punctuation)

    def preprocessing(self, data:tp.List[Document]):
        pass


    def fit_tf_idf(self, dataset = None):
        '''
        fit tf_idf with dataset.
        обучаем tf_idf и далее используем
        '''
        self.tf_idf.fit(dataset)
        

   

    def get_categories(self, data: tp.List[Document] | tp.Set[Document], refit = False, add_to_data = False) -> tp.Tuple[tp.List[Document], tp.List[float]]:
        '''
        input: get list of documents(List(Document))
            refit - Если хотим переобучить модель на self.data
            add_to_data - если хотим добавлять данные;
        return: 
            features - names of features,
            scores - value of each feature;
        '''
        if add_to_data:
            self.data += data
            data = self.data

        if refit:
            tf_idf_mat = self.tf_idf.fit_transform(data)
        else:
            tf_idf_mat = self.tf_idf.transform(data)


        features = self.tf_idf.get_feature_names_out()
        scores = tf_idf_mat.toarray()

        return features, scores


    def clastorize(self, data: tp.List[Document] | tp.Set[Document], refit = False, add_to_data = False) -> tp.List[int]:
        '''
        input: see get_categories
        output: для каждого документа номер класса к которому он относится 
        '''
        features, scores = self.get_categories(data, refit, add_to_data)
        res = self.kmeans.fit_transform(scores)
        print(res)
        return [np.argmax(r) for r in res]
    
    def get_docs_in_classter(self, classter: int) -> tp.List[Document]:
        return [self.data[j] for i in np.where(self.kmeans.labels_ == classter) for j in i]

    def _zip_scor_feat(self, features, scores):
        res = []
        for i in scores:
            res.append(sorted(zip(i.tolist(), features), reverse = True))
        return res



dada = ['финансы деньги ляля.',
'деньги мало бебе, хер.',
'финансы очень плохо но похуй.',
'окей и пох финансы хуй деньи.',

"Отчет не сделан да и хер с ним",
"Надо сделать отчет по алгосам",
"Отчет сука мне нужен отчет!",
"Отчет по финансы ",

"Хакатон делаем хакатон",
"выйграем деньги на хакатон",
"Отчет по хакатон хакатон",
"Хакатон: сасать америка"]
if __name__ == "__main__":
    cl = clastorizer()
    z = cl._zip_scor_feat(*cl.get_categories(dada, True, True))
    for i in z:
        print(i, '\n')
    claDocuments = cl.clastorize(dada, True )
    for i in range(cl.num_classes):
        print(cl.get_docs_in_classter(i), '\n')
    
    #cl.clastorize(pred_data + new_data)
