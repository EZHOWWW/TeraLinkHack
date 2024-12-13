import string
import typing as tp

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from torch import Tensor, no_grad
from transformers import AutoModel, AutoTokenizer

nltk.download("punkt")
nltk.download("stopwords")

Document = str


class Clusterizer:
    def __init__(
        self, model_name: str = "bert-base-multilingual-cased", num_clusters: int = 3
    ):
        self.num_clusters = num_clusters
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.clustering = AgglomerativeClustering(n_clusters=self.num_clusters)

        self.documents: tp.List[Document] = []
        self.embeddings = None

        self.stop_words = set(stopwords.words("russian"))
        self.punctuation = set(string.punctuation)
        self.stemmer = SnowballStemmer("russian")

    def preprocess_documents(self, documents: tp.List[Document]) -> tp.List[Document]:
        preprocessed = []
        for doc in documents:
            tokens = word_tokenize(
                doc.lower()
                .replace("«", "")
                .replace("»", "")
                .replace('"', "")
                .replace("  ", " ")
                .replace("-", "")
            )
            filtered_tokens = [
                self.stemmer.stem(token)
                for token in tokens
                if token not in self.stop_words and token not in self.punctuation
            ]
            preprocessed.append(" ".join(filtered_tokens))
        return preprocessed

    def embed_documents(self, documents: tp.List[Document]) -> Tensor:
        with no_grad():
            return self.model(
                **self.tokenizer(
                    documents, padding=True, truncation=True, return_tensors="pt"
                )
            ).last_hidden_state.mean(dim=1)

    def fit(self, documents: tp.List[Document]):
        documents = self.preprocess_documents(documents)
        self.documents += documents
        self.embeddings = self.embed_documents(self.documents).cpu().numpy()
        self.clustering.fit(self.embeddings)

    def predict(self, documents: tp.List[Document]) -> tp.List[int]:
        documents = self.preprocess_documents(documents)
        return self.clustering.fit_predict(
            self.embed_documents(documents).cpu().numpy()
        ).tolist()

    def get_docs_in_cluster(self, cluster: int) -> tp.List[Document]:
        return [
            self.documents[i]
            for i in range(len(self.documents))
            if self.clustering.labels_[i] == cluster
        ]

    def evaluate_clustering(self):
        return {
            # меньше - лучше
            "Silhouette Score": silhouette_score(
                self.embeddings, self.clustering.labels_
            ),
            # больше - лучше
            "Calinski-Harabasz Index": calinski_harabasz_score(
                self.embeddings, self.clustering.labels_
            ),
            # меньше - лучше
            "Davies-Bouldin Index": davies_bouldin_score(
                self.embeddings, self.clustering.labels_
            ),
        }


otsosat_documents = [
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


if __name__ == "__main__":
    from pprint import pprint
    import pandas as pd

    # https://www.kaggle.com/datasets/anzerone/clickbait-titles-ru
    newsdataset = pd.read_csv("runews.csv", sep=";")["titles"].dropna().tolist()

    def find_optimal_clusters(embeddings, max_clusters=10):
        best_score = -1
        best_num_clusters = 2
        for num_clusters in range(3, max_clusters):
            clustering = AgglomerativeClustering(n_clusters=num_clusters)
            clustering.fit(embeddings)
            score = silhouette_score(embeddings, clustering.labels_)
            if score > best_score:
                best_score = score
                best_num_clusters = num_clusters

        return best_num_clusters

    cl = Clusterizer(num_clusters=3)
    cl.fit(newsdataset)

    cl.num_clusters = find_optimal_clusters(cl.embeddings, max_clusters=40)
    print(f"Optimal number of clusters: {cl.num_clusters}")
    cl.clustering = AgglomerativeClustering(n_clusters=cl.num_clusters)
    cl.clustering.fit(cl.embeddings)

    for id in range(cl.num_clusters):
        print(f"Cluster {id}:", end=" ")
        pprint(cl.get_docs_in_cluster(id)[:10])

    print("\nMetrics:")
    for metric, value in cl.evaluate_clustering().items():
        print(f"{metric}: {value:.4f}")
