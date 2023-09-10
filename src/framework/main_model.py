import abc
import json
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import src.framework.classificator as classificator
import src.framework.instance as instance
from src.framework.instance_preprocessing import Preprocessor


class AbstractModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, inst: instance.Instance):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def dump(self, path: str):
        raise NotImplementedError


class SimpleModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self._inst = None
        self._pre_inst = None
        self._result = None
        self._clusters = None

    def fit(self, inst: instance.Instance):
        self._inst = inst

    def train(self):
        # preprocessing
        prep = Preprocessor()
        self._pre_inst = prep.composition([
            prep.replace_abbreviations,
            prep.token_lemmatization_natasha,
            prep.replace_anglicisms,
            prep.delete_question_mark
        ], self._inst)

        # getting sentiments
        sentiments = prep.get_sentiments(self._pre_inst)

        # more preprocessing

        # getting clusters
        clustor = classificator.Clusterizator()
        clustor.fit(self._pre_inst)
        clustor.train()
        clusters = clustor.get_clusters()
        self._clusters = clusters

        self._inst.clusters = clusters
        self._inst.sentiments = sentiments
        self._inst.corrected = self._pre_inst.answers

    def dump(self, path: str):
        j_dict = {"question": self._pre_inst.question, "id": self._pre_inst.id_, "answers": []}
        for i in range(len(self._pre_inst)):
            a_dict = {
                "answer": self._inst.answers[i],
                "count": self._inst.counts[i],
                "cluster": self._inst.clusters[i],
                "sentiment": self._inst.sentiments[i],
                "corrected": self._inst.corrected[i]
            }
            j_dict["answers"].append(a_dict)

        with open(f"{path}/{self._inst.id_}_labeled.json", "w") as file:
            json.dump(j_dict, file, separators=(",\n", ": "), ensure_ascii=False)

    @property
    def clusters(self):
        return self._clusters

    def get_stats_dataframe(self):
        df = pd.DataFrame({
            "question": [self._inst.question for _ in range(len(self._inst))],
            "id": [self._inst.id_ for _ in range(len(self._inst))],
            "answer": self._inst.answers,
            "count": self._inst.counts,
            "cluster": self.clusters,
            "sentiment": self._inst.sentiments,
            "corrected": self._inst.corrected
        })
        return df

    def save_to_excel(self, path: str):
        df = self.get_stats_dataframe()

        freq1 = Counter(self.clusters)

        df1 = pd.DataFrame(freq1, index=[0]).T.reset_index()
        df1.columns = ["cluster_name", "frequency"]

        writer = pd.ExcelWriter(f"{path}/semantic_clusters_stats_{self._inst.id_}.xlsx", engine="xlsxwriter")
        df.to_excel(writer, sheet_name="labeled_json")
        df1.to_excel(writer, sheet_name="clusters_frequency")

        writer.close()

    def draw_word_cloud(self, path):
        word_cloud = WordCloud(
            width=3000,
            height=2000,
            random_state=100,
            background_color="white",
            colormap="cool",
            collocations=False,
        ).generate_from_frequencies(Counter(self.clusters))

        plt.imshow(word_cloud)
        plt.axis("off")

        word_cloud.to_file(f"{path}/semantic_clusters_freq_in_{self._inst.id_}_json.pdf")
