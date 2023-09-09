import instance
import abc
from instance_preprocessing import InstancePreprocessor
import classificator
import json


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

    def fit(self, inst: instance.Instance):
        self._inst = inst

    def train(self):
        # preprocessing
        self._pre_inst = InstancePreprocessor.replace_abbreviations(self._inst)

        # getting sentiments
        temp_inst = InstancePreprocessor.token_lemmatization_natasha(self._inst)
        temp_inst = InstancePreprocessor.replace_anglicisms(temp_inst)
        sentiments = InstancePreprocessor.get_sentiments(temp_inst)

        # more preprocessing
        self._pre_inst = InstancePreprocessor.replace_anglicisms(self._inst)
        self._pre_inst = InstancePreprocessor.delete_question_mark(self._inst)

        self._pre_inst.sentiments = sentiments

        # getting clusters
        clustor = classificator.Clusterizator()
        clustor.fit(self._pre_inst)
        clustor.train()
        self._pre_inst.clusters = clustor.get_clusters()

    def dump(self, path: str):
        j_dict = {"question": self._pre_inst.question, "id": self._pre_inst.id_, "answers": []}
        for i in range(len(self._pre_inst)):
            a_dict = {
                "answer": self._pre_inst.answers[i],
                "count": self._pre_inst.counts[i],
                "cluster": self._pre_inst.clusters[i],
                "sentiment": self._pre_inst.sentiments[i]
            }
            j_dict["answers"].append(a_dict)

        with open(path, "w") as file:
            json.dump(j_dict, file, separators=(",\n", ": "), ensure_ascii=False)
