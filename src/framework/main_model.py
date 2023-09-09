import instance
import abc
from instance_preprocessing import InstancePreprocessor


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
        self._pre_inst = InstancePreprocessor.delete_non_letters(self._inst)

    def dump(self, path: str):
        # with open(path, "w") as file:
        #     file.writelines(json.dump(self._result))
        pass
