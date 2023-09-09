import typing as tp
from dataclasses import dataclass


@dataclass
class Instance:
    question: str
    id_: int
    answers: tp.List[str]
    counts: tp.List[int]
    sentiments: tp.List[str] = None
    clusters: tp.List[str] = None
    corrected: tp.List[str] = None

    def __len__(self):
        return len(self.answers)