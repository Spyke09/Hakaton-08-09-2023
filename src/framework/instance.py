from dataclasses import dataclass
import typing as tp


@dataclass
class Instance:
    question: str
    id: int
    answers: tp.List[str]
    counts: tp.List[int]
    sentiments: tp.List[str] = None
