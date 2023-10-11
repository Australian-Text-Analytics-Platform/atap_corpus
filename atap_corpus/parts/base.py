from abc import ABCMeta, abstractmethod
from typing import TypeVar

from atap_corpus.interfaces import Clonable


class BaseFreqTable(metaclass=ABCMeta):
    pass


TFreqTable = TypeVar("TFreqTable", bound=BaseFreqTable)


class BaseDTM(metaclass=ABCMeta, Clonable):
    # todo: core interface functions need to be defined here.
    @abstractmethod
    def to_freqtable(self) -> TFreqTable:
        pass
