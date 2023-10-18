from typing import *

from atap_corpus.slicer.op import BaseOperation
from atap_corpus.interfaces import Container


# dev - These are placeholder no implementations yet.
class BaseOperations(Container):

    def __init__(self, ops: Optional[list[BaseOperation]] = None):
        super(BaseOperation).__init__()
        self._ops = list() if not ops else ops

    def add(self, op: BaseOperation):
        pass

    def remove(self, op: Union[int, BaseOperation]):
        pass

    def items(self) -> list['BaseOperation']:
        pass

    def clear(self):
        pass

    def get(self, idx: int):
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass
