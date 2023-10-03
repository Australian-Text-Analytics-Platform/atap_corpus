""" BaseCorpus

Must support:
1. corpus name
2. serialisation
3. clonable
"""

from abc import ABCMeta
from pathlib import Path

from atap_corpus.interfaces import Clonable, Serialisable
from atap_corpus.types import TPathLike


class BaseCorpus(metaclass=ABCMeta, Clonable, Serialisable):
    """ Base Corpus

    All Corpus types should inherit from this class.
    """

    def serialise(self, path: TPathLike) -> TPathLike:
        return Path(path).with_suffix(".corp")
