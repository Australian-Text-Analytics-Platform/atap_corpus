""" BaseCorpus

Must support:
1. corpus name
2. serialisation
3. clonable
"""
import uuid
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Iterable, Hashable, TypeVar, Optional
import logging

from atap_corpus.interfaces import Clonable, Serialisable, Container
from atap_corpus._types import PathLike

logger = logging.getLogger(__name__)


class BaseCorpus(Clonable, Serialisable, metaclass=ABCMeta):
    """ Base Corpus

    Base Corpus objects have unique IDs.
    This allows for a hidden centralised single-entry GlobalCorpora that is accessable at runtime.
    Note that the UniqueNameProvider does not have to be GlobalCorpora, as long as the names are unique.

    All Corpus types should inherit from this class.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        from atap_corpus.registry import _Global_Corpora
        if name is None:
            name = _Global_Corpora.unique_name()
        self._name = name
        self._id = _Global_Corpora.unique_id()
        _Global_Corpora.add(corpus=self)

    @property
    def id(self) -> uuid.UUID:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        from atap_corpus.registry import _Global_Corpora
        if not _Global_Corpora.is_unique_name(name):
            raise ValueError(f"{name} already exists.")
        self._name = name

    def __hash__(self) -> int:
        return hash(self.id.int)

    @abstractmethod
    def __len__(self) -> int:
        """ Returns the number of documents in the Corpus. """
        raise NotImplementedError()


TBaseCorpus = TypeVar("TBaseCorpus", bound=BaseCorpus)


class BaseCorpora(Container, metaclass=ABCMeta):
    """ Base Corpora

    All Corpora (corpus containers) should implement this class.
    This Base class does not impose anything but the need for the instances within a BaseCorpora to be a BaseCorpus.

    It also changes the argument names for the relevant inherited Container functions.
    """

    def __init__(self, corpus: Optional[TBaseCorpus | Iterable[TBaseCorpus]] = None):
        if corpus is not None:
            corpus = list(corpus)
            for c in corpus:
                if not isinstance(c, BaseCorpus):
                    raise TypeError(f"Corpora can only store Corpus objects. Got {c.__class__.__name__}.")

    @abstractmethod
    def add(self, corpus: TBaseCorpus):
        """ Adds a corpus to the corpora.
        :arg corpus - a subclass of BaseCorpus. (renamed from 'obj' in Container abc)
        """
        pass

    @abstractmethod
    def remove(self, name: Hashable):
        """ Removes a corpus from the corpora.
        :arg name - the name of the Corpus. (renamed from 'key' in Container abc)
        """
        pass

    @abstractmethod
    def get(self, name: Hashable) -> Optional[TBaseCorpus]:
        """ Returns the Corpus object from the Corpora. """
        pass


TBaseCorpora = TypeVar("TBaseCorpora", bound=BaseCorpus)
