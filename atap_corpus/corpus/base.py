""" BaseCorpus

Must support:
1. corpus name
2. serialisation
3. clonable
"""

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Iterable, Hashable, TypeVar, Optional
import logging

from atap_corpus.interfaces import Clonable, Serialisable, Container
from atap_corpus.types import PathLike
import sys

logger = logging.getLogger(__name__)


class BaseCorpus(Clonable, Serialisable, metaclass=ABCMeta):
    """ Base Corpus

    Base Corpus objects have unique names.
    This allows for a hidden centralised single-entry GlobalCorpora that is accessable at runtime.
    Note that the UniqueNameProvider does not have to be GlobalCorpora, as long as the names are unique.

    All Corpus types should inherit from this class.
    """

    def __init__(self, name: Optional[str] = None):
        if name is None:
            from atap_corpus.registry import _Unique_Name_Provider
            name = _Unique_Name_Provider.unique_name()
        self._name = name
        from atap_corpus.registry import _Global_Corpora
        _Global_Corpora.add(corpus=self)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError(f"name must be a str. Got {name.__class__.__name__}.")
        from atap_corpus.registry import _Unique_Name_Provider
        if not _Unique_Name_Provider.is_unique_name(name):
            raise TypeError(f"{name} is not unique. Provide a unique name. See GlobalCorpora.")
        self._name = name

    def serialise(self, path: PathLike) -> PathLike:
        return Path(path).with_suffix(".corp")

    def __hash__(self) -> int:
        return hash(self.name)

    def __del__(self):
        # no super call required here - all abstract classes.
        from atap_corpus.registry import _Global_Corpora
        _Global_Corpora.remove(self.name)
        logger.debug(f"Corpus collected {id(self)}.")


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
