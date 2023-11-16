""" BaseCorpus

Must support:
1. corpus name
2. serialisation
3. clonable
"""
import uuid
from abc import ABCMeta, abstractmethod
from typing import Iterable, Hashable, Optional
import logging

import srsly

from atap_corpus.interfaces import Clonable, Serialisable, Container, Filterable
from atap_corpus._types import TCorpus, MPK_SUPPORTED

logger = logging.getLogger(__name__)


class BaseCorpus(Clonable, Serialisable, metaclass=ABCMeta):
    """ Base Corpus

    Base Corpus objects have unique IDs.
    This allows for a hidden centralised single-entry GlobalCorpora that is accessable at runtime.
    Note that the UniqueNameProvider does not have to be GlobalCorpora, as long as the names are unique.

    All Corpus types should inherit from this class.
    """

    _COL_DOC: str = 'document_'

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        from atap_corpus.registry import _Global_Corpora
        if name is None:
            name = _Global_Corpora.unique_name()
        self._name = name
        self._id = _Global_Corpora.unique_id()
        _Global_Corpora.add(corpus=self)

        self._attributes = dict()

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

    def rename(self, name: str):
        self.name = name

    @abstractmethod
    def docs(self) -> Filterable:
        raise NotImplementedError()

    @property
    def attributes(self):
        return self._attributes

    def attribute(self, key: Hashable, value: MPK_SUPPORTED):
        """ Attribute an attribute to the Corpus. These are corpus-level metadata."""
        self._attributes[key] = value

    def unattribute(self, key):
        """ Remove an attribute from the Corpus. These are corpus-level metadata.
        If not found, nothing happens.
        """
        try:
            del self._attributes[key]
        except KeyError as _:
            pass

    def __hash__(self) -> int:
        """ Do not override this or __eq__(). GlobalCorpora depends on this. """
        return hash(self.id.int)

    def __eq__(self, other):
        """ Do not override this or __hash__(). GlobalCorpora depends on this. """
        return super().__eq__(other)

    @abstractmethod
    def __len__(self) -> int:
        """ Returns the number of documents in the Corpus. """
        raise NotImplementedError()


class BaseCorpora(Container, metaclass=ABCMeta):
    """ Base Corpora

    All Corpora (corpus containers) should implement this class.
    This Base class does not impose anything but the need for the instances within a BaseCorpora to be a BaseCorpus.

    It also changes the argument names for the relevant inherited Container functions.
    """

    def __init__(self, corpus: Optional[TCorpus | Iterable[TCorpus]] = None):
        if corpus is not None:
            corpus = list(corpus)
            for c in corpus:
                if not isinstance(c, BaseCorpus):
                    raise TypeError(f"Corpora can only store Corpus objects. Got {c.__class__.__name__}.")

    @abstractmethod
    def add(self, corpus: TCorpus):
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
    def get(self, name: Hashable) -> Optional[TCorpus]:
        """ Returns the Corpus object from the Corpora. """
        pass


# dev - this is a strict class hierarchy because it's very much implementation dependent.
#   e.g. for DataFrameCorpus, it'll require reference to the dataframe which is a third party and how to
#       add metadata to it is very much their API specific.
class BaseCorpusWithMeta(BaseCorpus, metaclass=ABCMeta):
    """ BaseCorpus that also holds metadata. """

    @abstractmethod
    def metas(self) -> list[str]:
        """ Return a list of names for the metadata collections in the corpus. """
        raise NotImplementedError()

    @abstractmethod
    def add_meta(self, meta: list | tuple, name: str):
        """ Adds a metadata collection in the Corpus with name. """
        if len(meta) != len(self):
            raise ValueError(
                f"Added meta {meta} does not align with Corpus size. Expecting {len(self)} Got {len(meta)}"
            )
        ...

    @abstractmethod
    def remove_meta(self, name: str):
        """ Remove a metadata collection from the Corpus with name. """
        raise NotImplementedError()

    @abstractmethod
    def get_meta(self, name: str):
        """ Get the metadata collection for this Corpus with name. """
        if name == self._COL_DOC:
            raise KeyError(f"{name} is reserved for Corpus documents. It is never used for meta data.")
        ...
