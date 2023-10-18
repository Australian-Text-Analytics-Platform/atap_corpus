import uuid
from typing import Type, Iterable, Optional
import weakref as wref
import logging
from datetime import datetime

from atap_corpus.corpus.base import BaseCorpora, TCorpus, BaseCorpus
from atap_corpus.mixins import UniqueIDProviderMixin, UniqueNameProviderMixin
from atap_corpus.utils import format_dunder_str

logger = logging.getLogger(__name__)


class UniqueCorpora(BaseCorpora):
    """ UniqueCorpora
    UniqueCorpora is a container for BaseCorpus objects.
    BaseCorpus ensures each Corpus id is unique, UniqueCorpora uses this property to ensure uniqueness.
    """

    def __init__(self, corpus: Optional[BaseCorpus | Iterable[BaseCorpus]] = None):
        super().__init__(corpus)
        collection: dict[uuid.UUID, TCorpus] = dict()
        if corpus is not None:
            for c in corpus:
                if c.id in collection.keys():
                    logger.warning(f"Corpus ID: {c.id} is duplicated. Only one is kept for uniqueness.")
                else:
                    collection[c.id] = c
        self._collection = collection

    def add(self, corpus: TCorpus):
        """ Adds a Corpus into the Corpora. Corpus name is used as the name for get(), remove().
        If the same corpus is added again, it'll have no effect.
        """
        self._collection[corpus.id] = corpus

    def remove(self, id_: uuid.UUID | str):
        """ Remove a Corpus from the Corpora.
        If Corpus does not exist, it'll have no effect.
        """
        try:
            if isinstance(id_, str):
                id_ = uuid.UUID(id_)
            id_: uuid.UUID
            del self._collection[id_]
        except KeyError as ke:
            pass

    def items(self) -> list[TCorpus]:
        """ Returns a list of Corpus in the Corpora. Shallow copies. """
        return list(self._collection.values()).copy()

    def get(self, id_: uuid.UUID | str) -> Optional[TCorpus]:
        """ Return a reference to a Corpus with the specified name. """
        if isinstance(id_, str): id_ = uuid.UUID(id_)
        return self._collection.get(id_, None)

    def clear(self):
        """ Clears all Corpus in the Corpora. """
        self._collection = dict()

    def __len__(self) -> int:
        """ Returns the number of Corpus in the Corpora."""
        return len(self._collection)

    def __str__(self) -> str:
        return format_dunder_str(self.__class__, **{"size": len(self)})


class _GlobalCorpora(UniqueIDProviderMixin, UniqueNameProviderMixin, BaseCorpora):
    """ GlobalCorpora

    Global corpora holds weak references to all created Corpus objects in a WeakKeyDictionary.
    This allows us to:
    1. obtain a view of all the corpus that are created via a single entry point.
    2. extendable to provide runtime manipulations on all existing Corpus objects.
    3. weak reference ensures we won't have a dangling reference to a Corpus.

    This class is a Singleton.
    The WeakKeyDictionary holds the n

    This class is not designed to let you have random access to specific Corpus based on its ID or Name.
    You can however be able to iterate through all the existing Corpus via items().
    """

    _instance = None

    def __new__(cls: Type['_GlobalCorpora']) -> '_GlobalCorpora':
        if cls._instance is None:
            instance = super(_GlobalCorpora, cls).__new__(cls)
            cls._instance = instance
            instance._collection: wref.WeakKeyDictionary[wref.ReferenceType[TCorpus], dict]
            instance._collection = wref.WeakKeyDictionary()
            logger.debug("GlobalCorpora singleton created.")
        return cls._instance

    def add(self, corpus: TCorpus):
        self._collection[corpus] = dict(created=datetime.now())

    def get(self, id_: uuid.UUID | str) -> Optional[TCorpus]:
        raise NotImplementedError(f"Do not get directly from {self.__class__.__name__}.")

    def remove(self, id_: uuid.UUID | str):
        raise NotImplementedError(f"Do not remove directly from {self.__class__.__name__}.")

    def items(self) -> list[TCorpus]:
        return list(self._collection.keyrefs())

    def clear(self):
        return NotImplementedError(f"Do not clear directly from {self.__class__.__name__}")

    # UniqueIDProviderMixin
    def is_unique_id(self, id_: uuid.UUID) -> bool:
        return hash(id_.int) not in self._collection.keys()

    # UniqueNameProviderMixin
    def is_unique_name(self, name: str) -> bool:
        # this is O(n)
        for corpus in self._collection.keys():
            if name == corpus.name:
                return False
        return True

    def __len__(self) -> int:
        return len(set(self._collection.keys()))
