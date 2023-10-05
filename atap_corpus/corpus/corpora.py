from typing import Type, Iterable, Optional
import weakref as wref
from weakref import ReferenceType
import logging

from atap_corpus.corpus.base import BaseCorpora, TBaseCorpus, BaseCorpus
from atap_corpus.corpus.mixins import UniqueNameProviderMixin
from atap_corpus.utils import format_dunder_str

logger = logging.getLogger(__name__)


class UniqueCorpora(BaseCorpora):
    """ UniqueCorpora
    UniqueCorpora is a container for BaseCorpus objects.
    BaseCorpus ensures each Corpus name is unique, UniqueCorpora uses this property to ensure uniqueness.
    """

    def __init__(self, corpus: Optional[BaseCorpus | Iterable[BaseCorpus]] = None):
        super().__init__(corpus)
        collection = dict()
        if corpus is not None:
            for c in corpus:
                if c.name in collection.keys():
                    raise ValueError(f"Corpus already exist. Unable to retain uniqueness in {self.__class__.__name__}.")
                else:
                    collection[c.name] = c
        self._collection = collection
        logger.debug("UniqueCorpora init is called.")

    def add(self, corpus: TBaseCorpus):
        """ Adds a Corpus into the Corpora. Corpus name is used as the name for get(), remove().
        If the same corpus is added again, it'll have no effect.
        """
        self._collection[corpus.name] = corpus

    def remove(self, name: str):
        """ Remove a Corpus from the Corpora.
        If Corpus does not exist, it'll have no effect.
        """
        try:
            del self._collection[name]
        except KeyError as ke:
            pass

    def items(self) -> list[TBaseCorpus]:
        """ Returns a list of Corpus in the Corpora. Shallow copies. """
        return list(self._collection.values()).copy()

    def get(self, name: str) -> TBaseCorpus:
        """ Return a reference to a Corpus with the specified name. """
        return self._collection.get(name, None)

    def clear(self):
        """ Clears all Corpus in the Corpora. """
        self._collection = dict()

    def __len__(self) -> int:
        """ Returns the number of Corpus in the Corpora."""
        return len(self._collection)

    def __str__(self) -> str:
        return format_dunder_str(self.__class__, **{"size": len(self)})


class _GlobalCorpora(UniqueCorpora, UniqueNameProviderMixin):
    """ GlobalCorpora

    Global corpora holds weak references to all created Corpus objects.
    This allows us to:
    1. obtain a view of all the corpus that are created via a single entry point.
    2. extendable to provide runtime manipulations on created Corpus objects.
    3. weak reference ensures the reference to a Corpus is dropped when there isn't anymore.

    This class is a Singleton.
    All references held but this singleton object is a weak reference.
    """
    _instance = None

    def __new__(cls: Type['_GlobalCorpora']) -> '_GlobalCorpora':
        if cls._instance is None:
            instance = super().__new__(cls)
            cls._instance = instance
            collection: dict[str, ReferenceType[TBaseCorpus]] = dict()
            cls._instance._collection = collection
            logger.debug("GlobalCorpora singleton created.")
        return cls._instance

    def __init__(self, *args, **kwargs):
        if self._instance is not None: return  # otherwise super().__init__() is called again which empties collection.
        super().__init__(*args, **kwargs)

    def add(self, corpus: TBaseCorpus):
        if not self.is_unique_name(corpus.name):
            raise RuntimeError(f"{corpus.name} is not unique. GlobalCorpora will be invalid.")
        corpus: wref.ReferenceType[TBaseCorpus] = wref.ref(corpus)
        self._collection[corpus().name] = corpus
        logger.debug(f"Added Corpus: id={id(corpus())}. GlobalCorpora size: {len(self)}.")
        # dev - here Corpus haven't been fully instantiated yet.

    def remove(self, name: str):
        super().remove(name)
        logger.debug(f"Removed Corpus with name: {name}. GlobalCorpora size: {len(self)}")

    def get(self, name: str) -> Optional[TBaseCorpus]:
        return self._collection.get(name)

    def items(self) -> list[TBaseCorpus]:
        return list([c() for c in self._collection.values()])

    def is_unique_name(self, name: str) -> bool:
        return name not in self._collection.keys()
