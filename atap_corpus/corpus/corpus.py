import logging
from typing import Optional, TypeVar

import numpy as np
import pandas as pd

from atap_corpus.corpus.base import BaseCorpus
from atap_corpus.corpus.utils import generate_name, ensure_docs
from atap_corpus.types import PathLike, Docs

logger = logging.getLogger(__name__)

TCorpus = TypeVar("TCorpus", bound="Corpus")


class Corpus(BaseCorpus):
    """ Corpus
    This class abstractly represents a corpus which is a collection of documents.
    Each document is also described by their metadata and is used for functions such as slicing.

    An important component of the Corpus is that it also holds the document-term matrix which you can access through
    the accessor `.dtm`. See class DTM. The dtm is lazily loaded and is always computed for the root corpus.
    (read further for a more detailed explanation.)

    A main design feature of the corpus is to allow for easy slicing and dicing based on the associated metadata,
    text in document. See class CorpusSlicer. After each slicing operation, new but sliced Corpus object is
    returned exposing the same descriptive functions (e.g. summary()) you may wish to call again.

    Internally, documents are stored as rows of string in a dataframe. Metadata are stored in the meta registry.
    Slicing is equivalent to creating a `cloned()` corpus and is really passing a boolean mask to the dataframe and
    the associated metadata series. When sliced, corpus objects are created with a reference to its parent corpus.
    This is mainly for performance reasons, so that the expensive DTM computed may be reused and a shared vocabulary
    is kept for easier analysis of different sliced sub-corpus. You may choose the corpus to be `detached()` from this
    behaviour, and the corpus will act as the root, forget its lineage and a new dtm will need to be rebuilt.
    """
    _COL_DOC: str = 'document'

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, col_doc: str = _COL_DOC, name: str = None) -> TCorpus:
        if col_doc not in df.columns:
            raise ValueError(f"Column {col_doc} not found. You must set the col_doc argument.\n"
                             f"Available columns: {df.columns}")
        corpus = Corpus(df[col_doc], name=name)
        return corpus

    def to_dataframe(self):
        """ Export corpus as a dataframe. """
        return self._df.copy().reset_index(drop=True)

    def serialise(self, path: PathLike) -> PathLike:
        path = super().serialise(path)
        raise NotImplementedError()

    @classmethod
    def deserialise(cls, path: PathLike) -> TCorpus:
        raise NotImplementedError()

    def __init__(self, text: pd.Series, name: str = None):
        self._name = None
        self.name = name if name else generate_name()

        self._df: pd.DataFrame = pd.DataFrame(ensure_docs(text), columns=[self._COL_DOC])
        # ensure initiated object is well constructed.
        assert len(list(filter(lambda x: x == self._COL_DOC, self._df.columns))) <= 1, \
            f"More than 1 {self._COL_DOC} column in dataframe."

        self._parent: Optional[Corpus] = None

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name):
        global _ALL_CORPUS_NAMES
        while name in _ALL_CORPUS_NAMES:
            new_name = name + '_'
            logger.info(f"{name} already exists. It renamed to {new_name}")
            name = new_name
        _ALL_CORPUS_NAMES.add(name)

        if self.name is not None:
            try:
                _ALL_CORPUS_NAMES.remove(self.name)
            except KeyError:
                logger.debug(f'Failed to remove {self.name} from global corpus name cache.')
        self._name = name

    def rename(self, name: str):
        self.name = name

    @property
    def parent(self) -> TCorpus:
        return self._parent

    @property
    def is_root(self) -> bool:
        return self._parent is None

    def find_root(self) -> TCorpus:
        """ Find and return the root corpus. """
        if self.is_root: return self
        parent = self._parent
        while not parent.is_root:
            parent = parent._parent
        return parent

    # statistics
    @property
    def num_terms(self) -> int:
        return self.dtm.total

    @property
    def vocab(self) -> set[str]:
        return set(self.dtm.vocab(nonzero=True))

    def docs(self) -> Docs:
        return self._df.loc[:, self._COL_DOC]

    def summary(self) -> pd.DataFrame:
        """ Basic summary statistics of the corpus. """
        describe_cols_to_drop = ['count', 'std', '25%', '50%', '75%']
        docs_info = pd.Series(self.dtm.docs_size_vector).describe().drop(describe_cols_to_drop).astype(
            int)  # Show only integer numbers.

        mapper = {row_idx: f"{row_idx} Words per Document" for row_idx in docs_info.index}
        docs_info.rename(index=mapper, inplace=True)

        other_info = pd.Series({
            'Name': self.name,
            'Parent': self.parent.name if self.parent is not None else '',
            "Corpus Type": self.__class__.__name__,
            "Number of Documents": len(self),
            "Number of Total Words": self.dtm.total,
            "Size of Vocabulary": len(self.dtm.vocab(nonzero=True)),
        })

        meta_info = pd.Series({
            "metas": ', '.join(self._meta_registry.keys())
        })
        return pd.concat([other_info, docs_info, meta_info]).to_frame(name='')

    def sample(self, n: int, rand_stat=None) -> TCorpus:
        """ Uniformly sample from the corpus. """
        mask = pd.Series(np.zeros(len(self)), dtype=bool, index=self._df.index)
        mask[mask.sample(n=n, random_state=rand_stat).index] = True
        return self.cloned(mask)

    def cloned(self, mask: 'pd.Series[bool]') -> TCorpus:
        """ Returns a (usually smaller) clone of itself with the boolean mask applied. """
        cloned_docs = self._cloned_docs(mask)
        cloned_metas = self._cloned_metas(mask)
        # cloned_dtms = self._cloned_dtms(mask)

        clone = Corpus(cloned_docs, cloned_metas)
        # clone._dtm_registry = cloned_dtms
        clone._parent = self
        return clone

    def _cloned_docs(self, mask) -> pd.Series:
        return self.docs().loc[mask]

    def detached(self) -> 'Corpus':
        """ Detaches from corpus tree and returns the corpus as root.

        DTM will be regenerated when accessed - hence a different vocab.
        """
        df = self._df.copy().reset_index(drop=True)
        detached = self.__class__(text=df[self._COL_DOC])
        return detached

    def __len__(self):
        return len(self._df) if self._df is not None else 0

    def __iter__(self):
        col_text_idx = self._df.columns.get_loc(self._COL_DOC)
        for i in range(len(self)):
            yield self._df.iat[i, col_text_idx]

    def __getitem__(self, item: int | slice) -> Docs:
        """ Returns """
        if isinstance(item, int):
            return self.docs().iloc[item]
        elif isinstance(item, slice):  # i.e. type=slice
            if item.step is not None:
                raise NotImplementedError("Slicing with step is currently not implemented.")
            start = item.start
            stop = item.stop
            if start is None: start = 0
            if stop is None: stop = len(self._df)
            return self.docs().iloc[start:stop]
        else:
            raise NotImplementedError("Only supports int and slice.")

    def __eq__(self, other: 'Corpus') -> bool:
        # todo: == should compare the len, index, dataframe.
        is_eq_df: bool = self._df.equals(other._df)
        raise NotImplementedError()
