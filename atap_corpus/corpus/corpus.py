import logging
from typing import Optional, Generator

import numpy as np
import pandas as pd
import spacy
import spacy.tokens
from tqdm.auto import tqdm

from atap_corpus.corpus.base import BaseCorpus
from atap_corpus.corpus.mixins import SpacyDocsMixin
from atap_corpus.registry import _Unique_Name_Provider
from atap_corpus.types import PathLike, Docs, Mask
from atap_corpus.utils import format_dunder_str

logger = logging.getLogger(__name__)


def ensure_docs(docs: pd.Series) -> Docs:
    if isinstance(docs, list | set | tuple):
        docs = pd.Series(docs)
    if not isinstance(docs, pd.Series):
        raise TypeError(f"Docs must be pd.Series for DataFrameCorpus. Got {type(docs)}.")
    return docs.apply(lambda d: str(d) if not isinstance(d, spacy.tokens.Doc) else d)


class DataFrameCorpus(BaseCorpus, SpacyDocsMixin):
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
    _COL_DOC: str = 'document_'

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, col_doc: str = _COL_DOC, name: str = None) -> 'Corpus':
        if col_doc not in df.columns:
            raise ValueError(f"Column {col_doc} not found. You must set the col_doc argument.\n"
                             f"Available columns: {df.columns}")
        corpus = cls(df[col_doc], name=name)
        return corpus

    def to_dataframe(self):
        """ Export corpus as a dataframe. """
        return self._df.copy().reset_index(drop=True)

    def serialise(self, path: PathLike) -> PathLike:
        path = super().serialise(path)
        raise NotImplementedError()

    @classmethod
    def deserialise(cls, path: PathLike) -> 'Corpus':
        raise NotImplementedError()

    def __init__(self, docs: Optional[pd.Series | list[str]] = None, name: str = None):
        super().__init__(name=name)
        if docs is None: docs = pd.Series(list())
        self._df: pd.DataFrame = pd.DataFrame(ensure_docs(docs), columns=[self._COL_DOC])
        # ensure initiated object is well constructed.
        assert len(list(filter(lambda x: x == self._COL_DOC, self._df.columns))) <= 1, \
            f"More than 1 {self._COL_DOC} column in dataframe."

        self._mask: Mask = pd.Series(np.full(len(self._df), True))
        # dev - a full mask is kept for root to avoid excessive conditional checks. Binary mask is memory cheap.
        # a million documents should equate to ~1Mb
        self._parent: Optional['Corpus'] = None

    def rename(self, name: str):
        self.name = name

    @property
    def parent(self) -> 'Corpus':
        """ Returns a read-only reference to the parent corpus. """
        return self._parent

    @property
    def is_root(self) -> bool:
        return self._parent is None

    def find_root(self) -> 'Corpus':
        """ Find and return a copied reference to the root corpus. """
        if self.is_root: return self
        parent = self.parent
        while not parent.is_root:
            parent = parent.parent
        return parent

    def cloned(self, mask: 'pd.Series[bool]', name: Optional[str] = None) -> 'Corpus':
        """ Returns a clone of itself by applying the boolean mask.
        The returned clone will retain a parent-child relationship from which it is cloned.
        To create a clone without the parent-child relationship, call detached() and del cloned.

        subcorpus names follows the dot notation.
        """
        if not isinstance(mask, pd.Series): raise TypeError(f"Mask is not a pd.Series. Got {type(mask)}.")
        if not mask.isin((0, 1)).all():
            raise ValueError(f"Mask pd.Series is not a valid mask. Must be either boolean or binary.")
        mask = mask.astype('bool')
        name = f"{self.name}." if name is None else f"{self.name}.{name}"  # dot notation
        name = _Unique_Name_Provider.unique_name_number_suffixed(name)
        clone = self.__class__(name=name)
        clone._parent = self
        clone._mask = mask
        return clone

    def detached(self) -> 'Corpus':
        """ Detaches from corpus tree and returns a new Corpus instance as root. """
        df = self._df.copy().reset_index(drop=True)
        name = f"{self.name}-detached"
        name = _Unique_Name_Provider.unique_name_number_suffixed(name=name)
        detached = self.__class__(df[self._COL_DOC], name=name)
        return detached

    def docs(self) -> Docs:
        return self.find_root()._df.loc[self._mask, self._COL_DOC]

    @property
    def metas(self) -> list[str]:
        cols = list(self._df.columns)
        cols.remove(self._COL_DOC)
        return cols

    def get_meta(self, name: str) -> pd.Series:
        """ Get the meta series based on its name and return the entire series. """
        if name == self._COL_DOC:
            raise KeyError(f"{name} is reserved for Corpus documents. It is never used for meta data.")
        return self._df.loc[:, name]

    def add_meta(self, series: pd.Series | list | tuple, name: Optional[str] = None):
        """ Adds a meta series into the Corpus. Realigns index with Corpus.
        If mismatched size: raises ValueError.
        """
        if len(series) != len(self):
            raise ValueError(
                f"Added meta {series} does not align with Corpus size. Expecting {len(self)} Got {len(series)}"
            )
        if not isinstance(series, pd.Series | list | tuple):
            raise TypeError("Meta must either be pd.Series, list or tuple.")
        if isinstance(series, list | tuple):
            series = pd.Series(series)
        series = series.reindex(self._df.index)
        if name is None: name = series.name
        if name == self._COL_DOC:
            raise KeyError(f"Name of meta {name} conflicts with internal document name. Please rename.")
        try:
            # df.itertuples() uses namedtuple
            # see https://docs.python.org/3/library/collections.html#collections.namedtuple
            _ = namedtuple('_', [name])
        except ValueError as _:
            raise KeyError(f"Name of meta {name} must be a valid field name. Please rename.")
        self._df[name] = series

    def remove_meta(self, name: str):
        """ Removes the meta series from the Corpus. """
        self._df.drop(name, axis=1, inplace=True)
        assert name not in self.metas, f"meta: {name} did not get removed from Corpus. Try again."

    def sample(self, n: int, rand_stat=None) -> 'Corpus':
        """ Uniformly sample from the corpus. This creates a clone. """
        mask = pd.Series(np.zeros(len(self)), dtype=bool, index=self._df.index)
        mask[mask.sample(n=n, random_state=rand_stat).index] = True
        name = _Unique_Name_Provider.unique_name_number_suffixed(f"{self.name}-{n}samples")
        return self.cloned(mask, name=name)

    def __len__(self):
        if self.is_root:
            return len(self._df) if self._df is not None else 0
        else:
            return sum(self._mask)

    def __iter__(self):
        col_text_idx = self._df.columns.get_loc(self._COL_DOC)
        for i in range(len(self)):
            yield self._df.iat[i, col_text_idx]

    def __getitem__(self, item: int | slice) -> Docs:
        """ Returns a document or slice of corpus. or metadata series if str."""
        if len(self) == 0:
            raise KeyError("Corpus is empty.")
        if isinstance(item, str):
            # todo: accesses meta data
            pass
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

    def __str__(self) -> str:
        return format_dunder_str(self.__class__, self.name, {"size": len(self)})

    # -- SpacyDocMixin --
    def uses_spacy(self) -> bool:
        if len(self) > 0:
            return type(self[0]) == spacy.tokens.Doc
        else:
            return False

    def run_spacy(self, nlp: spacy.Language,
                  reprocess_prompt: bool = True,
                  progress_bar: bool = True,
                  *args, **kwargs, ) -> None:
        """ Process the Corpus with a spacy pipeline from the root.
        If you only want to process a subcorpus, first call .detached().

        If the Corpus has already been processed and reprocess = True, it'll reprocessed from scratch.
        :param nlp: spacy pipeline
        :param reprocess_prompt: Set as False to skip user input. If False, prompt user whether to reprocess.
        :param progress_bar: show progress bar.
        """
        # dev - spacy always processed from root unless detached() otherwise it'll introduce too much complexity.
        super().run_spacy(nlp=nlp, *args, **kwargs)
        run_spacy_on: DataFrameCorpus = self.find_root()
        docs: Generator[str, None, None]
        pb_desc, pb_colour = "Processing: ", 'orange'
        if self.uses_spacy():
            logger.warning("This Corpus has already been processed by spacy. It'll be reprocessed.")
            if reprocess_prompt:
                inp = input("Are you sure you want to reprocess the Corpus? (y/n): ")
                if not inp.lower() == 'y':
                    return
            # dev - sometimes spacy pipelines are incompatible, better to be reprocessed as string.
            docs = (d.text for d in run_spacy_on.docs())
            pb_desc, pb_colour = "Reprocessing: ", 'blue'
        else:
            docs = (d for d in run_spacy_on.docs())

        if progress_bar:
            docs = tqdm(docs, total=len(run_spacy_on), desc=pb_desc, colour=pb_colour)

        run_spacy_on._df[run_spacy_on._COL_DOC] = pd.Series(nlp.pipe(docs))
        if not run_spacy_on.uses_spacy():
            raise RuntimeError("Did not seem to have properly processed Corpus with spacy. Corpus could be invalid.")
