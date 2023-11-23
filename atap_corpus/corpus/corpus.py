import logging
import weakref as wref
import zipfile
from collections import namedtuple
from typing import Optional, Generator, Type, IO, Iterator, Hashable

import numpy as np
import pandas as pd
import spacy
import spacy.tokens
import srsly
from tqdm.auto import tqdm

from atap_corpus._types import PathLike, Docs, Mask, MPK_SUPPORTED
from atap_corpus.corpus.base import BaseCorpusWithMeta
from atap_corpus.mixins import SpacyDocsMixin, ClonableDTMRegistryMixin
from atap_corpus.parts.base import BaseDTM
from atap_corpus.parts.dtm import DTM
from atap_corpus.registry import _Unique_Name_Provider
from atap_corpus.slicer.slicer import CorpusSlicer
from atap_corpus.utils import format_dunder_str

logger = logging.getLogger(__name__)


def ensure_docs(docs: pd.Series) -> Docs:
    if isinstance(docs, list | set | tuple):
        docs = pd.Series(docs)
    if not isinstance(docs, pd.Series):
        raise TypeError(f"Docs must be pd.Series for DataFrameCorpus. Got {type(docs)}.")
    return docs.apply(lambda d: str(d) if not isinstance(d, spacy.tokens.Doc) else d)


class DataFrameCorpus(SpacyDocsMixin, ClonableDTMRegistryMixin, BaseCorpusWithMeta):
    """ Corpus
    This class abstractly represents a corpus which is a collection of documents.
    Each document is also described by their metadata and is used for functions such as slicing.

    An important component of the Corpus is that it also holds the document-term matrix as a sparse matrix.

    A main design feature of the corpus is to allow for seamless slicing and dicing based on the associated metadata
    and text in the documents. See class CorpusSlicer. After each slicing operation, new but sliced DataFrameCorpus is
    returned exposing the same descriptive functions (e.g. summary()) you may wish to call again.

    Internally, documents are stored as rows of string in a dataframe. Metadata are stored in the meta registry.
    Slicing is equivalent to creating a `cloned()` corpus and is really passing a boolean mask to the dataframe and
    the associated metadata series. When sliced, corpus objects are created with a reference to its parent corpus.
    This is mainly for memory and performance reasons, so that the expensive DTM computed may be reused and
    a shared vocabulary is kept for easier analysis of different sliced sub-corpus.

    You may choose the corpus to be `detached()`. This returns itself as a root DataFrameCorpus and the lineage is
    discarded.
    """

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, col_doc: Optional[str] = None, name: str = None) -> 'DataFrameCorpus':
        if col_doc is None: col_doc = cls._COL_DOC
        if col_doc not in df.columns:
            raise ValueError(f"Column {col_doc} not found. You must set the col_doc argument.\n"
                             f"Available columns: {df.columns}")
        df = df.copy().reset_index(drop=True)

        corpus = cls(df[col_doc], name=name)
        col_metas = [c for c in df.columns if c != col_doc]
        for col_meta in col_metas:
            corpus.add_meta(df.loc[:, col_meta], col_meta)
        return corpus

    def to_dataframe(self):
        """ Export corpus as a dataframe. """
        return self._root_df_with_masked_applied().copy().reset_index(drop=True)

    def serialise(self, path_or_file: PathLike | IO,
                  metas: list[str] | bool = True, dtms: list[str] | bool = True) -> PathLike:
        """ Serialises the Corpus as a zip.
        :param path_or_file: path or io.
        :param metas: serialise all metadata or provided list.
        :param dtms: serialise all dtms or provided list.
        :return: serialised path or IO (closed).
        """
        file, should_close = super().serialise(path_or_file=path_or_file)
        if not self.is_root:
            logger.warning("You are serialising a subcorpus. When you deserialise this it'll be a root corpus.")

        if metas is True: metas = self.metas
        cols = [self._COL_DOC]
        cols.extend(metas)
        if dtms is True: dtms = list(self.dtms.keys())

        to_serialise_df = self._root_df_with_masked_applied().loc[:, cols]
        with zipfile.ZipFile(file, 'w') as z:
            with z.open("corpus.parquet", 'w') as zdf:
                to_serialise_df.to_parquet(zdf, engine='pyarrow', index=False)

            with z.open("corpus.attribs", 'w') as zattrib:
                zattrib.write(srsly.msgpack_dumps(self._attributes))

            # python3.10 uses .writestr and >3.11 uses .mkdir
            dtm_dir = "dtms"
            z.writestr(f"{dtm_dir}/", '')  # works on MacOS, Linux (binder instance)
            for dtm_key in dtms:
                dtm: BaseDTM = self.dtms.get(dtm_key)
                with z.open(f"{dtm_dir}/{dtm_key}.zip", 'w') as dtmz:
                    dtm.serialise(dtmz)

            z.writestr("name", self.name.encode("utf-8"))
        if should_close: file.close()
        return path_or_file

    @classmethod
    def deserialise(cls, path_or_file: PathLike | IO) -> 'DataFrameCorpus':
        """ Deserialises your path or IO into a DataFrameCorpus.
        :param path_or_file: path or io.
        :return: DataFrameCorpus.
        """
        file, shoud_close = super().deserialise(path_or_file)

        df: Optional[pd.DataFrame] = None
        name: Optional[str] = None
        dtms: dict[str, cls.dtm_cls()] = dict()
        with zipfile.ZipFile(file, 'r') as z:
            dtm_dir = "dtms"
            files = z.namelist()
            f: str
            for f in files:
                if f.endswith("/"): continue
                if f.startswith(f"{dtm_dir}/"):
                    name = f[f.rfind("/") + 1:f.rfind('.')]  # zipname is used as dtm name. (exclude suffix)
                    with z.open(f, 'r') as dtm_h:
                        dtm = cls.dtm_cls().deserialise(dtm_h)
                        dtms[name] = dtm
                if f == "corpus.parquet":
                    with z.open(f, 'r') as df_h:
                        df = pd.read_parquet(df_h)
                if f == "name":
                    with z.open(f, 'r') as n_h:
                        name = str(n_h.read().decode('utf-8'))
                if f == "corpus.attribs":
                    with z.open(f, 'r') as a_h:
                        attribs = srsly.msgpack_loads(a_h.read())
        if shoud_close: file.close()
        if df is None:
            raise FileNotFoundError("Missing corpus.parquet file in your zip.")
        else:
            if name is None: logger.warning("Missing corpus name from your zip.")
            corpus = cls(docs=df, name=name)
            corpus._attributes = attribs
            corpus._ClonableDTMRegistryMixin__dtms = dtms
            return corpus

    def __init__(self, docs: Optional[pd.DataFrame | pd.Series | list[str]] = None, name: str = None):
        super().__init__(name=name)
        if not self.is_root:
            self._df = None  # allow for cloned to not hold a self._df
        else:
            if docs is None:
                docs = list()
            if isinstance(docs, list | Iterator):
                docs = pd.Series(docs, name=self._COL_DOC)
                self._df: pd.DataFrame = pd.DataFrame(ensure_docs(docs))
            elif isinstance(docs, pd.Series):
                docs.name = self._COL_DOC
                self._df: pd.DataFrame = pd.DataFrame(ensure_docs(docs))
            elif isinstance(docs, pd.DataFrame):
                if self._COL_DOC not in docs.columns:
                    raise ValueError(f"Column {self._COL_DOC} not found. You must set the col_doc argument.\n"
                                     f"Available columns: {docs.columns}")
                ensure_docs(docs.loc[:, self._COL_DOC])
                self._df = docs
            else:
                raise ValueError(f"{docs} must be either a Series, list or DataFrame.")

        # ensure initiated object is well constructed.
        if self._df is not None:
            assert len(list(filter(lambda x: x == self._COL_DOC, self._df.columns))) <= 1, \
                f"More than 1 {self._COL_DOC} column in dataframe."

        # from super - Clonable - nothing is overwritten here just typehints.
        self._parent: Optional[DataFrameCorpus]
        self._mask: Optional[Mask]

        self._slicer = CorpusSlicer(wref.ref(self))

    def cloned(self, mask: Mask, name: Optional[str] = None) -> 'DataFrameCorpus':
        """ Returns a clone of itself by applying the boolean mask.
        The returned clone will retain a parent-child relationship from which it is cloned.
        To create a clone without the parent-child relationship, call detached() and del cloned.

        subcorpus names follows the dot notation.
        """
        if not isinstance(mask, pd.Series): raise TypeError(f"Mask is not a pd.Series. Got {type(mask)}.")
        if not mask.isin((0, 1)).all():
            raise ValueError(f"Mask pd.Series is not a valid mask. Must be either boolean or binary.")
        mask = mask.astype('bool')
        name = self.name if name is None else name  # dot notation
        name = _Unique_Name_Provider.unique_name_number_suffixed(name)
        clone: DataFrameCorpus
        clone = super().cloned(mask, name=name)
        clone._attributes = self.attributes
        return clone

    def detached(self) -> 'DataFrameCorpus':
        """ Detaches from corpus tree and returns a new Corpus instance as root. """
        df = self._root_df_with_masked_applied().copy().reset_index(drop=True)
        name = f"{self.name}-detached"
        name = _Unique_Name_Provider.unique_name_number_suffixed(name=name)
        detached = self.__class__(df[self._COL_DOC], name=name)
        return detached

    def _root_df_with_masked_applied(self) -> pd.DataFrame:
        """ Return a 'copy' of the root dataframe with current mask applied. """
        return self._df if self.is_root else self.find_root()._df.loc[self._mask, :]

    def attribute(self, key: Hashable, value: MPK_SUPPORTED):
        # serialisation in DataFrameCorpus uses msgpack for attributes.
        try:
            srsly.msgpack_dumps(dict(key=value))
        except TypeError:
            raise TypeError("key or value is not serialisable by msgpack.")
        super().attribute(key=key, value=value)

    def docs(self) -> Docs:
        """ Return the collection of docs. """
        return self._root_df_with_masked_applied().loc[:, self._COL_DOC]

    @property
    def slicer(self) -> CorpusSlicer:
        """ Returns the slicer for this corpus. """
        return self._slicer

    @property
    def s(self) -> CorpusSlicer:
        """ Shorthand for slicer. """
        return self.slicer

    @property
    def metas(self) -> list[str]:
        """ Returns a list of strings representing the metadata in the Corpus. """
        cols = list(self.find_root()._df.columns)
        cols.remove(self._COL_DOC)
        return cols

    def get_meta(self, name: str) -> pd.Series:
        """ Get a shallow copy of the metadata collection based on its name. """
        if name == self._COL_DOC:
            raise KeyError(f"{name} is reserved for Corpus documents. It is never used for meta data.")
        return self._root_df_with_masked_applied().loc[:, name].copy()

    def add_meta(self, meta: pd.Series | np.ndarray | list | tuple, name: Optional[str] = None):
        """ Adds a meta series into the Corpus. Realigns index with Corpus.

        :arg meta - your metadata collection. Can be a series, list or tuple.
        :arg name - provide a name for the metadata. (It can only be None if meta is a series)
        :raises ValueError - if name is invalid.
        :raises ValueError - if metadata size is mismatched.

        If metadata is added to a clone, it'll populate the root and have NaN values for
        the documents that are not in the clone.

        There is a restriction on the string for names as pandas uses namedtuple under the hood
        and downstream fn invocations may fail if it can't be used as a namedtuple. This is
        tested in the function and raises an error if it fails.
        """
        if not isinstance(meta, pd.Series | np.ndarray | list | tuple):
            raise TypeError("Meta must either be pd.Series, list or tuple.")
        meta = pd.Series(meta)
        if len(meta) != len(self):
            raise ValueError(f"Metadata collection size did not match. Expecting {len(self)}. Got {len(meta)}.")
        meta: pd.Series
        meta = meta.reindex(self._root_df_with_masked_applied().index)
        if name is None: name = meta.name
        else: meta.name = name
        if name == self._COL_DOC:
            raise KeyError(f"Name of meta {name} conflicts with internal document name. Please rename.")
        if not isinstance(name, str):
            # dev - this is due to our syntactic sugar in __getitem__
            raise ValueError("Only str meta names are supported.")
        try:
            # df.itertuples() use namedtuple - https://docs.python.org/3/library/collections.html#collections.namedtuple
            # field name restrictions apply
            _ = namedtuple('_', [name])
        except ValueError as _:
            raise ValueError(f"Name of meta {name} is not a valid field name. Please rename.")

        # mutating - hence do not use _masked_root_df
        self.find_root()._df[name] = meta

    def remove_meta(self, name: str):
        """ Removes the meta series from the Corpus. """

        # mutating - hence do not use _masked_root_df
        self.find_root()._df.drop(name, axis=1, inplace=True)
        assert name not in self.metas, f"meta: {name} did not get removed from Corpus. Try again."

    def sample(self, n: int, rand_stat: Optional[int] = None) -> 'DataFrameCorpus':
        """ Uniformly sample from the corpus. This creates a clone. """
        if n > len(self): raise ValueError(f"Unable to sample {n} from a corpus of size {len(self)}.")
        mask = pd.Series(np.zeros(shape=len(self.find_root())), dtype=bool)
        rng = np.random.default_rng(rand_stat)
        true_indices = rng.choice(self._root_df_with_masked_applied().index, size=n, replace=False)
        mask.loc[true_indices] = True
        name = f"{self.name}.{n}samples"
        return self.cloned(mask, name=_Unique_Name_Provider.unique_name_number_suffixed(name, delimiter='-'))

    def join(self, other: 'DataFrameCorpus', name: Optional[str] = None) -> 'DataFrameCorpus':
        """ Joins 2 DataFrameCorpus that are from the same tree together and return a joined clone from
        their first common ancestor.
        :param other: the DataFrameCorpus to join.
        :param name: name of the Corpus (default: common_parent.self.name+other.name)
        :return: joined DataFrameCorpus.
        """
        if not self.find_root() == other.find_root():
            raise TypeError(f"{other.name} is not derived from the same tree as {self.name}.")
        lca: 'DataFrameCorpus' = self.find_lowest_common_ancestor(other)
        mask = (self._mask | other._mask)
        name = f"{lca.name}.({self.name}+{other.name})" if name is None else name
        return lca.cloned(mask, name=name)

    def equals(self, other: 'DataFrameCorpus') -> bool:
        """ Checks if the other DataFrameCorpus is equivalent to this.
        The dataframe and all dtms are compared and must be exactly equal.
        Does not have to have the same Corpus name.
        Does not have to have the same Corpus lineage.
        """
        if not other._root_df_with_masked_applied().equals(self._root_df_with_masked_applied()):
            return False
        if not other._ClonableDTMRegistryMixin__dtms.keys() == self._ClonableDTMRegistryMixin__dtms.keys():
            return False
        for dtm_key in self._ClonableDTMRegistryMixin__dtms.keys():
            this_dtm = self._ClonableDTMRegistryMixin__dtms[dtm_key]
            other_dtm = other._ClonableDTMRegistryMixin__dtms[dtm_key]
            if this_dtm != other_dtm: return False
        return True

    def __len__(self):
        if self.is_root:
            return len(self._df) if self._df is not None else 0
        else:
            return sum(self._mask)

    def __iter__(self):
        col_text_idx = self._root_df_with_masked_applied().columns.get_loc(self._COL_DOC)
        for i in range(len(self)):
            yield self._df.iat[i, col_text_idx]

    def __getitem__(self, item: int | slice | str) -> Docs:
        """ Returns a document or slice of corpus. or metadata series if str.

        This raises an error when Corpus is empty or when accessing an index larger than Corpus size.
        """
        if len(self) == 0:
            raise IndexError("Empty corpus.")
        if isinstance(item, str):
            return self.get_meta(item)
        if isinstance(item, int):
            if item >= len(self):
                raise IndexError(f"You have given an index exceeding the Corpus size: {len(self)}.")
            if item < 0:
                raise IndexError("Index can only be positive integers.")
            return self.docs().iloc[item]
        elif isinstance(item, slice):  # i.e. type=slice
            if item.step is not None:
                raise NotImplementedError("Slicing with step is currently not implemented.")
            start = item.start
            stop = item.stop
            if start is None: start = 0
            if stop is None: stop = len(self._root_df_with_masked_applied())
            return self.docs().iloc[start:stop]
        else:
            raise NotImplementedError("Only supports int and slice.")

    def __str__(self) -> str:
        return format_dunder_str(self.__class__, self.name, **{"size": len(self)})

    # -- ClonableDTMRegistryMixin
    @classmethod
    def dtm_cls(cls) -> Type[BaseDTM]:
        return DTM

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
