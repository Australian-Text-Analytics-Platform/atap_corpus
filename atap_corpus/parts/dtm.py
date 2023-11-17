import contextlib
import io
import itertools
import logging
from collections import Counter
from pathlib import Path
from typing import Union, Iterable, Optional, Callable
import zipfile

import pandas as pd
import numpy as np
import scipy.sparse
import srsly
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import CountVectorizer

from atap_corpus.interfaces import TSerialisable
from atap_corpus.parts.base import BaseDTM, TFreqTable
from atap_corpus._types import Docs, Doc, PathLike

""" Document Term Matrix DTM

DTM is a container for the document term sparse matrix.
This container allows you to access term vectors and document vectors.
It also allows you to clone itself with a row index. In the cloned DTM,
a reference to the root dtm is passed down and the row index is used to
slice the child dtm each time.
This serves 3 purposes:
    1. preserve the original/root vocabulary.
    2. performance reasons so that we don't need to rebuild a dtm each time.
    3. indexing is a very inexpensive operation.

dev notes:
+ csr matrix is chosen for efficient cloning since they're row-wise slicing operations.
"""

DEFAULT_COUNTVEC_TOKENISER_PATTERN = r'(?u)\b\w{3,}\b'  # includes single letter words like 'a'

logger = logging.getLogger(__name__)


class DTM(BaseDTM):
    """ DTM
    This class is an abstract representation of the document-term matrix. It serves as a component
    of the Corpus class and exposes various functionalities that allows the slicing and dicing to be
    done seamlessly.

    Internally, DTM stores a sparse matrix which is computed using sklearn's CountVectorizer.
    """

    @classmethod
    def deserialise(cls, path_or_file: PathLike | io.IOBase) -> TSerialisable:
        """ Deserialise your path or IO into a DTM.
        If you've supplied an IO, it'll be closed.
        :param path_or_file: path or io.
        :return: DTM
        """
        # dev - probably not a good idea to close the IO if it is one.
        # but since base function converts path to IO, we need some flag OR scrap the pattern all togther.
        file, should_close = super().deserialise(path_or_file)

        with zipfile.ZipFile(file, mode='r') as z:
            mtx_path = Path("matrix").with_suffix(".npz")
            trm_path = Path("terms").with_suffix(".mpk")
            with z.open(mtx_path.__str__(), 'r') as mh:
                matrix = load_npz(mh)
            with z.open(trm_path.__str__(), 'r') as th:
                terms = srsly.msgpack_loads(th.read())
        if should_close: file.close()
        return cls.from_matrix(matrix, terms)

    def serialise(self, path_or_file: PathLike | io.IOBase) -> PathLike:
        """ Serialises the DTM as a zip.
        If you've supplied an IO, it'll be closed.

        :param path_or_file:
        :return: serialised path or IO (closd).
        """
        # dev - probably not a good idea to close the IO if it is one.
        # but since base function converts path to IO, we need some flag OR scrap the pattern all togther.
        file, should_close = super().serialise(path_or_file)

        with zipfile.ZipFile(file, mode='w') as z:
            mtx_path = Path("matrix").with_suffix(".npz")
            trm_path = Path("terms").with_suffix(".mpk")
            with z.open(mtx_path.__str__(), 'w') as mh:
                save_npz(mh, self.matrix)
            with z.open(trm_path.__str__(), 'w') as th:
                th.write(srsly.msgpack_dumps(self.terms))
        if should_close: file.close()
        return path_or_file

    @classmethod
    def from_docs_with_vectoriser(cls, docs: Docs | Iterable[Doc],
                                  token_pattern: str = DEFAULT_COUNTVEC_TOKENISER_PATTERN) -> 'DTM':
        """ Initialise a DTM from a collection of documents. """
        cvectoriser = CountVectorizer(token_pattern=token_pattern)
        matrix = cvectoriser.fit_transform(docs)
        terms = cvectoriser.get_feature_names_out()
        dtm = cls()
        return dtm._init(matrix, terms)

    @classmethod
    def from_matrix(cls, matrix: Union[np.ndarray, np.matrix, csr_matrix], terms: np.ndarray | list[str]):
        """ Initialise a DTM from a matrix (np, scipy.sparse). """
        num_terms: int
        if isinstance(terms, list): num_terms = len(terms)
        elif isinstance(terms, np.ndarray) and len(terms.shape) == 1: num_terms = terms.shape[0]
        elif isinstance(terms, np.ndarray) and len(terms.shape) == 2: num_terms = terms.shape[1]
        else: raise ValueError(f"Expecting terms to be either list or array but got {type(terms)}.")
        assert matrix.shape[1] == num_terms, f"Mismatched terms. Matrix shape {matrix.shape} and {num_terms} terms."
        if not scipy.sparse.issparse(matrix): matrix = csr_matrix(matrix)
        return cls()._init(matrix, terms)

    @classmethod
    def from_docs(cls, docs: Docs, tokeniser_func: Callable[[Doc], list[str]]):
        docs = pd.Series(docs)  # dev - using pandas dependency here, we should refactor this to DataFrameDTM.
        series_of_terms: 'pd.Series[list[str]]' = docs.apply(tokeniser_func)
        import inspect
        return_annot = inspect.signature(tokeniser_func).return_annotation
        if return_annot is not list[str]:
            # dev - tradeoff taken here, a list[Any] is accepted else check will take too long.
            if not series_of_terms.apply(type).eq(list).all():
                raise TypeError("The tokeniser_func provided did not return a list.")

        terms = np.array(sorted(set(itertools.chain.from_iterable(series_of_terms))))
        matrix = lil_matrix((len(docs), len(terms)),
                            dtype=np.int32)  # perf: lil_matrix is most efficient for row-wise replacement.
        for i, doc_terms in enumerate(series_of_terms):
            doc_terms = Counter(doc_terms)
            count_vector: np.ndarray = np.array([doc_terms.get(t, 0) for t in terms], dtype=np.int32)
            matrix[i] = count_vector

        dtm = cls()
        return dtm._init(matrix.tocsr(), terms)

    def __init__(self):
        super().__init__()
        self._matrix: Optional[csr_matrix] = None
        self._terms: Optional[np.ndarray] = None  # dev - np.ndarray is used to enable masking.
        self._term_idx_map: Optional[dict[str, int]] = None

        # only used for child dtms (do not override these)
        self._row_indices = None
        self._col_indices = None

    def _init(self, matrix: csr_matrix, terms: np.ndarray) -> 'DTM':
        """ Proper initialisation of the root DTM.
        _matrix and _terms needs to hold None to adhere to tree-based Clonable DTM (saves memory)

        dev - use this function when initialising a root DTM. e.g. in from_* class methods.
        """
        self._matrix: csr_matrix = matrix
        self._terms: np.ndarray[str] = terms
        self._term_idx_map: dict[str, int] = {self._terms[idx]: idx for idx in range(len(self._terms))}
        return self

    @property
    def matrix(self) -> csr_matrix:
        matrix = self.find_root()._matrix
        if self._row_indices is not None:
            matrix = matrix[self._row_indices, :]
        if self._col_indices is not None:
            matrix = matrix[:, self._col_indices]
        return matrix

    @property
    def terms(self) -> list[str]:
        """ Return the terms in the current dtm. """
        terms = self.find_root()._terms
        terms = terms if self._col_indices is None else terms[self._col_indices]
        return list(terms)

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def num_terms(self):
        return self.matrix.shape[1]

    @property
    def num_docs(self):
        return self.matrix.shape[0]

    @property
    def total(self):
        return self.matrix.sum()

    @property
    def terms_vector(self):
        """ Returns a vector of term counts for each term. """
        return np.asarray(self.matrix.sum(axis=0)).squeeze(axis=0)

    @property
    def docs_vector(self):
        """ Returns a vector of term counts for each document. """
        return np.asarray(self.matrix.sum(axis=1)).squeeze(axis=1)

    def vocab(self, nonzero: bool = False) -> set[str]:
        """ Returns a set of terms in the current dtm. """
        if nonzero:
            return set(self.terms[self.terms_freq_vector.nonzero()[0]])
        else:
            return set(self.terms)

    def vectors_of(self, terms: Union[str, list[str]]) -> csr_matrix:
        """ Return a subset of DTM matrix given terms.
        If a provided term is not found, those vectors will be missing from the matrix.
        """
        cols: Union[int, list[int]]
        term_idx_map = self.find_root()._term_idx_map
        if isinstance(terms, str):
            cols = term_idx_map.get(terms, list())
        else:
            cols = list()
            for term in terms:
                col = term_idx_map.get(term, None)
                if col is None: continue
                cols.append(col)
        return self.matrix[:, cols]

    def _is_built(self) -> bool:
        return self.find_root()._matrix is not None

    def cloned(self, mask: 'pd.Series[bool]') -> 'DTM':
        clone = super().cloned(mask)
        row_indices = mask[mask].index
        clone._row_indices = row_indices
        if clone._is_built:
            try:
                clone.matrix
            except Exception as e:
                raise RuntimeError([RuntimeError("Failed to clone DTM."), e])
        return clone

    def detached(self) -> 'DTM':
        detached = self.__class__()
        detached._init(self.matrix, np.array(self.terms))
        return detached

    def to_dataframe(self) -> 'pd.DataFrame[csr_matrix]':
        return pd.DataFrame.sparse.from_spmatrix(self.matrix, columns=self.terms)

    def to_lists_of_terms(self) -> list[list[str]]:
        """ Return the DTM as lists of list of terms."""
        nonzeros = self.matrix.nonzero()
        word_lists = [list() for _ in range(self.shape[0])]
        for row, col in zip(*nonzeros):
            freq = self.matrix[row, col]
            term = self.terms[col]
            terms = [term] * int(freq)
            word_lists[row].extend(terms)
        return word_lists

    def to_freqtable(self) -> TFreqTable:
        pass  # todo

    def shares_vocab(self, other: 'DTM') -> bool:
        """ Check if the other DTM shares current DTM's vocab """
        this, other = self.vocab(nonzero=True), other.vocab(nonzero=True)
        if not len(this) == len(other): return False
        return len(this.difference(other)) == 0

    def __repr__(self):
        if self._is_built:
            return f"<DTM {self.num_docs} docs X {self.num_terms} terms>"
        else:
            return f"<DTM Uninitialised>"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other: 'DTM') -> bool:
        return (other.matrix.shape == self.matrix.shape and
                self._terms_aligned(other) and
                other.matrix.dtype == self.matrix.dtype and
                other.matrix.nnz == self.matrix.nnz and  # number of non-zero values (i.e. stored values)
                not (other.matrix != self.matrix).sum() == np.prod(self.matrix.shape))  # reversed logic != is faster.

    # -- allow for context manager to remove terms temporarily.
    # possible use case: when we want to temporarily remove stopwords from the DTM for downstream analysis.

    @contextlib.contextmanager
    def without_terms(self, terms: Union[list[str], set[str]]) -> 'DTM':
        """ Expose a temporary dtm object without a list of terms. Terms not found are ignored. """
        try:
            features = self.find_root()._terms
            self._col_indices = np.isin(features, list(terms), invert=True).nonzero()[0]
            yield self
        finally:
            self._col_indices = None

    @contextlib.contextmanager
    def with_terms(self, terms: Union[list[str], set[str]]) -> 'DTM':
        try:
            features = self.find_root()._terms
            self._col_indices = np.isin(features, list(terms), invert=False).nonzero()[0]
            yield self
        finally:
            self._col_indices = None

    # -- merging with another DTM - this function is not used.

    def merged(self, other: 'DTM'):
        """Merge other DTM with current."""
        if self._terms_aligned(other):
            m = scipy.sparse.vstack((self.matrix, other.matrix))
            feature_names_out = self._terms  # unchanged since terms are shared
        else:
            if len(other.terms) >= len(self.terms):
                big, small = other, self
            else:
                big, small = self, other

            top_right, indx_missing = self._build_top_right_merged_matrix(big, small)
            top_left = self._build_top_left_merged_matrix(small)
            top = scipy.sparse.hstack((top_left, top_right))

            num_terms_sm_and_bg = small.num_terms + indx_missing.shape[0]
            assert top.shape[0] == small.num_docs and top.shape[1] == num_terms_sm_and_bg, \
                f"Top matrix incorrect shape: Expecting ({small.num_docs, num_terms_sm_and_bg}. Got {top.shape}."

            bottom_left = self._build_bottom_left_merged_matrix(big, small)  # shape=(big.num_docs, small.num_terms)
            bottom_right = self._build_bottom_right_merged_matrix(big,
                                                                  indx_missing)  # shape=(big.num_docs, missing terms from big)
            bottom = scipy.sparse.hstack((bottom_left, bottom_right))
            logger.debug(f"MERGE: merged bottom matrix shape: {bottom.shape} type: {type(bottom)}.")
            assert bottom.shape[0] == big.num_docs and bottom_left.shape[1] == small.num_terms, \
                f"Bottom matrix incorrect shape: Expecting ({big.num_docs}, {num_terms_sm_and_bg}). Got {bottom.shape}."

            m = scipy.sparse.vstack((top, bottom))
            logger.debug(f"MERGE: merged matrix shape: {m.shape} type: {type(m)}.")
            assert m.shape[1] == num_terms_sm_and_bg, \
                f"Terms incorrectly merged. Total unique terms: {num_terms_sm_and_bg}. Got {m.shape[1]}."
            num_docs_sm_and_bg = big.num_docs + small.num_docs
            assert m.shape[0] == num_docs_sm_and_bg, \
                f"Documents incorrectly merged. Total documents: {num_docs_sm_and_bg}. Got {m.shape[0]}."
            feature_names_out = np.concatenate([small.terms, big.terms[indx_missing]])

        # replace with new matrix.
        other = DTM()
        other._matrix = m
        other._terms = feature_names_out
        return other

    def _build_top_right_merged_matrix(self, big, small):
        # 1. build top matrix: shape = (small.num_docs, small.num_terms + missing terms from big)
        # perf: assume_uniq - improves performance and terms are unique.
        mask_missing = np.isin(big.terms, small.terms, assume_unique=True, invert=True)
        indx_missing = mask_missing.nonzero()[0]
        # create zero matrix in top right since small doesn't have these terms in their documents.
        top_right = scipy.sparse.csr_matrix((small.num_docs, indx_missing.shape[0]), dtype=small.matrix.dtype)
        return top_right, indx_missing

    def _build_top_left_merged_matrix(self, small):
        return small.matrix

    def _build_bottom_left_merged_matrix(self, big, small):
        # 2. build bottom matrix: shape = (big.num_docs, small.num_terms + missing terms from big)
        # bottom-left: shape = (big.num_docs, small.num_terms)
        #   align overlapping term indices from big with small term indices
        intersect = np.intersect1d(big.terms, small.terms, assume_unique=True, return_indices=True)
        intersect_terms, bg_intersect_indx, sm_intersect_indx = intersect
        bottom_left = scipy.sparse.lil_matrix((big.num_docs, small.num_terms))  # perf: lil for column replacement
        for i, idx in enumerate(sm_intersect_indx):
            bottom_left[:, idx] = big.matrix[:, bg_intersect_indx[i]]
        logger.debug(f"MERGE: bottom left matrix shape: {bottom_left.shape}")
        bottom_left = bottom_left.tocsr(copy=False)  # convert to csr to match with rest of submatrices
        return bottom_left

    def _build_bottom_right_merged_matrix(self, big, indx_missing):
        return big.matrix[:, indx_missing]

    def _terms_aligned(self, other: 'DTM') -> bool:
        """ Check if the other DTM's terms are index aligned with current DTM """
        if not len(self.terms) == len(other.terms): return False
        return self.terms == other.terms
