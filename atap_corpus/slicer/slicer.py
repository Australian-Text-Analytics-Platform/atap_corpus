import logging
import weakref as wref
from datetime import datetime
from typing import Union, Callable, Optional, Any, Iterator

import pandas as pd

from atap_corpus._types import TCorpusWithMeta
from atap_corpus.slicer.op import *

logger = logging.getLogger(__name__)

__all__ = ["CorpusSlicer"]


# dev - current only supports dataframe corpus slicing - most of the code is re-usable but just not organised yet.
class CorpusSlicer(object):
    """ 'DataFrameCorpus'Slicer

    The Corpus slicer is used to slice the dataframe corpus.
    """

    def __init__(self, corpus: wref.ReferenceType[TCorpusWithMeta]):
        self._corpus: wref.ReferenceType[TCorpusWithMeta] = corpus
        # weakref is used to avoid cyclic refs in composition.

    def filter_by_condition(self, name: str, cond_func: Callable[[Any], bool]):
        """ Filter by condition
        :arg name: str - meta name
        :arg cond_func -  Callable that returns a boolean.
        """
        meta = self._corpus().get_meta(name)
        op = CallableOp(meta, cond_func)
        return self._corpus().cloned(op.mask())

    def filter_by_item(self, name: str, items):
        """ Filter by item - items can be str or numeric types.

        :arg name: str - meta name
        :arg items - the list of items to include OR just a single item.
        """
        meta = self._corpus().get_meta(name)
        op = ItemOp(meta, items)
        return self._corpus().cloned(op.mask())

    def filter_by_range(self, name: str, min_: Optional[Union[int, float]] = None,
                        max_: Optional[Union[int, float]] = None):
        """ Filter by a range [min, max). Max is non inclusive. """
        meta = self._corpus().get_meta(name)
        if min_ is None and max_ is None: return self._corpus()
        op = RangeOp(meta, min_, max_)
        return self._corpus().cloned(op.mask())

    def filter_by_regex(self, name: str, regex: str, ignore_case: bool = False):
        """ Filter by regex.

        :arg name: str - meta name
        :arg id - meta id
        :arg regex - the regex pattern
        :arg ignore_case - whether to ignore case
        """
        if name == self._corpus()._COL_DOC:
            meta = self._corpus().docs()
        else:
            meta = self._corpus().get_meta(name)
        op = RegexOp(meta, regex, ignore_case)
        return self._corpus().cloned(op.mask())

    def filter_by_datetime(self, name: str,
                           start: Optional[str | datetime] = None,
                           end: Optional[str | datetime] = None,
                           strftime: Optional[str] = None):
        """ Filter by datetime in range (start, end].
        :arg name: str - meta name
        :arg start - any datetime string recognised by pandas.
        :arg end - any datetime string recognised by pandas.
        :arg strftime - datetime string format

        If no start or end is provided, it'll return the corpus unsliced.
        """
        meta = self._corpus().get_meta(name)
        if start is None and end is None: return self._corpus()
        op = DatetimeOp(meta, start, end, strftime)
        return self._corpus().cloned(op.mask())

    def group_by(self, name: str, grouper: pd.Grouper = None) -> Iterator[tuple[str, TCorpusWithMeta]]:
        """ Return groups of the subcorpus based on their metadata.

        :arg name: str - meta name
        :arg grouper: pd.Grouper - as you would in pandas
        :return Iterable[tuple[groupid, subcorpus]]
        """
        meta = self._corpus().get_meta(name)
        return ((gid, self._corpus().cloned(mask)) for gid, mask in meta.groupby(grouper=grouper))
