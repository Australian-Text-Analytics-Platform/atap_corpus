import logging
import weakref as wref
from typing import Union, Callable, Optional, Any, Iterator

import pandas as pd

from atap_corpus.slicer.op import *

logger = logging.getLogger(__name__)

__all__ = ["DataFrameCorpusSlicer"]


# dev - current only supports dataframe corpus slicing - most of the code is re-usable but just not organised yet.
class DataFrameCorpusSlicer(object):
    """ 'DataFrameCorpus'Slicer

    The Corpus slicer is used to slice the dataframe corpus.
    """

    def __init__(self, corpus: wref.ReferenceType['DataFrameCorpus']):
        self._corpus: wref.ReferenceType['DataFrameCorpus'] = corpus

    def filter_by_condition(self, id_, cond_func: Callable[[Any], bool]):
        """ Filter by condition
        :arg id_ - meta's id
        :arg cond_func -  Callable that returns a boolean.
        """
        meta = self._corpus().get_meta(id_)
        mask = self._mask_by_condition(meta, cond_func)
        return self._corpus().cloned(mask)

    def filter_by_item(self, id_, items):
        """ Filter by item - items can be str or numeric types.

        :arg id_ - meta's id.
        :arg items - the list of items to include OR just a single item.
        """
        meta = self._corpus().get_meta(id_)
        op = ItemOp(meta, items)
        return self._corpus().cloned(op.mask())

    def filter_by_range(self, id_, min_: Optional[Union[int, float]] = None, max_: Optional[Union[int, float]] = None):
        """ Filter by a range [min, max). Max is non inclusive. """
        meta = self._corpus().get_meta(id_)
        if min_ is None and max_ is None: return self._corpus()
        op = RangeOp(meta, min_, max_)
        return self._corpus().cloned(op.mask())

    def filter_by_regex(self, id_, regex: str, ignore_case: bool = False):
        """ Filter by regex.
        :arg id - meta id
        :arg regex - the regex pattern
        :arg ignore_case - whether to ignore case
        """
        if id_ == self._corpus().COL_DOC:
            meta = self._corpus().docs()
        else:
            meta = self._corpus().get_meta(id_)
        op = RegexOp(meta, regex, ignore_case)
        return self._corpus().cloned(op.mask())

    def filter_by_datetime(self, id_, start: Optional[str] = None, end: Optional[str] = None,
                           strftime: Optional[str] = None):
        """ Filter by datetime in range (start, end].
        :arg start - any datetime string recognised by pandas.
        :arg end - any datetime string recognised by pandas.
        :arg strftime - datetime string format

        If no start or end is provided, it'll return the corpus unsliced.
        """
        meta = self._corpus().get_meta(id_)
        if start is None and end is None: return self._corpus()
        op = DatetimeOp(meta, start, end, strftime)
        return self._corpus().cloned(op.mask())

    def group_by(self, id_, grouper: pd.Grouper = None) -> Iterator[tuple[str, 'DataFrameCorpus']]:
        """ Return groups of the subcorpus based on their metadata.

        :arg grouper: pd.Grouper - as you would in pandas
        :return tuple[groupid, subcorpus]
        """
        meta = self._corpus().get_meta(id_)
        return ((gid, self._corpus().cloned(mask)) for gid, mask in meta.groupby(grouper=grouper))

    def _mask_by_condition(self, meta, cond_func):
        mask = meta.apply(cond_func)
        try:
            mask = mask.astype('boolean')
        except TypeError:
            raise TypeError("Does your condition function return booleans?")
        return mask
