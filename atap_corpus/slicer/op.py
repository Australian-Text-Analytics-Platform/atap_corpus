""" Operation

A behaviour class that encompasses the slicing operation.
"""
import re
from datetime import datetime
from typing import Union, IO, Callable
from abc import abstractmethod

import pandas as pd
from spacy.matcher import Matcher
from spacy.util import filter_spans

from atap_corpus.corpus.base import BaseCorpus
from atap_corpus.interfaces import Serialisable, Filterable
from atap_corpus.mixins import SpacyDocsMixin
from atap_corpus._types import PathLike, TSerialisable, TCorpus

import colorlog

logger = colorlog.getLogger(__name__)

__all__ = ["CallableOp", "ItemOp", "RangeOp", "RegexOp", "DatetimeOp", "MatcherOp"]


class BaseOperation(Serialisable):

    @classmethod
    def deserialise(cls, path_or_file: PathLike | IO) -> TSerialisable:
        file = super().deserialise(path_or_file)
        raise NotImplementedError()

    def serialise(self, path_or_file: PathLike | IO, *args, **kwargs) -> PathLike | IO:
        file = super().serialise(path_or_file)
        raise NotImplementedError()

    def __init__(self, filterable: Filterable, *args, **kwargs):
        if not isinstance(filterable, Filterable):
            raise TypeError(f"{filterable} is not a {Filterable.__class__.__name__}.")
        self._filterable = filterable

    @abstractmethod
    def condition_func(self, any_) -> bool:
        """ This method is used in df.apply() and should return a boolean to create a mask. """
        raise NotImplementedError()

    def mask(self):
        """ Returns the mask of the corpus after slicing. """
        mask = self._filterable.apply(self.condition_func)
        try:
            mask = mask.astype('boolean')
        except TypeError:
            raise TypeError("Does your condition function return booleans?")
        return mask

    def __str__(self):
        return f"<{self.__class__.__name__}>"


class CallableOp(BaseOperation):

    def __init__(self, filterable: Filterable, condition_fn: Callable):
        super().__init__(filterable)
        self._callable = condition_fn

    def condition_func(self, any_) -> bool:
        return self._callable(any_)


class ItemOp(BaseOperation):
    def __init__(self, filterable: Filterable, items: str | list | tuple | set):
        super().__init__(filterable)
        items = [items] if isinstance(items, str) else items
        items = [items] if not type(items) in (list, tuple, set) else items
        items = set(items)
        self.items = items

    def condition_func(self, any_):
        items = self.items
        if isinstance(any_, str):
            return any_ in items
        elif isinstance(any_, int) or isinstance(any_, float):
            return any_ in items
        elif isinstance(any_, dict):
            return not set(any_.keys()).isdisjoint(items)
        elif type(any_) in (list, tuple, set):
            return not set(any_).isdisjoint(items)
        else:
            raise TypeError(f"Unable to filter {type(any_)}. Only string or iterables.")

    def __str__(self):
        s = super().__str__()
        return s[:-1] + f" items: {', '.join(str(item) for item in self.items)}>"


class RangeOp(BaseOperation):

    def __init__(self, filterable: Filterable, min_: Union[int, float], max_: Union[int, float]):
        """ Range Operation
        :param min_: Inclusive minimum
        :param max_: Exclusive maximum
        """
        super().__init__(filterable)
        self.min_ = min_
        self.max_ = max_

    def condition_func(self, any_) -> bool:
        min_, max_ = self.min_, self.max_
        if min_ is None and max_ is None: return True
        if None not in (min_, max_):
            return min_ <= any_ <= max_
        elif min_ is not None:
            return min_ <= any_
        else:
            return any_ <= max_

    def __str__(self):
        s = super().__str__()
        return s[:-1] + f" min {self.min_} max {self.max_}>"


class RegexOp(BaseOperation):
    def __init__(self, filterable: Filterable, regex: str, ignore_case: bool = True):
        super().__init__(filterable)
        self.regex = regex
        self.ignore_case = ignore_case
        self._flags = 0 if not ignore_case else re.IGNORECASE
        self.pattern = re.compile(regex, flags=self._flags)

    def condition_func(self, any_) -> bool:
        return self.pattern.search(str(any_)) is not None

    def __str__(self):
        s = super().__str__()
        return s[:-1] + f" {self.regex}\tcased: {not self.ignore_case}>"


class DatetimeOp(BaseOperation):
    def __init__(self, filterable: Filterable, start: str | datetime, end: str | datetime, strftime: str = None):
        super().__init__(filterable)
        self.strftime = strftime
        self.start = pd.to_datetime(start, dayfirst=True, format=self.strftime)
        self.end = pd.to_datetime(end, dayfirst=True, format=self.strftime)

        if self.start is not None:
            logger.debug(f"{'Converted start datetime'.ljust(25)}: {self.start.strftime('%Yy %mm %dd %H:%M:%S')}")
        if self.end is not None:
            logger.debug(f"{'Converted end datetime'.ljust(25)}: {self.end.strftime('%Yy %mm %dd %H:%M:%S')}")

    def condition_func(self, any_) -> bool:
        start, end = self.start, self.end
        if None not in (start, end):
            return start <= any_ < end
        elif start is not None:
            return start <= any_
        elif end is not None:
            return any_ < end
        else:
            return True

    def __str__(self):
        s = super().__str__()
        return s[:-1] + f" start {self.start} end {self.end}>"


# -- Only for TCorpus with SpacyDocMixin

class MatcherOp(BaseOperation):
    def __init__(self, corpus: TCorpus, matcher: Matcher, min_: int = 1, max_: int = None):
        if not isinstance(corpus, SpacyDocsMixin):
            raise TypeError(f"{corpus} does not inherit from {SpacyDocsMixin.__class__.__name__}.")
        if not isinstance(corpus, BaseCorpus):
            raise TypeError(f"{corpus} must be a {BaseCorpus.__class__.__name__}.")
        super().__init__(corpus.docs())
        self.matcher = matcher
        if min_ is None: min_ = 1
        self.min_, self.max_ = min_, max_

        self._matched = []

    def condition_func(self, doc) -> bool:
        min_, max_ = self.min_, self.max_

        matched_spans = [doc[s:e] for _, s, e in self.matcher(doc)]
        matched_spans = filter_spans(matched_spans)
        matched_spans_str = ', '.join([span.text for span in matched_spans]) if len(matched_spans) > 0 else ''
        self._matched.append(matched_spans_str)
        matched = len(matched_spans)
        if max_ is not None:
            return min_ <= matched < max_
        else:
            return min_ <= matched

    def retrieve_matched(self):
        return self._matched
