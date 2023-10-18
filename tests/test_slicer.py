""" CorpusSlicer tests

Related:
+ corpus.corpus.py
+ slicer.slicer.py

This test suite ensures core behaviours of the CorpusSlicer behaves correctly.
Namely, these are:
+ filter_by_condition
+ filter_by_item
+ filter_by_range
+ filter_by_regex
+ filter_by_datetime
+ group_by -> skipped right now.
"""
import re
import string
from typing import Iterable
from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd

from test_dfcorpus import MockDataFrameCorpus, data

_now = datetime.now()
metas: dict[str, pd.Series | list | tuple] = {
    'meta_int': np.arange(len(data)),
    'meta_str': [string.ascii_uppercase[i % len(string.ascii_uppercase)] for i in range(len(data))],
    'meta_dt': [(_now + timedelta(days=i)) for i in range(len(data))],
}


def dfcorpus_compare(testcase: TestCase, root_meta, child_meta: pd.Series, mask: Iterable[bool]):
    """ You can only use this function if the meta came from a dataframe corpus since meta must be a series."""
    mask = pd.Series(mask, dtype=bool)
    idx = 0
    for metadatum, m in zip(root_meta, mask):
        if m:
            testcase.assertEqual(child_meta.iloc[idx], metadatum, f"Incorrect value at index={idx} given mask.")
            idx += 1


class TestCorpusSlicer(TestCase):
    def setUp(self):
        self.root = MockDataFrameCorpus(docs=data)
        for name, metadata in metas.items():
            self.root.add_meta(metadata, name)

    def test_given_dfcorpus_when_filter_by_condition_then_filtered_correctly(self):
        name = 'meta_int'
        mask = metas.get(name) >= len(self.root) / 2
        child: MockDataFrameCorpus = self.root.slicer.filter_by_condition(name, lambda i: i >= (len(self.root) / 2))
        self.assertEqual(len(child), mask.sum(),
                         "Corpus size after filtering is incorrect.")
        dfcorpus_compare(self, metas.get(name), child.get_meta(name), mask)

    def test_given_dfcorpus_when_filter_by_item_then_filtered_correctly(self):
        name = 'meta_str'
        items = ("A", "B")
        mask = [True if s in items else False for s in metas.get(name)]
        child: MockDataFrameCorpus = self.root.slicer.filter_by_item(name, items=items)
        self.assertEqual(len(child), len(items), "Corpus size after filtering is incorrect.")
        dfcorpus_compare(self, metas.get(name), child.get_meta(name), mask)

    def test_given_dfcorpus_when_filter_by_range_then_filtered_correctly(self):
        name = 'meta_int'
        max_ = int(len(self.root) / 2)
        mask = metas.get(name) >= max_
        child: MockDataFrameCorpus = self.root.slicer.filter_by_range(name, min_=max_)
        self.assertEqual(len(child), mask.sum(), "Corpus size after filtering is incorrect.")
        dfcorpus_compare(self, metas.get(name), child.get_meta(name), mask)

    def test_given_dfcorpus_when_filter_by_regex_then_filtered_correctly(self):
        name = 'meta_str'
        re_ptn = re.compile(r'[ABCDEFG]')
        mask = np.array([True if re_ptn.match(s) is not None else False for s in metas.get(name)])
        child: MockDataFrameCorpus = self.root.slicer.filter_by_regex(name, re_ptn.pattern, False)
        self.assertEqual(len(child), mask.sum(), "Corpus size after filtering is incorrect.")
        dfcorpus_compare(self, metas.get(name), child.get_meta(name), mask)

    def test_given_dfcorpus_when_filter_by_datetime_then_filtered_correctly(self):
        name = 'meta_dt'
        deltas = 2
        mask = [True] * deltas
        mask += [False] * (len(self.root) - deltas)
        mask = np.array(mask)
        child: MockDataFrameCorpus = self.root.slicer.filter_by_datetime(name,
                                                                         start=_now,
                                                                         end=_now + timedelta(deltas),
                                                                         strftime=None)
        self.assertEqual(len(child), mask.sum(), "Corpus size after filtering is incorrect.")
        dfcorpus_compare(self, metas.get(name), child.get_meta(name), mask)

    def test_given_dfcorpus_when_groupby_then_grouped_correctly(self):
        # todo:
        pass
