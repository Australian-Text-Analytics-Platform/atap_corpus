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
import string
from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd

from test_dfcorpus import MockDataFrameCorpus, data

metas: dict[str, pd.Series | list | tuple] = {
    'meta_int': np.arange(len(data)),
    'meta_str': [string.ascii_uppercase[i % len(string.ascii_uppercase)] for i in range(len(data))],
    'meta_dt': [(datetime.now() + timedelta(days=i)) for i in range(len(data))],
}


class TestCorpusSlicer(TestCase):
    def setUp(self):
        self.root = MockDataFrameCorpus(docs=data)
        for name, metadata in metas.items():
            self.root.add_meta(metadata, name)

    def test_given_dfcorpus_when_filter_by_condition_then_filtered_correctly(self):
        name = 'meta_int'
        child: MockDataFrameCorpus = self.root.slicer.filter_by_condition(name, lambda i: i > (len(self.root) / 2))
        self.assertEqual(len(child), (metas.get(name) > len(self.root) / 2).sum())
        meta = child.get_meta(name)
        idx = 0
        for metadatum, m in zip(metas.get(name), (metas.get(name) > len(self.root) / 2)):
            if m:
                self.assertEqual(meta.iloc[idx], metadatum, f"Incorrect value at index={idx} for the child meta.")
                idx += 1

    def test_given_dfcorpus_when_filter_by_item_then_filtered_correctly(self):
        pass

    def test_given_dfcorpus_when_filter_by_range_then_filtered_correctly(self):
        pass

    def test_given_dfcorpus_when_filter_by_regex_then_filtered_correctly(self):
        pass

    def test_given_dfcorpus_when_filter_by_datetime_then_filtered_correctly(self):
        pass

    def test_given_dfcorpus_when_groupby_then_grouped_correctly(self):
        pass
