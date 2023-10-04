""" Corpus tests

Related:
+ corpus.corpus.py

This test suite ensures core behaviours of the DataFrameCorpus is behaving correctly.
Namely, these are:
read behaviours - get, len, iter
clone behaviours - clone, detach, parent, root
"""

from unittest import TestCase

import pandas as pd

from atap_corpus.corpus.corpus import DataFrameCorpus


class MockDataFrameCorpus(DataFrameCorpus):
    def __init__(self):
        super().__init__(pd.Series(['a', 'b', 'c']), name=None)


class TestCorpus(TestCase):
    pass
