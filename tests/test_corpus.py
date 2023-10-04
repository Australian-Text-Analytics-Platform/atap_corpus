""" Corpus tests

clonable:
1. corpus cloned returns a corpus of size mask.
name: names are unique and on default generated. -> this requires global corpora.
2. create

3. serialisation:
    + serialise and deserialise and ensure same corpus is built.
"""

from unittest import TestCase

import pandas as pd

from atap_corpus.corpus.corpus import DataFrameCorpus


class MockDataFrameCorpus(DataFrameCorpus):
    def __init__(self):
        super().__init__(pd.Series(['a', 'b', 'c']), name=None)


class TestCorpus(TestCase):
    pass
