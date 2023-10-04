""" Test Global Corpora

This Test Suite tests for:
1. All Corpus are added into the GlobalCorpora.
"""

from unittest import TestCase

from test_corpus import MockDataFrameCorpus
from atap_corpus.registry import _Global_Corpora


class TestGlobalCorpora(TestCase):
    def test_given_subclass_dfcorpus_then_corpus_is_added_to_global_corpora(self):
        assert len(_Global_Corpora) == 0, "Global Corpora did not start empty."
        dfcorpus = MockDataFrameCorpus()
        assert len(_Global_Corpora) == 1, f"{dfcorpus} was not added to GlobalCorpora."

        gc_dfcorpus = _Global_Corpora.get(dfcorpus.name)
        assert gc_dfcorpus is not None, f"Expecting {dfcorpus} in GlobalCorpora. Got {gc_dfcorpus}."
