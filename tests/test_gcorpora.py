""" Test Global Corpora

Related:
+ corpus.corpora.py

GlobalCorpora is a runtime feature of the framework.
It keeps track of all the BaseCorpus instances that are created so that we have a
central management object.

This test suite ensures core global corpora behaviours.
1. All Corpus are added into the GlobalCorpora.
2. When a Corpus reference is dropped, it is also dropped from GlobalCorpora.
"""

from unittest import TestCase

from test_dfcorpus import MockDataFrameCorpus
from atap_corpus.registry import _Global_Corpora


class TestGlobalCorpora(TestCase):
    def setUp(self):
        assert len(_Global_Corpora) == 0, "Global Corpora did not start empty."

    def test_given_subclass_dfcorpus_then_corpus_is_added_to_global_corpora(self):
        dfcorpus = MockDataFrameCorpus()
        assert len(_Global_Corpora) == 1, f"{dfcorpus} was not added to GlobalCorpora."

        gc_dfcorpus = _Global_Corpora.get(dfcorpus.name)
        assert gc_dfcorpus is not None, f"Expecting {dfcorpus} in GlobalCorpora. Got {gc_dfcorpus}."

    def test_given_subclass_dfcorpus_ref_is_deleted_then_gcorpora_removes_it(self):
        dfcorpus = MockDataFrameCorpus()
        assert len(_Global_Corpora) == 1, f"{dfcorpus} was not added to GlobalCorpora."

        del dfcorpus
        assert len(_Global_Corpora) == 0, "Expecting dfcorpus to be removed but it wasn't."
