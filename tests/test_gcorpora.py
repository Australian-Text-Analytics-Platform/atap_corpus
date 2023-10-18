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

from test_dfcorpus import MockDataFrameCorpus, data
from atap_corpus.registry import _Global_Corpora


class TestGlobalCorpora(TestCase):

    def test_given_subclass_dfcorpus_then_corpus_is_added_to_global_corpora(self):
        gcorpora_len = len(_Global_Corpora)
        dfcorpus = MockDataFrameCorpus(docs=data)
        assert len(_Global_Corpora) == gcorpora_len + 1, f"{dfcorpus} was not added to GlobalCorpora."

    def test_given_subclass_dfcorpus_ref_is_deleted_then_gcorpora_removes_it(self):
        gcorpora_len = len(_Global_Corpora)
        dfcorpus = MockDataFrameCorpus(docs=data)
        assert len(_Global_Corpora) == gcorpora_len + 1, f"{dfcorpus} was not added to GlobalCorpora."

        del dfcorpus
        assert len(_Global_Corpora) == gcorpora_len, "Expecting dfcorpus to be removed but it wasn't."
