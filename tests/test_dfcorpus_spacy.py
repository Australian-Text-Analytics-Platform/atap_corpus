""" DataFrameCorpus SpacyDocMixin tests

Related:
+ corpus.corpus.py
+ corpus.mixins.py

This test suite ensures core behaviours of SpacyDocMixin in DataFrameCorpus is working.
Namely, these are:
+ run_spacy - process, reprocess.
+ uses_spacy - flag is correct.
"""

from unittest import TestCase

import spacy
import pandas as pd

from test_dfcorpus import MockDataFrameCorpus, data, test_child_mask


class TestDataFrameCorpusSpacyDocMixin(TestCase):
    def setUp(self):
        self.nlp = spacy.blank('en')
        spacy_data = pd.Series(self.nlp.pipe(['a', 'b', 'c']))
        self.root = MockDataFrameCorpus(docs=data)
        self.root_spacy = MockDataFrameCorpus(docs=spacy_data)

    def test_given_spacy_dfcorpus_then_uses_spacy_is_true(self):
        self.assertTrue(self.root_spacy.uses_spacy(), "It is using spacy docs. .uses_spacy() returned False.")

    def test_given_root_dfcorpus_when_run_spacy_then_doc_are_spacy(self):
        self.root.run_spacy(nlp=self.nlp, progress_bar=False)
        self.assertTrue(self.root.uses_spacy(), f"Expected Corpus to hold spacy docs. Got {type(self.root[0])}.")

    def test_given_child_dfcorpus_when_run_spacy_then_root_docs_are_spacy(self):
        child: MockDataFrameCorpus = self.root.cloned(test_child_mask)
        child.run_spacy(self.nlp, progress_bar=False)
        self.assertTrue(self.root.uses_spacy(),
                        f"Expected root Corpus to hold spacy docs. But {self.root.uses_spacy()=}")
        self.assertTrue(child.uses_spacy(),
                        f"Expected child Corpus to hold spacy docs. But {child.uses_spacy()=}")

    def test_given_dfcorpus_when_run_spacy_reprocessed_then_dfcorpus_can_be_reprocessed(self):
        from atap_corpus.corpus.corpus import logger
        import logging
        level = logger.level
        logger.setLevel(logging.ERROR)  # avoids WARNING message from reprocessing.
        self.root.run_spacy(self.nlp, progress_bar=False)
        self.root.run_spacy(self.nlp, progress_bar=False, reprocess_prompt=False)
        # no assertions for what this test is testing, if no exceptions is raised, it's passed.
        self.assertTrue(self.root.uses_spacy(), "Spacy reprocessed Corpus should hold spacy docs.")
        logger.setLevel(level)
