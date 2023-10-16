""" DTM tests

Related:
+ parts.dtm.py

This test suite ensures core behaviours of the DTM class is behaving correctly.
Namely, these are:
+ clone behaviours - clone, root, detached
"""
import os
from pathlib import Path
from unittest import TestCase
import pandas as pd
import tempfile

from atap_corpus.parts.dtm import DTM

# generated from ChatGPT
docs = pd.Series(['The sun set over the horizon, painting the sky in shades of orange and pink.',
                  'She opened the mysterious book, eager to discover the secrets hidden within its pages.',
                  'As the clock struck midnight, the city came alive with the sounds of celebration.'])
test_parent_mask = pd.Series([0, 1, 1], dtype=bool)
test_child_mask = pd.Series([0, 1, 0], dtype=bool)


class TestDTM(TestCase):
    def setUp(self):
        self.root = DTM.from_docs(docs, lambda doc: doc.split())

    def test_given_docs_when_from_docs_then_dtm_is_correct(self):
        tokeniser_func = lambda doc: doc.split()
        dtm = DTM.from_docs(docs, tokeniser_func)
        self.assertEqual(len(docs), dtm.shape[0], "Number of docs in DTM does not match number of given docs.")
        for i, doc in enumerate(docs):
            terms = tokeniser_func(doc)
            self.assertEqual(len(terms), dtm.matrix[i, :].sum(), f"Number of terms in in doc idx {i} is incorrect.")

    def test_given_dtm_when_clone_then_cloned_is_correct(self):
        parent = self.root.cloned(test_parent_mask)
        self.assertEqual(parent.shape[0], sum(test_parent_mask),
                         f"Expecting {sum(test_parent_mask)} rows. Got {parent.shape[0]}.")
        self.assertEqual(parent.shape[1], self.root.shape[1],
                         "Term columns should be the same between clone and root.")
        child = parent.cloned(test_child_mask)
        self.assertEqual(child.shape[0], sum(test_child_mask),
                         f"Expecting {sum(test_child_mask)} rows. Got {child.shape[0]}.")
        self.assertEqual(child.shape[1], self.root.shape[1],
                         "Term columns should be the same between clone and root.")

    def test_given_dtm_when_clone_then_parent_child_refs_are_valid(self):
        parent = self.root.cloned(test_parent_mask)
        self.assertTrue(parent.parent is self.root,
                        f"Expecting parent as {id(self.root)}. Got {id(parent.parent)}")
        self.assertTrue(parent.find_root() is self.root,
                        f"Expecting root is {id(self.root)}. Got {id(parent.find_root())}")
        child = parent.cloned(test_child_mask)
        self.assertTrue(child.parent is parent,
                        f"Expecting parent as {id(parent)}. Got {id(child.parent)}")
        self.assertTrue(child.find_root() is self.root,
                        f"Expecting root is {id(self.root)}. Got {id(child.find_root())}")

    def test_given_dtm_when_detached_then_detached_is_correct(self):
        parent = self.root.cloned(test_parent_mask)
        detached = parent.detached()
        self.assertEqual(detached.total, parent.total,
                         "Detached DTM matrix is not the same as the original.")
        self.assertEqual(detached.terms, parent.terms,
                         "Detached DTM terms is not the same as the original.")

    def test_given_dtm_when_detached_then_detached_is_root_and_tree_structure_kept(self):
        parent = self.root.cloned(test_parent_mask)
        detached = parent.detached()
        self.assertTrue(detached.find_root() is detached,
                        f"Expecting root is {id(self.root)}. Got {id(parent.find_root())}")
        # assert tree structure is kept.
        self.assertTrue(parent.parent is self.root,
                        f"Expecting parent as {id(self.root)}. Got {id(parent.parent)}")
        self.assertTrue(parent.find_root() is self.root,
                        f"Expecting root is {id(self.root)}. Got {id(parent.find_root())}")

        child = parent.cloned(test_child_mask)
        self.assertTrue(detached.find_root() is detached,
                        f"Expecting root is {id(self.root)}. Got {id(child.find_root())}")
        # assert tree structure is kept.
        self.assertTrue(child.parent is parent,
                        f"Expecting parent as {id(parent)}. Got {id(child.parent)}")
        self.assertTrue(child.find_root() is self.root,
                        f"Expecting root is {id(self.root)}. Got {id(child.find_root())}")

    # -- serialisation
    def test_given_dtm_when_serialise_then_deserialise_rebuilds_equivalent_dtm(self):
        path = Path(tempfile.mktemp(suffix=".zip"))
        self.root.serialise(path_or_file=path)
        self.assertTrue(path.is_file(), "Failed to serialise dtm to file.")
        deserialised = DTM.deserialise(path)
        self.assertEqual(self.root, deserialised)
        os.remove(path)
