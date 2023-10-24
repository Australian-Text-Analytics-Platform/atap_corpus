""" DataFrameCorpus tests

Related:
+ corpus.corpus.py

This test suite ensures core behaviours of the DataFrameCorpus is behaving correctly.
Namely, these are:
+ read behaviours - get, len, iter
+ clone behaviours - clone, parent, root, detached      (also for meta and dtms)
"""
import logging.config
import os
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd

from atap_corpus.corpus.corpus import DataFrameCorpus

data = pd.Series(['a', 'b', 'c'])
test_parent_mask = pd.Series([0, 1, 1], dtype=bool)
test_child_mask = pd.Series([0, 1, 0], dtype=bool)
test_parent2_mask = pd.Series([1, 0, 0], dtype=bool)


class MockDataFrameCorpus(DataFrameCorpus):
    def __init__(self, docs=None, name=None):
        super().__init__(docs, name=name)  # generates unique name.


def compare(testcase: TestCase, root: MockDataFrameCorpus, child: MockDataFrameCorpus, mask: pd.Series):
    assert mask.dtype == bool, "Mask is not a boolean."
    assert root.is_root, "arg: root must be your root corpus. Try .find_root()"
    child_idx = 0
    child_docs = child.docs()
    for root_idx, (m, d) in enumerate(zip(mask, root.docs())):
        if m:
            testcase.assertEqual(d, child_docs.iloc[child_idx], f"Invalid doc at {root_idx=} {child_idx=}")
            child_idx += 1


class TestDataFrameCorpus(TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.ERROR)

    def setUp(self):
        self.root = MockDataFrameCorpus(docs=data)

        # used to test corpus dtms
        self.tokeniser_func = lambda doc: doc.split()

    def test_create_empty_dfcorpus(self):
        empty = DataFrameCorpus()
        self.assertEqual(len(empty), 0)
        with self.assertRaises(IndexError):
            _ = empty[0]
            _ = empty[0: 10]

    def test_given_dfcorpus_of_size_when_index_larger_than_size_then_raise_IndexError(self):
        size = len(self.root)
        with self.assertRaises(IndexError):
            self.root[size + 1]

    def test_given_dfcorpus_when_negative_index_then_raise_indexError(self):
        with self.assertRaises(IndexError):
            self.root[-1]

    # clonable - tree behaviour of corpus.
    def test_given_dfcorpus_when_get_int_returns_correct_doc(self):
        get_int, get_int_label = self.root[0], data.iloc[0]
        self.assertEqual(get_int, get_int_label, f"Expecting {get_int_label}. Got {get_int}.")

    def test_given_dfcorpus_when_get_slice_returns_correct_doc_slice(self):
        get_slice, get_slice_label = self.root[0:2], data.iloc[0:2]
        self.assertTrue(get_slice.equals(get_slice_label), f"Expecting {get_slice_label}. Got {get_slice}.")

    def test_given_dfcorpus_when_len_returns_correct_len(self):
        len_root, len_data = len(self.root), len(data)
        self.assertEqual(len_root, len_data, f"Expecting {len_data}. Got {len_root}.")

    def test_given_dfcorpus_when_iter_returns_all_correct_docs(self):
        iter_root = iter(self.root)
        i = -1
        for i, doc in enumerate(iter_root):
            self.assertEqual(doc, data.iloc[i], f"Incorrect doc returned at idx={i}.")
        self.assertEqual(i + 1, len(data), f"Incorrect number of docs returned.")

    def test_given_dfcorpus_when_cloned_returns_correct_subcorpus_len(self):
        parent = self.root.cloned(test_parent_mask)
        self.assertEqual(len(parent), sum(test_parent_mask),
                         f"Expecting size {sum(test_parent_mask)}. Got {len(parent)}.")

        child = parent.cloned(test_child_mask)
        self.assertEqual(len(child), sum(test_child_mask),
                         f"Expecting size {sum(test_child_mask)}. Got {len(child)}.")

    def test_given_dfcorpus_when_cloned_returns_correct_subcorpus_docs(self):
        parent = self.root.cloned(test_parent_mask)
        compare(self, self.root, parent, test_parent_mask)
        child = parent.cloned(test_child_mask)
        compare(self, parent.find_root(), child, test_child_mask)

    def test_given_dfcorpus_when_cloned_then_parent_child_refs_are_valid(self):
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

    def test_given_dfcorpus_when_detached_then_detached_is_root_and_tree_structure_kept(self):
        parent: DataFrameCorpus = self.root.cloned(test_parent_mask)
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

    # meta data
    def test_given_dfcorpus_when_add_and_get_meta_then_correct_meta_is_returned(self):
        meta = pd.Series(np.arange(len(self.root)), name='meta')
        self.root.add_meta(meta)
        self.assertTrue(meta.name in self.root.metas, f"{meta.name} not in Corpus's metas.")
        got_meta = self.root.get_meta(meta.name)
        self.assertTrue(got_meta.equals(meta), f"Gotten meta did not match added meta.")

    def test_given_dfcorpus_when_add_and_remove_meta_then_meta_is_removed(self):
        meta = pd.Series(np.arange(len(self.root)), name='meta')
        self.root.add_meta(meta)
        self.assertTrue(meta.name in self.root.metas, f"{meta.name} not in Corpus's metas.")
        self.root.remove_meta(meta.name)
        self.assertTrue(meta.name not in self.root.metas, f"{meta.name} still in Corpus after removal.")

    def test_given_dfcorpus_with_meta_when_getitem_then_correct_meta_is_returned(self):
        # this is effectively the same as get_meta() but with syntactic sugar of __getitem__
        # e.g. corpus['meta']
        meta = pd.Series(np.arange(len(self.root)), name='meta')
        self.root.add_meta(meta)
        self.assertTrue(meta.name in self.root.metas, f"{meta.name} not in Corpus's metas.")
        got_meta = self.root['meta']
        self.assertTrue(got_meta.equals(meta), f"Gotten meta did not match added meta.")

    def test_given_dfcorpus_when_add_dtm_then_dtm_is_added(self):
        self.root.add_dtm(self.tokeniser_func, name='tokens')
        self.assertTrue(self.root.dtms['tokens'] is not None, "Missing tokens DTM after adding.")

    def test_given_dfcorpus_when_cloned_and_add_dtm_then_dtm_is_added_to_root_and_clones(self):
        parent = self.root.cloned(test_parent_mask)
        parent.add_dtm(self.tokeniser_func, name='tokens')
        self.assertTrue(self.root.dtms['tokens'] is not None, "Missing tokens DTM in root after adding in parent.")
        self.assertTrue(parent.dtms['tokens'] is not None, "Missing tokens DTM in parent after adding in parent.")
        child = parent.cloned(test_child_mask)
        child.add_dtm(self.tokeniser_func, name='tokens2')
        self.assertTrue(self.root.dtms['tokens2'] is not None, "Missing tokens2 DTM in root after adding in child.")
        self.assertTrue(child.dtms['tokens2'] is not None, "Missing tokens2 DTM in child after adding in child.")

    def test_given_dfcorpus_with_dtm_and_cloned_then_child_dtms_are_correct(self):
        self.root.add_dtm(self.tokeniser_func, name='tokens')
        parent = self.root.cloned(test_parent_mask)
        self.assertEqual(parent.dtms['tokens'].shape[0], len(parent),
                         "Mismatched parent DTM number of docs and parent corpus")
        child = parent.cloned(test_child_mask)
        self.assertEqual(child.dtms['tokens'].shape[0], len(child),
                         "Mismatched child DTM number of docs and child corpus")

    def test_given_dfcorpus_and_cloned_and_add_dtm_then_child_dtms_are_correct(self):
        parent = self.root.cloned(test_parent_mask)
        parent.add_dtm(self.tokeniser_func, name='tokens')
        self.assertEqual(parent.dtms['tokens'].shape[0], len(parent),
                         "Mismatched parent DTM number of docs and parent corpus")
        child = parent.cloned(test_child_mask)
        self.assertEqual(child.dtms['tokens'].shape[0], len(child),
                         "Mismatched child DTM number of docs and child corpus")

    # -- serialisation
    def test_given_dfcorpus_without_dtm_and_serialised_then_deserialised_rebuilds_equivalent_dfcorpus(self):
        path = Path(tempfile.mktemp(suffix=".zip"))
        self.root.serialise(path_or_file=path)
        self.assertTrue(path.is_file(), "Did not serialise corpus to file.")
        deserialised = DataFrameCorpus.deserialise(path)
        self.assertTrue(self.root.equals(deserialised))
        os.remove(path)

    def test_given_dfcorpus_with_dtm_and_serialised_then_deserialised_rebuilds_equivalent_dfcorpus(self):
        path = Path(tempfile.mktemp(suffix=".zip"))
        self.root.add_dtm(tokeniser_func=self.tokeniser_func, name='tokens')
        self.root.serialise(path_or_file=path)
        self.assertTrue(path.is_file(), "Did not serialise corpus to file.")
        deserialised = DataFrameCorpus.deserialise(path)
        os.remove(path)
        self.assertTrue(self.root.equals(deserialised))

    # -- sampling
    def test_given_dfcorpus_and_sample_then_sample_size_is_correct(self):
        n0, n1 = 2, 1
        sampled_0 = self.root.sample(n0)
        self.assertEqual(len(sampled_0), n0, "Sampling did not sample the correct number of samples.")
        sampled_1 = sampled_0.sample(n1)
        self.assertEqual(len(sampled_1), n1, "Sampling did not sample the correct number of samples.")

    # -- join
    def test_given_two_subdfcorpus_of_same_subtree_and_joined_then_joined_is_correct(self):
        parent = self.root.cloned(test_parent_mask)
        child = parent.cloned(test_child_mask)
        joined = parent.join(child)
        self.assertEqual(len(joined), len(parent), "Should be the size of the first common parent.")
        self.assertEqual(joined.parent, parent, f"Expecting common parent {parent}. Got {joined.parent}.")

    def test_given_two_subdfcorpus_of_diff_subtree_and_joined_then_joined_is_correct(self):
        parent0 = self.root.cloned(test_parent_mask)
        parent1 = self.root.cloned(test_parent2_mask)
        or_size = (test_parent_mask | test_parent2_mask).sum()
        joined = parent0.join(parent1)
        self.assertEqual(len(joined), or_size, f"Incorrect size after join. Expecting {or_size}. Got {len(joined)}")
        self.assertEqual(joined.parent, self.root, f"Expecting common parent {self.root}. Got {joined.parent}.")
