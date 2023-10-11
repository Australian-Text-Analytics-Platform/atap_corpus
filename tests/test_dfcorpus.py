""" DataFrameCorpus tests

Related:
+ corpus.corpus.py

This test suite ensures core behaviours of the DataFrameCorpus is behaving correctly.
Namely, these are:
+ read behaviours - get, len, iter
+ clone behaviours - clone, parent, root, detached
"""

from unittest import TestCase

import numpy as np
import pandas as pd

from atap_corpus.corpus.corpus import DataFrameCorpus

data = pd.Series(['a', 'b', 'c'])
test_parent_mask = pd.Series([0, 1, 1], dtype=bool)
test_child_mask = pd.Series([0, 1, 0], dtype=bool)


class MockDataFrameCorpus(DataFrameCorpus):
    def __init__(self, docs=None, name=None):
        if docs is None: docs = data
        super().__init__(docs, name=name)  # generates unique name.


class TestDataFrameCorpus(TestCase):
    def setUp(self):
        self.root = MockDataFrameCorpus()

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

    def test_given_dfcorpus_when_cloned_returns_correct_subcorpus(self):
        parent = self.root.cloned(test_parent_mask)
        self.assertEqual(len(parent), sum(test_parent_mask),
                         f"Expecting size {sum(test_parent_mask)}. Got {len(parent)}.")
        child = parent.cloned(test_child_mask)
        self.assertEqual(len(child), sum(test_child_mask),
                         f"Expecting size {sum(test_child_mask)}. Got {len(child)}.")

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
