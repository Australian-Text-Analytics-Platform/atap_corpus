from abc import ABCMeta

from atap_corpus.interfaces import Clonable, Serialisable


class BaseCorpus(metaclass=ABCMeta, Clonable, Serialisable):
    """ Base Corpus

    All Corpus types should inherit from this class.
    """
    pass
