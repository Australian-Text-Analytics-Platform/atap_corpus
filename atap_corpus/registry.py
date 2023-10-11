""" Registry

Registry defines global behaviours used in the framework.

_Unique_ID_Provider: provide unique IDs for each BaseCorpus for use in the GlobalCorpora.
_Unique_Name_Provider: provide unique names for each BaseCorpus for use in the GlobalCorpora.
_Global_Corpora: the container object that holds all the BaseCorpus references.

These are all implemented by the GlobalCorpora singleton right now.
When necessary, these can be changed and decoupled
"""

from atap_corpus.corpus.corpora import _GlobalCorpora

_Unique_ID_Provider = _GlobalCorpora()  # singleton - must inherit UniqueIDProviderMixin.
_Unique_Name_Provider = _GlobalCorpora()  # singleton - must inherit UniqueNameProviderMixin.
_Global_Corpora = _GlobalCorpora()  # singleton
