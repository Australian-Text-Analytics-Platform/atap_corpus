""" Registry

Registry holds global behaviours used in the framework.
"""

from atap_corpus.corpus.corpora import _GlobalCorpora

# must inherit UniqueNameProviderMixin.
_Unique_Name_Provider = _GlobalCorpora()  # singleton
_Global_Corpora = _GlobalCorpora()  # singleton
