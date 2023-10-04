from unittest import TestCase

from atap_corpus.corpus.mixins import UniqueNameProviderMixin


class TestRegistry(TestCase):
    def test_unique_name_provider(self):
        """ Tests global registry unique name conforms to UniqueNameProvider protocol. """
        from atap_corpus.registry import _Unique_Name_Provider
        assert isinstance(_Unique_Name_Provider, UniqueNameProviderMixin), \
            f"_Unique_Name_Provider does have {UniqueNameProviderMixin.__name__}."

    def test_global_corpora(self):
        from atap_corpus.registry import _Global_Corpora
        from atap_corpus.corpus.corpora import _GlobalCorpora
        assert isinstance(_Global_Corpora, _GlobalCorpora), f"Only supports GlobalCorpora at the moment."
