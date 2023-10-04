from unittest import TestCase

from atap_corpus.corpus.mixins import UniqueNameProviderMixin


class TestRegistry(TestCase):
    def test_unique_name_provider(self):
        """ Tests global registry unique name conforms to UniqueNameProvider protocol. """
        from atap_corpus.registry import _Unique_Name_Provider
        assert isinstance(_Unique_Name_Provider, UniqueNameProviderMixin), \
            f"_Unique_Name_Provider does have {UniqueNameProviderMixin.__name__}."
