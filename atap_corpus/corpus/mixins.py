from abc import abstractmethod

import coolname


class UniqueNameProviderMixin(object):
    """ Provides unique name generator functions as a Mixin. """

    @abstractmethod
    def is_unique_name(self, name: str) -> bool:
        raise NotImplementedError()

    def unique_name(self) -> str:
        while name := coolname.generate_slug(2):
            if not self.is_unique_name(name):
                return name
