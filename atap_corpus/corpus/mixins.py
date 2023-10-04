import uuid
from abc import abstractmethod

import coolname
import logging

logger = logging.getLogger(__name__)

_NAME_LEN = 2
_MAX_COMBINATIONS = coolname.get_combinations_count(_NAME_LEN)
_CURRENT_COUNT = 0


class UniqueNameProviderMixin(object):
    """ Provides unique name generator functions as a Mixin. """

    @abstractmethod
    def is_unique_name(self, name: str) -> bool:
        raise NotImplementedError()

    def unique_name(self) -> str:
        global _CURRENT_COUNT, _MAX_COMBINATIONS
        while name := coolname.generate_slug(_NAME_LEN):
            if _CURRENT_COUNT >= _MAX_COMBINATIONS:
                logger.debug(f"exhausted names from the coolname package with maximum={_MAX_COMBINATIONS}.")
                # dev - this will probably never happen (len=2, combinations=320289), if it does then increase _NAME_LEN
                break
            if not self.is_unique_name(name):
                _CURRENT_COUNT += 1
                return name
        raise RuntimeError("unique names exhausted.")
