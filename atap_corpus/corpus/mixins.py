from abc import abstractmethod
from typing import Optional, Callable
import functools
import logging

import spacy
import coolname

from atap_corpus.types import Doc

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
            else:
                return name
        raise RuntimeError("all unique names exhausted.")


class SpacyDocsMixin(object):
    """ """

    @abstractmethod
    def uses_spacy(self) -> bool:
        """ Whether spacy is used. """
        raise NotImplementedError()

    @abstractmethod
    def run_spacy(self, nlp: spacy.Language, *args, **kwargs) -> None:
        if not isinstance(nlp, spacy.Language):
            raise TypeError(f"{nlp} is not a spacy pipeline.")

    def get_tokeniser(self, nlp: Optional[spacy.Language] = None) -> Callable[[str], Doc | list[str]]:
        """ Returns the tokeniser of the spacy nlp pipeline based on whether uses_spacy() is True or False.
        If uses_spacy() then, returns spacy's tokeniser.
        otherwise, returns a callable that uses a blank spacy tokeniser to tokeniser into a list of str.
        :raises RuntimeError if no tokeniser found in pipeline.
        """
        if nlp is None: nlp = spacy.blank('en')
        tokeniser = getattr(nlp, "tokenizer", None)
        if tokeniser is None:
            logger.debug("Could not find a spacy tokenizer via the nlp.tokenizer attribute.")
            logger.debug(f"All spacy components: {', '.join(nlp.component_names)}.")
            raise RuntimeError(f"The spacy pipline does not have a tokeniser.")
        if self.uses_spacy():
            return tokeniser
        else:
            def tokenise(tokeniser: spacy.tokenizer.Tokenizer, text: str) -> list[str]:
                return list([t.text for t in tokeniser(text)])

            return functools.partial(tokenise, tokeniser)
