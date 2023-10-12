import uuid
from abc import abstractmethod
from typing import Optional, Callable, Hashable
import functools
import logging

import spacy
import coolname

from atap_corpus.types import Doc

logger = logging.getLogger(__name__)


class UniqueIDProviderMixin(object):
    """ Provides unique id generator functions as a Mixin.
    This only provides a resuable generation of ids but not the state.
    Inherit this Mixin and provide your own state.
    Implementation:
        + uses UUID4 to generate unique IDs. (it can theoretically generate a non-unique ID although highly unlikely)
        + this ensures all IDS will for certain be unique due to your tracking state.

    Why not just use composition?
    1. with Mixin, I'm just keeping one state which is shared (e.g. between GlobalCorpora and UniqueIDProvider)
    2. I can also check if a class is a provider.
    3. Plus, you can always inherit from this class make a separate Class and use composition that way if you want.
    Cons: Won't be able to use dependency injection - although I suspect this isn't needed for this scenario.
    """

    _WARNING_AT = 20
    _ERROR_AT = 50

    @abstractmethod
    def is_unique_id(self, id_: uuid.UUID | str):
        raise NotImplementedError()

    def unique_id(self) -> uuid.UUID:
        counter = 0
        while id_ := uuid.uuid4():
            if self.is_unique_id(id_):
                return id_
            counter += 1
            if counter == self._WARNING_AT:
                logger.warning(f"Generated {counter} collided unique IDs. Issue with UUID?.")
            if counter >= self._ERROR_AT:
                logger.error(f"Generated {counter} collided unique IDs. ")
                raise RuntimeError("Too many IDs are colliding. Issue with UUID?.")


class SpacyDocsMixin(object):
    """ This Mixin class is not yet clearly defined as the usages are not completely established.

    SpacyDocsMixin is supposed to provide reusable spacy related functions.
    It also distinguishes classes (typically BaseCorpus children at this stage) from having the
    ability to use spacy docs and the functionalities that comes with it.
    """

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


#  UniqueNameProviderMixin is unused and kept for possible future uses only. Replaced by UniqueIDProviderMixin.
#  It was initially used for GlobalCorpora where the uniqueness property is exhibited by the Corpus names.
# dev - UniqueNameProvider & UniqueIDProvider may inherit from the same parent if it seems code should be reused.
#  although it could make the function names unclear.
#  some overlapping abstract behaviours are kept for these reasons.
class UniqueNameProviderMixin(object):
    """ Provides unique name generator functions as a Mixin. """

    _NAME_LEN = 2
    _MAX_COMBINATIONS = coolname.get_combinations_count(_NAME_LEN)
    _CURRENT_COUNT = 0

    @abstractmethod
    def is_unique_name(self, name: str) -> bool:
        raise NotImplementedError()

    def unique_name(self) -> str:
        """ Returns a randomly generated unique name. """
        while name := coolname.generate_slug(self._NAME_LEN):
            if self._CURRENT_COUNT >= self._MAX_COMBINATIONS:
                logger.debug(f"exhausted names from the coolname package with maximum={self._MAX_COMBINATIONS}.")
                # dev - this will probably never happen (len=2, combinations=320289), if it does then increase _NAME_LEN
                break
            if not self.is_unique_name(name):
                self._CURRENT_COUNT += 1
            else:
                return name
        raise RuntimeError("all unique names exhausted.")

    def unique_name_number_suffixed(self, name: str) -> str:
        """ Returns a unique name based on provided name by suffixing with a number that is infinitely incremented
        :param name: the name you want to retain.
        :return: a unique name suffixed with a number if the name you want to retain won't be unique.
        :raises MemoryError - very minimal possibility.
        Note: python supports infinite integers as long as you have enough memory. Until it raises MemoryError.
        """
        suffix = 0
        while name := name + str(suffix):
            if self.is_unique_name(name):
                return name
            suffix += 1
