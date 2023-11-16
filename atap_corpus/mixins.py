""" mixins.py

Provides the following Mixins:
+ UniqueIDProviderMixin - generates unique id with UUIDs, inheritter must override is_unique_id.
+ UniqueNameProviderMixin - generates unique names with coolname, inheritter must override is_unique_name.
+ SpacyDocMixin - for inheritting in the Corpus hierarchy to have spacy behaviours.
+ ClonableDTMRegistryMixin - for inheritting in the Corpus hierarchy to handle dtm cloning behaviours.
"""

import uuid
from abc import abstractmethod
from typing import Optional, Callable, Type
import functools
import logging

import spacy
import coolname

from atap_corpus.parts.base import BaseDTM
from atap_corpus._types import Doc
from atap_corpus.interfaces import Clonable

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

    _WARNING_COLLISION_COUNT = 20
    _ERROR_COLLISION_COUNT = 50

    @abstractmethod
    def is_unique_id(self, id_: uuid.UUID | str):
        raise NotImplementedError()

    def unique_id(self) -> uuid.UUID:
        counter = 0
        while id_ := uuid.uuid4():
            if self.is_unique_id(id_):
                return id_
            counter += 1
            if counter == self._WARNING_COLLISION_COUNT:
                logger.warning(f"Generated {counter} collided unique IDs. Issue with UUID?.")
            if counter >= self._ERROR_COLLISION_COUNT:
                logger.error(f"Generated {counter} collided unique IDs. ")
                raise RuntimeError("Too many IDs are colliding. Issue with UUID?.")


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

    def unique_name_number_suffixed(self, name: str, delimiter: str = '.') -> str:
        """ Returns a unique name based on provided name by suffixing with a number that is infinitely incremented
        If name provided is unique, name is returned.
        :param name: the name you want to retain.
        :param delimiter: when suffixed use this delimiter (defaults to '.')
        :return: a unique name suffixed with a number if the name you want to retain won't be unique.
        :raises MemoryError - very minimal possibility.
        Note: python supports infinite integers as long as you have enough memory. Until it raises MemoryError.
        """
        if self.is_unique_name(name): return name
        suffix = 0
        while uniq_name := name + delimiter + str(suffix):
            if self.is_unique_name(uniq_name):
                return uniq_name
            suffix += 1


class SpacyDocsMixin(object):
    """ This Mixin class is not yet clearly defined as the usages are not completely established.

    SpacyDocsMixin is supposed to provide reusable spacy related functions.
    It also distinguishes classes (typically BaseCorpus children at this stage) from having the
    ability to use spacy docs and the functionalities that comes with it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


class ClonableDTMRegistryMixin(object):
    """ ClonableDTMRegistryMixin provides re-usable clonable registry that stores DTMs.

    Use this Mixin to inherit the ability to store multiple DTMs and accessing the
    correct DTM clone associated with the clone of your Clonable class.

    What you'll need to define:
    + dtm_cls: the BaseDTM subclass you want to use with this Mixin.

    This Mixin must be used with a Clonable class.

    A Mixin is used instead of a strict class hierarchy because I don't forsee having a need for one
    and this behaviour can be standalone.
    (unless perhaps you want to have multiple DTM types in your class that all clones..?)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not issubclass(self.__class__, Clonable):
            raise TypeError(f"{self.__class__.__name__} must inherit from {Clonable.__name__}.")
        self: Clonable

        # prevents accidental overwriting in subclasses.
        # sorry, this is the only state in this Mixin i promise.
        self.__dtms: dict[str, BaseDTM] = dict()

    @classmethod
    @abstractmethod
    def dtm_cls(cls) -> Type[BaseDTM]:
        """ Define which BaseDTM subclass you're using in your class with this Mixin."""
        raise NotImplementedError()

    @property
    def dtms(self) -> dict[str, BaseDTM]:
        """ Returns a shallow copy of the dictionary storing the DTMs."""
        self: Clonable | 'ClonableDTMRegistryMixin'
        if self.is_root:
            return self.__dtms.copy()
        else:
            root: 'ClonableDTMRegistryMixin' = self.find_root()
            dtms: dict[str, BaseDTM] = dict()
            for name, root_dtm in root.__dtms.items():
                dtms[name] = root_dtm.cloned(self._mask)
            return dtms

    def get_dtm(self, name: str):
        self: Clonable | 'ClonableDTMRegistryMixin'
        root: 'ClonableDTMRegistryMixin' = self.find_root()
        if name not in root.__dtms.keys():
            raise KeyError(f"DTM: {name} does not exist.")
        root_dtm: BaseDTM = root.__dtms[name]
        if self.is_root:
            return root_dtm
        else:
            return root_dtm.cloned(self._mask)

    def add_dtm(self, dtm: BaseDTM, name: str):
        if not isinstance(dtm, BaseDTM):
            raise TypeError(f"dtm is not a {BaseDTM.__name__}. Got {dtm.__class__.__name__}")
        self: Clonable | 'ClonableDTMRegistryMixin'
        root = self.find_root()
        if name in root.__dtms.keys():
            raise ValueError(f"{name} already exist. Maybe remove it?")
        if dtm.num_docs != len(root):
            raise ValueError(f"Mismatched number of docs with root Corpus. "
                             f"Expecting {len(root)}. Got {dtm.num_docs}.")
        root.__dtms[name] = dtm

    def add_dtm_from_docs(self, tokeniser_func: Callable[[Doc], list[str]], name: str):
        """ Add a DTM using the provided tokeniser_func which tokenises the documents."""
        if not callable(tokeniser_func):
            raise TypeError(f"tokeniser_func is not a callable.")
        self: Clonable | 'ClonableDTMRegistryMixin'
        root = self.find_root()
        if name in root.__dtms.keys():
            raise ValueError(f"{name} already exist. Maybe remove it?")
        if not self.is_root:
            logger.warning(f"This corpus is not root. DTM {name} will be created from root.")
        dtm = self.dtm_cls().from_docs(root.docs(), tokeniser_func=tokeniser_func)
        root.__dtms[name] = dtm
        assert name in self.dtms.keys(), f"Missing {name} from DTMs after creation. This check should always pass."

    def remove_dtm(self, name: str):
        self: Clonable
        root = self.find_root()
        try:
            del root.__dtms[name]
        except KeyError:
            raise KeyError(f"DTM with name: {name} not found.")
