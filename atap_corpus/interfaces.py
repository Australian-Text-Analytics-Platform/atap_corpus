from typing import Any, Hashable, Protocol, runtime_checkable
from abc import ABCMeta, abstractmethod

from atap_corpus.types import TMask, TPathLike


class Clonable(metaclass=ABCMeta):
    @abstractmethod
    def cloned(self, mask: TMask) -> 'Clonable':
        raise NotImplementedError()


class Container(metaclass=ABCMeta):
    @abstractmethod
    def add(self, obj: Any):
        """ Add object to container. """
        raise NotImplementedError()

    @abstractmethod
    def remove(self, key: Hashable):
        """ Remove the object from container. """
        raise NotImplementedError()

    @abstractmethod
    def items(self) -> list[Any]:
        """ List all the objects in the container. """
        raise NotImplementedError()

    @abstractmethod
    def clear(self):
        """ Clears all the objects in the container. """
        raise NotImplementedError()

    @abstractmethod
    def get(self, key: Hashable) -> Any:
        """ Get the object in the container with key. """
        raise NotImplementedError()


class Serialisable(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def deserialise(cls, path: TPathLike) -> 'Serialisable':
        """ Deserialise configuration and return the deserialised object. """
        raise NotImplementedError()

    @abstractmethod
    def serialise(self, path: TPathLike) -> TPathLike:
        """ Serialises configuration into a persistent format. """
        raise NotImplementedError()


@runtime_checkable
class UniqueNameProvider(Protocol):
    """ Provides a unique string.

    Protocol - does not enforce a strict class hierarchy for future subclasses.
    """

    def unique_name(self) -> str:
        ...

    def is_unique_name(self, name: str) -> bool:
        ...
