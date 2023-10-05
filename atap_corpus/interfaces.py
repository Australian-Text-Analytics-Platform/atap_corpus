from typing import Any, Hashable, Protocol, runtime_checkable
from abc import ABCMeta, abstractmethod

from atap_corpus.types import Mask, PathLike


class Clonable(metaclass=ABCMeta):
    @abstractmethod
    def cloned(self, mask: Mask) -> 'Clonable':
        raise NotImplementedError()


class Container(metaclass=ABCMeta):
    """ Container abstract class
    This class provides a common interface and enforce implementations of
    all classes that acts as a container of
    """
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
    def deserialise(cls, path: PathLike) -> 'Serialisable':
        """ Deserialise configuration and return the deserialised object. """
        raise NotImplementedError()

    @abstractmethod
    def serialise(self, path: PathLike) -> PathLike:
        """ Serialises configuration into a persistent format. """
        raise NotImplementedError()


