from typing import Any, Hashable, Optional
from abc import ABCMeta, abstractmethod

from atap_corpus.types import Mask, PathLike


class Clonable(metaclass=ABCMeta):
    def __init__(self):
        self._parent: Optional['Clonable'] = None  # tracks the parent reference.

    # noinspection PyTypeChecker
    @abstractmethod
    def cloned(self, mask: Mask) -> 'Clonable':
        """ Returns the Clonable given a binary mask. """
        self._parent = self
        raise NotImplementedError()

    @abstractmethod
    def detached(self) -> 'Clonable':
        """ Detaches from the tree and return a copy of itself."""
        raise NotImplementedError()

    @property
    def parent(self) -> 'Clonable':
        return self._parent

    @property
    def is_root(self) -> bool:
        return self._parent is None

    def find_root(self) -> 'Clonable':
        """ Returns the root of the cloned object. """
        if self._parent is None: return self
        if self.is_root: return self
        parent = self.parent
        while not parent.is_root:
            parent = parent.parent
        return parent


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
    def serialise(self, path: PathLike, *args, **kwargs) -> PathLike:
        """ Serialises configuration into a persistent format. """
        raise NotImplementedError()
