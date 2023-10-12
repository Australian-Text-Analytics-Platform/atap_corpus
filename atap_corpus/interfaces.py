from typing import Any, Hashable, Optional, TypeVar
from abc import ABCMeta, abstractmethod

from atap_corpus.types import Mask, PathLike

TClonable = TypeVar("TClonable", bound='Clonable')
TSerialisable = TypeVar("TSerialisable", bound='Serialisable')


class Clonable(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        self._parent: Optional[TClonable] = None  # tracks the parent reference.
        self._mask: Optional[Mask] = None

    # noinspection PyTypeChecker
    @abstractmethod
    def cloned(self, mask: Mask, *args, **kwargs) -> TClonable:
        """ Returns the Clonable given a binary mask. """
        cloneable = self.__class__(*args, **kwargs)
        cloneable._parent = self
        cloneable._mask = mask
        return cloneable

    @abstractmethod
    def detached(self) -> TClonable:
        """ Detaches from the tree and return a copy of itself."""
        raise NotImplementedError()

    @property
    def parent(self) -> TClonable:
        return self._parent

    @property
    def is_root(self) -> bool:
        return self._parent is None

    def find_root(self) -> TClonable:
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
    def deserialise(cls, path: PathLike) -> TSerialisable:
        """ Deserialise configuration and return the deserialised object. """
        raise NotImplementedError()

    @abstractmethod
    def serialise(self, path: PathLike, *args, **kwargs) -> PathLike:
        """ Serialises configuration into a persistent format. """
        raise NotImplementedError()
