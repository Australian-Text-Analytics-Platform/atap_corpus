import io
import os
from typing import Any, Hashable, Optional, IO, Callable, Protocol, runtime_checkable
from abc import ABCMeta, abstractmethod

from atap_corpus._types import Mask, PathLike, TClonable, TSerialisable


class Clonable(metaclass=ABCMeta):
    def __new__(cls, *args, **kwargs):
        instance = super(Clonable, cls).__new__(cls)
        instance._parent: Optional[TClonable] = None  # tracks the parent reference.
        instance._mask: Optional[Mask] = None
        return instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # noinspection PyTypeChecker
    @abstractmethod
    def cloned(self, mask: Mask, *args, **kwargs) -> TClonable:
        """ Returns the Clonable given a binary mask. """
        # set the _parent and _mask before init is called so there's a separation of this
        # initialisation and hence clonable subclass will have _parent and _mask populated
        # before __init__ is called, therefore add any conditionals dependent on it as needed.
        clonable = self.__new__(self.__class__)
        clonable._parent = self
        clonable._mask = mask
        assert not clonable.is_root, "Cloned instances should never be root."
        clonable.__init__(*args, **kwargs)
        return clonable

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

    def find_lowest_common_ancestor(self, other: TClonable) -> TClonable:
        """ Returns the lowest common ancestor with the 'other' Clonable.
        :param other: the other Clonable to check against.
        :return: the lowest common ancestor/first common parent Clonable.
        """
        if not self.find_root() == other.find_root():
            raise TypeError(f"{other} do not share the same root as {self}.")
        parents = set()
        parent = self
        while parent is not None:
            parents.add(parent)
            parent = parent.parent

        parent = other.parent
        while parent is not None:
            if parent in parents:
                return parent
            parent = parent.parent
        # should never be raised.
        raise RuntimeError(f"No common parent found for {self} and {other} despite same root.")


# dev - not sure if we should use Protocol or ABC.
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


# dev - not sure if we should use Protocol or ABC.
class Serialisable(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def deserialise(cls, path_or_file: PathLike | IO) -> TSerialisable:
        """ Deserialise configuration and return the deserialised object.

        This base method transform path into binary io, otherwise io is passed through.
        """
        should_close = True
        if isinstance(path_or_file, str | os.PathLike):
            file = io.BufferedReader(io.FileIO(path_or_file, mode='r'))
        elif isinstance(path_or_file, io.IOBase):
            file = path_or_file
            if not path_or_file.readable():
                raise io.UnsupportedOperation(f"{path_or_file} is not readable.")
            should_close = False
        else:
            raise ValueError(f"{path_or_file} must be a path or IO.")
        return file, should_close

    @abstractmethod
    def serialise(self, path_or_file: PathLike | IO, *args, **kwargs) -> PathLike | IO:
        """ Serialises configuration into a persistent format.

        This base method transform path into binary io, otherwise io is passed through.
        """
        should_close = True
        if isinstance(path_or_file, str | os.PathLike):
            file = io.BufferedWriter(io.FileIO(path_or_file, mode='wb'))
        elif isinstance(path_or_file, io.IOBase):
            file = path_or_file
            if not file.writable():
                raise io.UnsupportedOperation(f"{path_or_file} is not writable.")
            should_close = False
        else:
            raise ValueError(f"{path_or_file} must be a path or IO.")
        return file, should_close


@runtime_checkable
class Filterable(Protocol):
    def apply(self, func: Callable) -> Mask:
        ...
