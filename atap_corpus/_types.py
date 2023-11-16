import os
from typing import TypeAlias, TypeVar

import pandas as pd
import spacy.tokens

Mask: TypeAlias = 'pd.Series[bool]'  # amend to this type alias as necessary.
Doc: TypeAlias = str | spacy.tokens.Doc
Docs: TypeAlias = 'pd.Series[Doc]'
PathLike: TypeAlias = str | os.PathLike[str]

# msgpack
_MPK_SUPPORTED_PRIMITIVES = str | int | float
_MPK_SUPPORTED_STRUCTURES = dict[_MPK_SUPPORTED_PRIMITIVES, _MPK_SUPPORTED_PRIMITIVES] | \
                            list[_MPK_SUPPORTED_PRIMITIVES] | \
                            tuple[_MPK_SUPPORTED_PRIMITIVES]
MPK_SUPPORTED = dict[_MPK_SUPPORTED_PRIMITIVES, _MPK_SUPPORTED_PRIMITIVES | _MPK_SUPPORTED_STRUCTURES] | \
                list[_MPK_SUPPORTED_PRIMITIVES | _MPK_SUPPORTED_STRUCTURES] | \
                tuple[_MPK_SUPPORTED_PRIMITIVES | _MPK_SUPPORTED_STRUCTURES]

# within this package
TClonable = TypeVar("TClonable", bound='Clonable')
TSerialisable = TypeVar("TSerialisable", bound='Serialisable')
TCorpus = TypeVar("TCorpus", bound='BaseCorpus')
TCorpusWithMeta = TypeVar("TCorpusWithMeta", bound="BaseCorpusWithMeta")
TCorpora = TypeVar("TCorpora", bound='BaseCorpora')
TFreqTable = TypeVar("TFreqTable", bound='BaseFreqTable')
