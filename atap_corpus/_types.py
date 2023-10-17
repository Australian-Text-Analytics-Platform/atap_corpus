import os
from typing import TypeAlias, TypeVar

import pandas as pd
import spacy.tokens

Mask: TypeAlias = 'pd.Series[bool]'  # amend to this type alias as necessary.
Doc: TypeAlias = str | spacy.tokens.Doc
Docs: TypeAlias = 'pd.Series[Doc]'
PathLike: TypeAlias = str | os.PathLike[str]

# within this package
TClonable = TypeVar("TClonable", bound='Clonable')
TSerialisable = TypeVar("TSerialisable", bound='Serialisable')
TCorpus = TypeVar("TCorpus", bound='BaseCorpus')
TCorpusWithMeta = TypeVar("TCorpusWithMeta", bound="BaseCorpusWithMeta")
TCorpora = TypeVar("TCorpora", bound='BaseCorpora')
TFreqTable = TypeVar("TFreqTable", bound='BaseFreqTable')
