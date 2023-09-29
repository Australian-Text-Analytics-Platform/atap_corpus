from typing import Union
from os import PathLike
import spacy.tokens

TMask = 'pd.Series[bool]'
TPathLike = Union[PathLike[str], PathLike[bytes]]
TDoc = Union[str, spacy.tokens.Doc]
