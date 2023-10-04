import os
from typing import TypeAlias

import spacy.tokens

Mask: TypeAlias = 'pd.Series[bool]'
Doc: TypeAlias = str | spacy.tokens.Doc
Docs: TypeAlias = 'pd.Series[Doc]'
PathLike: TypeAlias = os.PathLike[str] | os.PathLike[bytes]
