import os
from typing import TypeAlias

import pandas as pd
import spacy.tokens

Mask: TypeAlias = 'pd.Series[bool]'  # amend to this type alias as necessary.
Doc: TypeAlias = str | spacy.tokens.Doc
Docs: TypeAlias = 'pd.Series[Doc]'
PathLike: TypeAlias = str | os.PathLike[str]
