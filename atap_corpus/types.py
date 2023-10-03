from os import PathLike

import spacy.tokens

TMask = 'pd.Series[bool]'
TPathLike = PathLike[str] | PathLike[bytes]
TDoc = str | spacy.tokens.Doc
