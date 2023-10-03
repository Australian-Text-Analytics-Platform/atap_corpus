from os import PathLike

import spacy.tokens

TMask = 'pd.Series[bool]'
TDoc = str | spacy.tokens.Doc
TDocs = 'pd.Series[TDoc]'
TPathLike = PathLike[str] | PathLike[bytes]
