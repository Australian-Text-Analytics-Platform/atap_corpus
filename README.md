# ATAP Corpus

Provides a standardised base Corpus structure for ATAP tools.

Corpus can be sliced into a subcorpus based on different criterias and will always return a Corpus object. The slicing criteria is highly flexible by accepting a user defined function with more convenient slicing operations layered on top of it. Subcorpus maintains a relationship with the parent corpus in a tree internally.

Corpus can also be serialised and deserialised which can be used to carry across different ATAP analytical notebooks.

```shell
pip install atap_corpus
```


### Extras: Viz:
Out of the box, Corpus also comes with simple but quick visualisations such as word clouds, timelines etc.

```shell
pip install atap_corpus[viz]
```
