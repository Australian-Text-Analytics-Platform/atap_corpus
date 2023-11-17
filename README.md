<a href="https://atap.edu.au"><img src="https://www.atap.edu.au/atap-logo.png" width="125" height="50" align="right"></a>  
# ATAP Corpus

Provides a standardised base Corpus structure for ATAP tools.

Different Corpus can be sliced into subcorpus based on different criterias and will always return an subclass
instance of BaseCorpus.
The slicing criteria is flexible, it accepts a user defined function and comes with convenience slicing
operations layered on top of it out-of-the-box.
Subcorpus maintains a parent-child relationship with original corpus in a tree internally.

Corpus can also be serialised and deserialised which can be used to carry across different ATAP analytics notebooks.

```shell
pip install atap_corpus
```

[//]: # (### Extras: Viz:)

[//]: # ()
[//]: # (Out of the box, Corpus also comes with simple and quick visualisations such as word clouds, timelines etc.)

[//]: # ()
[//]: # (```shell)

[//]: # (pip install atap_corpus[viz])

[//]: # (```)

## Tests

To run all the unit tests, there is a script you can execute.

```shell
./scripts/run_tests.sh
```

This repo originated from Juxtorpus and is a decoupling effort.
Juxtorpus repo may be accessed [here](https://github.com/Sydney-Informatics-Hub/juxtorpus).
