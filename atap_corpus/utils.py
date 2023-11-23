""" Collections of utility functions

"""
from pathlib import Path
from typing import Type, Any, Optional

from atap_corpus._types import PathLike, TCorpus


def format_dunder_str(cls: Type[Any], *args, **kwargs) -> str:
    """ Utility function to standardise overridden __str__ formatting.

    Example returned string:
    <class_name arg0,arg1 key0=value0, key1=value1>
    """
    _args: str = ",".join((str(a) for a in args))
    _kwargs: str = ",".join([f"{k}: {v}" for k, v in kwargs.items()])
    return f"<{cls.__name__} {_args} {_kwargs}>"


# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def is_jupyter() -> bool:
    """ Checks if the environment is jupyter. """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


_IS_JUPYTER = is_jupyter()


def setup_loggers(path: Optional[PathLike] = None):
    if path is None: path = "./logging_conf.ini"
    import logging.config
    # loads logging configuration file at root.
    logging.config.fileConfig(path)
    logger = logging.getLogger(__name__)
    logger.debug(f"Loggers configured with {path}")


# --- Ad Hoc ----
# below are ad hoc functions added as part of v0.1.x
# pending Corpus Loader. HASS-227


# note: this is an ad-hoc download function (jupyter only)
def download(corpus: TCorpus) -> 'FileDownload':
    if _IS_JUPYTER:
        import tempfile
        import panel as pn
        pn.extension()
        tmpd = Path(tempfile.mkdtemp())
        path = tmpd.joinpath(corpus.name + ".zip")
        path = Path(corpus.serialise(path))
        default_fname = path.name
        return pn.widgets.FileDownload(file=path.__str__(), filename=default_fname,
                                       icon='download')
    else:
        raise NotImplementedError("Download is only available in jupyter.")


def corpus_uploader() -> tuple['pn.Column', 'Corpora']:
    """ Creates and return a File Uploader for Corpus and a Corpora that stores them.
    Each Corpus uploaded will be stored in the Corpora. There can be multiple selected or uploaded multiple times.
    """
    import panel as pn
    import io
    from atap_corpus.corpus import Corpus, Corpora
    pn.extension()
    finp = pn.widgets.FileInput(accept='.zip', multiple=True)
    corpora = Corpora()

    def on_upload(_):
        for _bytes in finp.value:
            b_io = io.BytesIO(_bytes)
            corpus = Corpus.deserialise(b_io)
            b_io.close()
            corpora.add(corpus)

    finp.param.watch(on_upload, 'value')
    header = pn.pane.HTML("<h4>Upload Corpus</h4><span>Each upload will be added to the Corpora.</span>")
    return pn.Column(header, finp), corpora

# ------------------------------------------------------
