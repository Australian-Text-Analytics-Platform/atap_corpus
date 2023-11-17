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


# note: this is an ad-hoc download function (jupyter only)
def download(corpus: TCorpus) -> 'HTML':
    if _IS_JUPYTER:
        from IPython.display import HTML
        path = corpus.name + ".zip"
        path = Path(corpus.serialise(path))
        import html
        href = html.escape("./" + path.name)
        default_fname = path.name
        return HTML(f'<a href=/files/{href} download={default_fname}>Download {default_fname}</a>')
    else:
        raise NotImplementedError("Download is only available in jupyter.")
