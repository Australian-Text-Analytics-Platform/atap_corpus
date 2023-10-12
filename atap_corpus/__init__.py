from atap_corpus.utils import _IS_JUPYTER

if _IS_JUPYTER:
    from atap_corpus.utils import setup_loggers

    setup_loggers("./logging_conf.ini")
