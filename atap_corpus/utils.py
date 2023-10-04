""" Collections of utility functions

"""
from typing import Type, Any


def format_dunder_str(cls: Type[Any], *args, **kwargs) -> str:
    """ Utility function to standardise overridden __str__ formatting.

    Example returned string:
    <class_name arg0,arg1 key0=value0, key1=value1>
    """
    args = ",".join(args)
    kwargs = ",".join([f"{k}: {v}" for k, v in kwargs.items()])
    return f"<{cls.__name__} {args} {kwargs}>"
