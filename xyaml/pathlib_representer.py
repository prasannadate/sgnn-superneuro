"""Provides a custom representer for Path objects


.. autofunction:: represent_path

Examples
--------

.. code-block:: python-console

    >>> import novel_swarms.yaml
    >>> import pathlib as pl
    >>> novel_swarms.yaml.dump(pl.Path('foo/bar'), sys.stdout)

.. code-block:: yaml

    !!python/object/apply:pathlib.Path
    args:
    - foo/bar
    resolved: /home/username/foo/bar

"""

import yaml
import pathlib


def represent_path(dumper: yaml.Dumper, data: pathlib.Path):
    """Represent a pathlib.Path object as a reconstructable string.

    Parameters
    ----------
    dumper : yaml.Dumper
    data : pathlib.Path

    Returns
    -------
    node
        str node
    """
    tag = "tag:yaml.org,2002:python/object/apply:"
    function = type(data)
    function_name = f"{function.__module__}.{function.__name__}"
    value = {'args': [str(data)],}
    if isinstance(data, pathlib.Path):
        try:
            value['resolved'] = str(data.expanduser().resolve())
        except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError):
            value['resolved'] = str(data.expanduser())
    return dumper.represent_mapping(tag + function_name, value)
