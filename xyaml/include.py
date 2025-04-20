"""
Handles !include and !relpath tags

.. py:class:: IncludeLoader(self, stream: IO)

    YAML Loader with `!include` constructor.


.. autofunction:: search_file

.. autofunction:: construct_include

.. autofunction:: construct_relative_path

.. autodata:: INCLUDE_TAG

.. autodata:: RELPATH_TAG

"""

import os
import json
import pathlib as pl

import yaml

from typing import Any, IO

# se: https://stackoverflow.com/questions/528281/how-can-i-include-a-yaml-file-inside-another
# metaclass example: https://gist.github.com/joshbode/569627ced3076931b02f?permalink_comment_id=2309157#gistcomment-2309157
# possibly helpful: https://matthewpburruss.com/post/yaml/


class IncludeLoader(yaml.FullLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""
        self.file_path = pl.Path(stream.name)

        super().__init__(stream)


def search_file(parent_path: os.PathLike, path_str: os.PathLike) -> pl.Path:
    """Include file referenced at node.

    resolve order:

    1. absolute or resolvable/home paths (i.e. ~/foo.yaml)
    2. relative to yaml file
    3. relative to cwd

    see :doc:`/guide/yaml` for an example
    """
    node_path = pl.Path(path_str)
    parent_path = pl.Path(parent_path)

    cwd = pl.Path.cwd()
    resolved = node_path.expanduser().resolve()
    not_cwd = resolved.exists() and not resolved.is_relative_to(cwd)
    if node_path.is_absolute() or not_cwd:
        return resolved
    elif (path := parent_path / node_path).exists():
        return path
    elif (path := pl.Path.cwd() / node_path).exists():
        return path
    else:
        msg = f"Could resolve path: {node_path}"
        raise FileNotFoundError(msg)


def construct_include(loader: IncludeLoader, node: yaml.Node) -> Any:
    """Read the contents of a yaml/text/json file into a node"""
    node_path = search_file(loader.file_path.parent, loader.construct_scalar(node))

    ext = node_path.suffix

    with open(node_path, 'r') as f:
        if ext in ('.yaml', '.yml'):
            return yaml.load(f, IncludeLoader)
        elif ext in ('.json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())


def construct_relative_path(loader: IncludeLoader, node: yaml.Node) -> str:
    """Construct a path which is resolved absolutely or relative to the yaml file"""
    node_path = search_file(loader.file_path.parent, loader.construct_scalar(node))
    return str(node_path.resolve().absolute())

#: str : YAML tag for !include
INCLUDE_TAG = '!include'
#: str : YAML tag for !relpath
RELPATH_TAG = '!relpath'

yaml.add_constructor(INCLUDE_TAG, construct_include, IncludeLoader)
yaml.add_constructor(RELPATH_TAG, construct_relative_path, IncludeLoader)


if __name__ == '__main__':
    this_dir = pl.Path(__file__).resolve().parent
    with open(this_dir / 'test.yaml') as f:
        d = yaml.load(f, IncludeLoader)
    print(d)
