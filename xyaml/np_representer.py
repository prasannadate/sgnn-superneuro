""" Provides a slightly more human-readable YAML representer for numpy arrays.

.. autofunction:: represent_ndarray

    Examples
    --------

    For vectors and small matrices, we show the array as a sequence.
    This representation is used if:

    * The array is a vector with ``len(array) < 11``
    * The array is a 2D matrix with fewer than 9 elements

    .. code-block:: python

        >>> import sys
        >>> import numpy as np
        >>> import novel_swarms.yaml
        >>> novel_swarms.yaml.dump({
                'small_array': np.array([8, 8]),
                'big_array': np.ones((3, 3)),
                'very_big_array': np.ones((10, 10)),
            }, sys.stdout)

    .. code-block:: yaml

        big_array: !!python/object/apply:numpy.array
        args: !!python/tuple
        - [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        kwargs:
            dtype: float64
        small_array: !!python/object/apply:numpy.array
        args: !!python/tuple
        - - 8
            - 8
        kwargs:
            dtype: int64
        very_big_array: !!python/object/apply:numpy._core.multiarray._reconstruct
        args:
        - !!python/name:numpy.ndarray ''
        - !!python/tuple
            - 0
        - !!binary |
            Yg==
        state: !!python/tuple
        - 1
        - !!python/tuple
            - 10
            - 10
        - !!python/object/apply:numpy.dtype
            args:
            - f8
            - false
            - true
            state: !!python/tuple
            - 3
            - <
            - null
            - null
            - null
            - -1
            - 0
        - false
        - !!binary |
            AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8A
            AAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAA
            AAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAA
            AAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAA
            AADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAA
            APA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA
            8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADw
            PwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/
            AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8A
            AAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAA
            AAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAA
            AAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAA
            AADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAA
            APA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA
            8D8=


"""

import yaml

import numpy as np
import numpy


def represent_ndarray(dumper: yaml.Dumper, data: np.ndarray):
    """For small arrays, return a more human-readable representation.

    For large arrays, return the default representation.

    Parameters
    ----------
    dumper : yaml.Dumper
    data : np.ndarray

    Returns
    -------
    node
        Either a YAML sequence or mapping node.

    """
    is_vector = data.ndim == 1
    short_vector = is_vector and len(data) < 11
    very_short_vector = is_vector and len(data) < 3
    small_matrix = data.ndim <= 2 and np.prod(data.shape) <= 9
    if not (short_vector or small_matrix):
        # for big matrices, use the default representation
        return dumper.represent_object(data)
    # if short vector or small_matrix, use a more human-readable representation
    flow_style = False if very_short_vector else True  # use block style for very short vectors
    array_node = dumper.represent_sequence('tag:yaml.org,2002:seq', data.tolist(), flow_style=flow_style)
    tag = "tag:yaml.org,2002:python/object/apply:"
    function = np.array
    function_name = f"{function.__module__}.{function.__name__}"
    value = {
        'kwargs': {
            'dtype': str(data.dtype),
            # 'order': 'F' if data.flags['F_CONTIGUOUS'] else 'C',
        },
        'args': (),
    }
    node = dumper.represent_mapping(tag + function_name, value)
    subnode = None
    for subnodes in node.value:
        if subnodes[0].value == 'args':
            subnodes[1].value = (array_node,)
    return node
