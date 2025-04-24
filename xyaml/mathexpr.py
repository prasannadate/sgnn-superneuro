"""
Handle !np yaml tag. Allows for evaluation of simple math/numpy expressions.

.. WARNING::

    This code is not safe for arbitrary code execution as it uses ``eval()``.
    Do not use untrusted ``.yaml`` files.
    While the functionality has been limited to expressions, ``eval()`` is still dangerous.
    Assignments and named expressions (walrus ``:=``) and lambdas are disallowed to prevent
    the user from breaking things too much.

Examples
--------

.. code-block:: yaml
    :caption: YAML

    0: !np [radians(90 + 45), pi / 2, 3.14]  # -> [-2.356194490192345, 1.5707963267948966, 3.14]
    1: !np complex('2+2j')  # -> (2+2j)
    2: !np array(list(range(10)), dtype=float)  # -> array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

Allowed Names
-------------

.. autodata:: allowed_builtins_names

    Names of builtins that are allowed to be used in expressions.

.. autodata:: allowed_numpy_names

    Names of numpy objects that are allowed to be used in expressions.

"""
import yaml
import ast
import numpy
import numpy as np
import builtins
from yaml import SequenceNode, MappingNode, ScalarNode
from types import SimpleNamespace
from typing import Any, IO

allowed_builtins_names = [
    'all', 'any', 'bin', 'bool', 'complex', 'dict', 'complex', 'int',
    'float', 'len', 'range', 'list', 'max', 'min', 'oct', 'pow', 'repr',
    'round', 'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'zip',
    'Ellipsis', 'None', 'True', 'False'
]
numpy.PI = numpy.pi  # pyright: ignore[reportAttributeAccessIssue]
allowed_numpy_names = [
    'PI', 'Inf', 'Infinity', 'MAXDIMS', 'NAN', 'NINF', 'NZERO', 'NaN', 'PINF', 'PZERO', 'abs', 'absolute', 'acos', 'acosh', 'add', 'all', 'allclose', 'alltrue', 'amax', 'amin', 'angle', 'any', 'apply_along_axis', 'apply_over_axes', 'arange', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'argmax', 'argmin', 'argpartition', 'argsort', 'argwhere', 'around', 'array', 'array2string', 'array_equal', 'array_equiv', 'array_split', 'atan', 'atan2', 'atanh', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'average', 'bartlett', 'bincount', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'blackman', 'block', 'bmat', 'bool', 'bool_', 'broadcast', 'broadcast_arrays', 'broadcast_shapes', 'broadcast_to', 'busday_count', 'busday_offset', 'busdaycalendar', 'byte', 'bytes_', 'can_cast', 'cast', 'cbrt', 'cdouble', 'ceil', 'cfloat', 'char', 'character', 'chararray', 'choose', 'clip', 'clongdouble', 'clongfloat', 'column_stack', 'common_type', 'compare_chararrays', 'compat', 'complex128', 'complex256', 'complex64', 'complex_', 'complexfloating', 'compress', 'concatenate', 'conj', 'conjugate', 'convolve', 'copy', 'copysign', 'copyto', 'corrcoef', 'correlate', 'cos', 'cosh', 'count_nonzero', 'cov', 'cross', 'csingle', 'ctypeslib', 'cumprod', 'cumproduct', 'cumsum', 'datetime64', 'datetime_as_string', 'datetime_data', 'deg2rad', 'degrees', 'diag', 'diag_indices', 'diag_indices_from', 'diagflat', 'diagonal', 'diff', 'digitize', 'disp', 'divide', 'divmod', 'dot', 'double', 'dsplit', 'dstack', 'dtype', 'dtypes', 'e', 'ediff1d', 'einsum', 'einsum_path', 'emath', 'empty', 'empty_like', 'equal', 'errstate', 'euler_gamma', 'exceptions', 'exp', 'exp2', 'expand_dims', 'expm1', 'expm1x', 'extract', 'eye', 'fabs', 'fft', 'fill_diagonal', 'finfo', 'fix', 'flatiter', 'flatnonzero', 'flexible', 'flip', 'fliplr', 'flipud', 'float128', 'float16', 'float32', 'float64', 'float_', 'float_power', 'floating', 'floor', 'floor_divide', 'fmax', 'fmin', 'fmod', 'format_parser', 'frexp', 'full', 'full_like', 'gcd', 'generic', 'genfromtxt', 'geomspace', 'get_array_wrap', 'get_include', 'get_printoptions', 'getbufsize', 'geterr', 'geterrcall', 'geterrobj', 'gradient', 'greater', 'greater_equal', 'half', 'hamming', 'hanning', 'heaviside', 'histogram', 'histogram2d', 'histogram_bin_edges', 'histogramdd', 'hsplit', 'hstack', 'hypot', 'i0', 'identity', 'iinfo', 'imag', 'in1d', 'index_exp', 'indices', 'inexact', 'inf', 'info', 'infty', 'inner', 'insert', 'int16', 'int32', 'int64', 'int8', 'int_', 'intc', 'integer', 'interp', 'intersect1d', 'intp', 'invert', 'is_busday', 'isclose', 'iscomplex', 'iscomplexobj', 'isfinite', 'isin', 'isinf', 'isnan', 'isnat', 'isneginf', 'isposinf', 'isreal', 'isrealobj', 'isscalar', 'issctype', 'issubclass_', 'issubdtype', 'issubsctype', 'iterable', 'ix_', 'kaiser', 'kernel_version', 'kron', 'lcm', 'ldexp', 'left_shift', 'less', 'less_equal', 'lexsort', 'lib', 'linalg', 'linspace', 'little_endian', 'load', 'loadtxt', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'logspace', 'long', 'longcomplex', 'longdouble', 'longfloat', 'longlong', 'lookfor', 'ma', 'mask_indices', 'mat', 'matmul', 'matrix', 'max', 'maximum', 'mean', 'median', 'memmap', 'meshgrid', 'mgrid', 'min', 'min_scalar_type', 'minimum', 'mintypecode', 'mod', 'modf', 'moveaxis', 'msort', 'multiply', 'nan', 'nan_to_num', 'nanargmax', 'nanargmin', 'nancumprod', 'nancumsum', 'nanmax', 'nanmean', 'nanmedian', 'nanmin', 'nanpercentile', 'nanprod', 'nanquantile', 'nanstd', 'nansum', 'nanvar', 'nbytes', 'ndarray', 'ndenumerate', 'ndim', 'ndindex', 'nditer', 'negative', 'nested_iters', 'newaxis', 'nextafter', 'nonzero', 'not_equal', 'numarray', 'number', 'obj2sctype', 'object_', 'ogrid', 'oldnumeric', 'ones', 'ones_like', 'outer', 'packbits', 'pad', 'partition', 'percentile', 'pi', 'piecewise', 'place', 'poly', 'poly1d', 'polyadd', 'polyder', 'polydiv', 'polyfit', 'polyint', 'polymul', 'polynomial', 'polysub', 'polyval', 'positive', 'pow', 'power', 'printoptions', 'prod', 'product', 'promote_types', 'ptp', 'put', 'put_along_axis', 'putmask', 'quantile', 'rad2deg', 'radians', 'random', 'ravel', 'ravel_multi_index', 'real', 'real_if_close', 'rec', 'recarray', 'recfromcsv', 'recfromtxt', 'reciprocal', 'record', 'remainder', 'repeat', 'require', 'reshape', 'resize', 'result_type', 'right_shift', 'rint', 'roll', 'rollaxis', 'roots', 'rot90', 'round', 'round_', 'row_stack', 's_', 'safe_eval', 'sctype2char', 'sctypeDict', 'sctypes', 'searchsorted', 'select', 'set_numeric_ops', 'set_printoptions', 'set_string_function', 'setxor1d', 'shape', 'short', 'show_config', 'show_runtime', 'sign', 'signbit', 'signedinteger', 'sin', 'sinc', 'single', 'singlecomplex', 'sinh', 'size', 'sometrue', 'sort', 'sort_complex', 'source', 'spacing', 'split', 'sqrt', 'square', 'squeeze', 'stack', 'std', 'str_', 'string_', 'subtract', 'sum', 'swapaxes', 'take', 'take_along_axis', 'tan', 'tanh', 'tensordot', 'tile', 'timedelta64', 'trace', 'transpose', 'tri', 'tril', 'tril_indices', 'tril_indices_from', 'trim_zeros', 'triu', 'triu_indices', 'triu_indices_from', 'true_divide', 'trunc', 'typename', 'ubyte', 'ufunc', 'uint', 'uint16', 'uint32', 'uint64', 'uint8', 'uintc', 'uintp', 'ulong', 'ulonglong', 'unicode_', 'union1d', 'unique', 'unpackbits', 'unravel_index', 'unsignedinteger', 'unwrap', 'ushort', 'vander', 'var', 'vecdot', 'vdot', 'vectorize', 'version', 'void', 'vsplit', 'vstack', 'where', 'zeros', 'zeros_like'  # noqa: E501
]
builtins_dict = vars(builtins)
allowed_builtins = {name: builtins_dict[name] for name in allowed_builtins_names}
allowed_math = {k: vars(numpy)[k] for k in allowed_numpy_names if k in vars(numpy)}
allowed_np_prefixed = SimpleNamespace(**allowed_math)

eval_globals = {'__builtins__': allowed_builtins}
eval_names = allowed_math | {'np': allowed_np_prefixed}
var_names = allowed_builtins_names + allowed_numpy_names + ['np']
# for detecting undefined variables and replacing with string


def safe_eval(expr, variables=(), skip_errors=True):
    """
    Somewhat safely evaluate a a string containing a Python expression.
    """
    #: These names are very unsafe.
    unsafe_nodes = [
        'Delete', 'Assert', 'Raise', 'AnnAssign', 'Assign', 'AugAssign', 'NamedExpr', 'Import', 'ImportFrom',
        'Lambda', 'FunctionDef', 'Global', 'Nonlocal', 'ClassDef', 'Yield', 'YieldFrom', 'Return',
        'AsyncFor', 'AsyncWith', 'AsyncFunctionDef',
    ]
    undefined_names = []
    node = ast.parse(expr, mode='eval')
    for subnode in ast.walk(node):
        subnode_name = type(subnode).__name__
        if isinstance(subnode, ast.Name):
            name = subnode.id
            if name not in var_names:
                undefined_names.append(name)
        if subnode_name in unsafe_nodes:
            msg = f"Attempted unsafe evaluation of expression containing {subnode_name} statement."
            raise ValueError(msg)
    if undefined_names:
        return str(expr)
    try:
        return eval(expr, eval_globals, eval_names)
    except KeyboardInterrupt:
        raise
    except Exception:
        return str(expr)


def unwrap_node_recursive(loader, node):
    """Handles ``!np`` tag on sequences and mappings."""
    if isinstance(node, SequenceNode):
        return [unwrap_node_recursive(loader, n) for n in node.value]
    elif isinstance(node, MappingNode):
        return {unwrap_node_recursive(loader, k): unwrap_node_recursive(loader, v) for k, v in node.value}
    else:
        s = loader.construct_scalar(node)
        return safe_eval(s)


def construct_numexpr(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    """Handles the ``!np`` tag on YAML nodes."""
    return unwrap_node_recursive(loader, node)


if __name__ == '__main__':
    yaml.add_constructor('!np', construct_numexpr, yaml.SafeLoader)

    with open('src/novel_swarms/util/yaml/test.yaml') as f:
        d = yaml.load(f, yaml.SafeLoader)
    print(d)
