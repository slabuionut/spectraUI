import inspect
import functools
import sys
import warnings
from collections.abc import Iterable

import numpy as np

from shared._warnings import all_warnings, warn


__all__ = ['deprecated', 'get_bound_method_class', 'all_warnings',
           'safe_as_int', 'check_shape_equality', 'check_nD', 'warn',
           'reshape_nd', 'identity', 'slice_at_axis']


class skimage_deprecation(Warning):
    pass


def _get_stack_rank(func):
    if _is_wrapped(func):
        return 1 + _get_stack_rank(func.__wrapped__)
    else:
        return 0


def _is_wrapped(func):
    return "__wrapped__" in dir(func)


def _get_stack_length(func):
    return _get_stack_rank(func.__globals__.get(func.__name__, func))


class _DecoratorBaseClass:
    _stack_length = {}

    def get_stack_length(self, func):
        return self._stack_length.get(func.__name__,
                                      _get_stack_length(func))


class change_default_value(_DecoratorBaseClass):

    def __init__(self, arg_name, *, new_value, changed_version,
                 warning_msg=None):
        self.arg_name = arg_name
        self.new_value = new_value
        self.warning_msg = warning_msg
        self.changed_version = changed_version

    def __call__(self, func):
        parameters = inspect.signature(func).parameters
        arg_idx = list(parameters.keys()).index(self.arg_name)
        old_value = parameters[self.arg_name].default

        stack_rank = _get_stack_rank(func)

        if self.warning_msg is None:
            self.warning_msg = (
                f'The new recommended value for {self.arg_name} is '
                f'{self.new_value}. Until version {self.changed_version}, '
                f'the default {self.arg_name} value is {old_value}. '
                f'From version {self.changed_version}, the {self.arg_name} '
                f'default value will be {self.new_value}. To avoid '
                f'this warning, please explicitly set {self.arg_name} value.')

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            stacklevel = 1 + self.get_stack_length(func) - stack_rank
            if len(args) < arg_idx + 1 and self.arg_name not in kwargs.keys():
                # warn that arg_name default value changed:
                warnings.warn(self.warning_msg, FutureWarning,
                              stacklevel=stacklevel)
            return func(*args, **kwargs)

        return fixed_func


class remove_arg(_DecoratorBaseClass):

    def __init__(self, arg_name, *, changed_version, help_msg=None):
        self.arg_name = arg_name
        self.help_msg = help_msg
        self.changed_version = changed_version

    def __call__(self, func):

        parameters = inspect.signature(func).parameters
        arg_idx = list(parameters.keys()).index(self.arg_name)
        warning_msg = (
            f'{self.arg_name} argument is deprecated and will be removed '
            f'in version {self.changed_version}. To avoid this warning, '
            f'please do not use the {self.arg_name} argument. Please '
            f'see {func.__name__} documentation for more details.')

        if self.help_msg is not None:
            warning_msg += f' {self.help_msg}'

        stack_rank = _get_stack_rank(func)

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            stacklevel = 1 + self.get_stack_length(func) - stack_rank
            if len(args) > arg_idx or self.arg_name in kwargs.keys():
                # warn that arg_name is deprecated
                warnings.warn(warning_msg, FutureWarning,
                              stacklevel=stacklevel)
            return func(*args, **kwargs)

        return fixed_func


def docstring_add_deprecated(func, kwarg_mapping, deprecated_version):

    if func.__doc__ is None:
        return None
    try:
        from numpydoc.docscrape import FunctionDoc, Parameter
    except ImportError:
        # Return an unmodified docstring if numpydoc is not available.
        return func.__doc__

    Doc = FunctionDoc(func)
    for old_arg, new_arg in kwarg_mapping.items():
        desc = [f'Deprecated in favor of `{new_arg}`.',
                '',
                f'.. deprecated:: {deprecated_version}']
        Doc['Other Parameters'].append(
            Parameter(name=old_arg,
                      type='DEPRECATED',
                      desc=desc)
        )
    new_docstring = str(Doc)
    split = new_docstring.split('\n')
    no_header = split[1:]
    while not no_header[0].strip():
        no_header.pop(0)

    descr = no_header.pop(0)
    while no_header[0].strip():
        descr += '\n    ' + no_header.pop(0)
    descr += '\n\n'

    final_docstring = descr + '\n    '.join(no_header)

    final_docstring = '\n'.join(
        [line.rstrip() for line in final_docstring.split('\n')]
    )
    return final_docstring


class deprecate_kwarg(_DecoratorBaseClass):

    def __init__(self, kwarg_mapping, deprecated_version, warning_msg=None,
                 removed_version=None):
        self.kwarg_mapping = kwarg_mapping
        if warning_msg is None:
            self.warning_msg = ("`{old_arg}` is a deprecated argument name "
                                "for `{func_name}`. ")
            if removed_version is not None:
                self.warning_msg += (f'It will be removed in '
                                     f'version {removed_version}. ')
            self.warning_msg += "Please use `{new_arg}` instead."
        else:
            self.warning_msg = warning_msg

        self.deprecated_version = deprecated_version

    def __call__(self, func):

        stack_rank = _get_stack_rank(func)

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            stacklevel = 1 + self.get_stack_length(func) - stack_rank

            for old_arg, new_arg in self.kwarg_mapping.items():
                if old_arg in kwargs:
                    warnings.warn(self.warning_msg.format(
                        old_arg=old_arg, func_name=func.__name__,
                        new_arg=new_arg), FutureWarning,
                        stacklevel=stacklevel)
                    kwargs[new_arg] = kwargs.pop(old_arg)

            return func(*args, **kwargs)

        if func.__doc__ is not None:
            newdoc = docstring_add_deprecated(func, self.kwarg_mapping,
                                              self.deprecated_version)
            fixed_func.__doc__ = newdoc
        return fixed_func


class channel_as_last_axis:

    def __init__(self, channel_arg_positions=(0,), channel_kwarg_names=(),
                 multichannel_output=True):
        self.arg_positions = set(channel_arg_positions)
        self.kwarg_names = set(channel_kwarg_names)
        self.multichannel_output = multichannel_output

    def __call__(self, func):

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):

            channel_axis = kwargs.get('channel_axis', None)

            if channel_axis is None:
                return func(*args, **kwargs)


            if np.isscalar(channel_axis):
                channel_axis = (channel_axis,)
            if len(channel_axis) > 1:
                raise ValueError(
                    "only a single channel axis is currently supported")

            if channel_axis == (-1,) or channel_axis == -1:
                return func(*args, **kwargs)

            if self.arg_positions:
                new_args = []
                for pos, arg in enumerate(args):
                    if pos in self.arg_positions:
                        new_args.append(np.moveaxis(arg, channel_axis[0], -1))
                    else:
                        new_args.append(arg)
                new_args = tuple(new_args)
            else:
                new_args = args

            for name in self.kwarg_names:
                kwargs[name] = np.moveaxis(kwargs[name], channel_axis[0], -1)


            kwargs["channel_axis"] = -1

            out = func(*new_args, **kwargs)
            if self.multichannel_output:
                out = np.moveaxis(out, -1, channel_axis[0])
            return out

        return fixed_func


class deprecated:

    def __init__(self, alt_func=None, behavior='warn', removed_version=None):
        self.alt_func = alt_func
        self.behavior = behavior
        self.removed_version = removed_version

    def __call__(self, func):

        alt_msg = ''
        if self.alt_func is not None:
            alt_msg = f' Use ``{self.alt_func}`` instead.'
        rmv_msg = ''
        if self.removed_version is not None:
            rmv_msg = f' and will be removed in version {self.removed_version}'

        msg = f'Function ``{func.__name__}`` is deprecated{rmv_msg}.{alt_msg}'

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if self.behavior == 'warn':
                func_code = func.__code__
                warnings.simplefilter('always', skimage_deprecation)
                warnings.warn_explicit(msg,
                                       category=skimage_deprecation,
                                       filename=func_code.co_filename,
                                       lineno=func_code.co_firstlineno + 1)
            elif self.behavior == 'raise':
                raise skimage_deprecation(msg)
            return func(*args, **kwargs)

        doc = '**Deprecated function**.' + alt_msg
        if wrapped.__doc__ is None:
            wrapped.__doc__ = doc
        else:
            wrapped.__doc__ = doc + '\n\n    ' + wrapped.__doc__

        return wrapped


def get_bound_method_class(m):

    return m.im_class if sys.version < '3' else m.__self__.__class__


def safe_as_int(val, atol=1e-3):
    """
    Attempt to safely cast values to integer format.

    Parameters
    ----------
    val : scalar or iterable of scalars
        Number or container of numbers which are intended to be interpreted as
        integers, e.g., for indexing purposes, but which may not carry integer
        type.
    atol : float
        Absolute tolerance away from nearest integer to consider values in
        ``val`` functionally integers.

    Returns
    -------
    val_int : NumPy scalar or ndarray of dtype `np.int64`
        Returns the input value(s) coerced to dtype `np.int64` assuming all
        were within ``atol`` of the nearest integer.

    Notes
    -----
    This operation calculates ``val`` modulo 1, which returns the mantissa of
    all values. Then all mantissas greater than 0.5 are subtracted from one.
    Finally, the absolute tolerance from zero is calculated. If it is less
    than ``atol`` for all value(s) in ``val``, they are rounded and returned
    in an integer array. Or, if ``val`` was a scalar, a NumPy scalar type is
    returned.

    If any value(s) are outside the specified tolerance, an informative error
    is raised.

    Examples
    --------
    >>> safe_as_int(7.0)
    7

    >>> safe_as_int([9, 4, 2.9999999999])
    array([9, 4, 3])

    >>> safe_as_int(53.1)
    Traceback (most recent call last):
        ...
    ValueError: Integer argument required but received 53.1, check inputs.

    >>> safe_as_int(53.01, atol=0.01)
    53

    """
    mod = np.asarray(val) % 1                # Extract mantissa

    # Check for and subtract any mod values > 0.5 from 1
    if mod.ndim == 0:                        # Scalar input, cannot be indexed
        if mod > 0.5:
            mod = 1 - mod
    else:                                    # Iterable input, now ndarray
        mod[mod > 0.5] = 1 - mod[mod > 0.5]  # Test on each side of nearest int

    try:
        np.testing.assert_allclose(mod, 0, atol=atol)
    except AssertionError:
        raise ValueError(f'Integer argument required but received '
                         f'{val}, check inputs.')

    return np.round(val).astype(np.int64)


def check_shape_equality(*images):
    image0 = images[0]
    if not all(image0.shape == image.shape for image in images[1:]):
        raise ValueError('Input images must have the same dimensions.')
    return


def slice_at_axis(sl, axis):
    return (slice(None),) * axis + (sl,) + (...,)


def reshape_nd(arr, ndim, dim):
    if arr.ndim != 1:
        raise ValueError("arr must be a 1D array")
    new_shape = [1] * ndim
    new_shape[dim] = -1
    return np.reshape(arr, new_shape)


def check_nD(array, ndim, arg_name='image'):
    array = np.asanyarray(array)
    msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
    msg_empty_array = "The parameter `%s` cannot be an empty array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if array.size == 0:
        raise ValueError(msg_empty_array % (arg_name))
    if array.ndim not in ndim:
        raise ValueError(
            msg_incorrect_dim % (arg_name, '-or-'.join([str(n) for n in ndim]))
        )


def convert_to_float(image, preserve_range):

    if image.dtype == np.float16:
        return image.astype(np.float32)
    if preserve_range:

        if image.dtype.char not in 'df':
            image = image.astype(float)
    else:
        from dtype import img_as_float
        image = img_as_float(image)
    return image


def _validate_interpolation_order(image_dtype, order):

    if order is None:
        return 0 if image_dtype == bool else 1

    if order < 0 or order > 5:
        raise ValueError("Spline interpolation order has to be in the "
                         "range 0-5.")

    if image_dtype == bool and order != 0:
        raise ValueError(
            "Input image dtype is bool. Interpolation is not defined "
            "with bool data type. Please set order to 0 or explicitly "
            "cast input image to another data type.")

    return order


def _to_np_mode(mode):
    mode_translation_dict = dict(nearest='edge', reflect='symmetric',
                                 mirror='reflect')
    if mode in mode_translation_dict:
        mode = mode_translation_dict[mode]
    return mode


def _to_ndimage_mode(mode):
    mode_translation_dict = dict(constant='constant', edge='nearest',
                                 symmetric='reflect', reflect='mirror',
                                 wrap='wrap')
    if mode not in mode_translation_dict:
        raise ValueError(
            f"Unknown mode: '{mode}', or cannot translate mode. The "
             f"mode should be one of 'constant', 'edge', 'symmetric', "
             f"'reflect', or 'wrap'. See the documentation of numpy.pad for "
             f"more info.")
    return _fix_ndimage_mode(mode_translation_dict[mode])


def _fix_ndimage_mode(mode):
    grid_modes = {'constant': 'grid-constant', 'wrap': 'grid-wrap'}
    return grid_modes.get(mode, mode)


new_float_type = {
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    np.float16().dtype.char: np.float32,
    'g': np.float64,      # np.float128 ; doesn't exist on windows
    'G': np.complex128,   # np.complex256 ; doesn't exist on windows
}


def _supported_float_type(input_dtype, allow_complex=False):

    if isinstance(input_dtype, Iterable) and not isinstance(input_dtype, str):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)


def identity(image, *args, **kwargs):
    return image


def as_binary_ndarray(array, *, variable_name):
    array = np.asarray(array)
    if array.dtype != bool:
        if np.any((array != 1) & (array != 0)):
            raise ValueError(
                f"{variable_name} array is not of dtype boolean or "
                f"contains values other than 0 and 1 so cannot be "
                f"safely cast to boolean array."
            )
    return np.asarray(array, dtype=bool)
