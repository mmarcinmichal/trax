# coding=utf-8
# Copyright 2022 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NumPy like wrapper for Tensorflow."""


# Enable NumPy behavior globally
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

# Make everything from tensorflow.experimental.numpy available
# Import all from tensorflow.experimental.numpy
from tensorflow import bfloat16
from tensorflow.experimental.numpy import random
from tensorflow.experimental.numpy import *  # pylint: disable=wildcard-import
import numpy as _onp
import tensorflow as _tf
import tensorflow.experimental.numpy as _tfnp
from tensorflow.python.ops.numpy_ops.np_dtypes import (
    canonicalize_dtype,
    default_float_type,
    is_allow_float64,
    set_allow_float64,
)

# Define what should be accessible when someone imports from this module
__all__ = [
    "bfloat16",
    "canonicalize_dtype",
    "default_float_type",
    "is_allow_float64",
    "set_allow_float64",
]


def clip(a, a_min=None, a_max=None, out=None):  # pylint: disable=redefined-builtin
    """NumPy-compatible clip with explicit dtype promotion.

    tf.experimental.numpy.clip can mis-handle Python int inputs under jit,
    especially when bounds include negatives, by casting to unsigned dtypes
    too early. Force all operands to a common result dtype to match NumPy.
    """
    if a_min is None and a_max is None:
        return _tfnp.asarray(a)

    args = [a]
    if a_min is not None:
        args.append(a_min)
    if a_max is not None:
        args.append(a_max)

    arg_dtypes = []
    for arg in args:
        if _tf.is_tensor(arg):
            arg_dtypes.append(_onp.dtype(arg.dtype.as_numpy_dtype))
        elif hasattr(arg, "dtype"):
            arg_dtypes.append(_onp.dtype(arg.dtype))
        else:
            arg_dtypes.append(_onp.asarray(arg).dtype)
    dtype = _onp.result_type(*arg_dtypes)

    a = _tfnp.asarray(a, dtype=dtype)
    if a_min is not None:
        a_min = _tfnp.asarray(a_min, dtype=dtype)
    if a_max is not None:
        a_max = _tfnp.asarray(a_max, dtype=dtype)

    # Use tf ops to avoid dtype surprises under jit for Python scalar inputs.
    if a_min is not None:
        a = _tf.maximum(a, a_min)
    if a_max is not None:
        a = _tf.minimum(a, a_max)
    if out is not None:
        out[...] = a
        return out
    return a


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    """NumPy-compatible linspace with integer dtype behavior."""
    is_int_dtype = dtype is not None and _onp.issubdtype(_onp.dtype(dtype), _onp.integer)
    linspace_dtype = _onp.float64 if is_int_dtype else dtype

    if is_int_dtype:
        start = _tf.cast(_tfnp.asarray(start), _tf.float64)
        stop = _tf.cast(_tfnp.asarray(stop), _tf.float64)
        shape = _tf.broadcast_dynamic_shape(_tf.shape(start), _tf.shape(stop))
        start = _tf.broadcast_to(start, shape)
        stop = _tf.broadcast_to(stop, shape)

        rank = _tf.size(shape)
        total_rank = rank + 1
        axis_ = axis if axis >= 0 else axis + total_rank
        div = num - 1 if endpoint else num
        if num > 1:
            step = (stop - start) / _tf.cast(div, _tf.float64)
        else:
            step = _tf.zeros_like(start)

        rng = _tf.cast(_tf.range(num), _tf.float64)
        ones_before = _tf.ones([axis_], dtype=_tf.int32)
        ones_after = _tf.ones([total_rank - axis_ - 1], dtype=_tf.int32)
        rng_shape = _tf.concat([ones_before, [num], ones_after], axis=0)
        rng = _tf.reshape(rng, rng_shape)

        values = _tf.expand_dims(start, axis_) + _tf.expand_dims(step, axis_) * rng
        values = _tf.floor(values)
        if endpoint and num > 1:
            values = _tf.concat(
                [
                    _tf.gather(values, _tf.range(num - 1), axis=axis_),
                    _tf.expand_dims(stop, axis_),
                ],
                axis=axis_,
            )

        values = _tf.cast(values, _tf.as_dtype(dtype))
        return (values, step) if retstep else values

    result = _tfnp.linspace(
        start,
        stop,
        num,
        endpoint=endpoint,
        retstep=retstep,
        dtype=linspace_dtype,
        axis=axis,
    )
    if retstep:
        values, step = result
    else:
        values, step = result, None

    if is_int_dtype:
        values = values.astype(dtype)

    return (values, step) if retstep else values
