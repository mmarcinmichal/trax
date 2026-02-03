# coding=utf-8
# Copyright 2026 The Trax Authors.
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

"""JAX runtime helpers for supervised training/eval."""

import jax
import numpy as np

from trax import fastmath
from trax.fastmath import numpy as jnp


def reshape_by_device(x, n_devices, pure_np=False):
    def f(y):
        y_arr = np.asarray(y) if pure_np else y
        y_shape = list(y_arr.shape)
        batch_size = y_shape[0]
        batch_size_per_device = batch_size // n_devices
        if batch_size_per_device * n_devices != batch_size:
            raise ValueError(
                "Number of devices (%d) does not evenly divide batch size (%d)."
                % (n_devices, batch_size)
            )
        new_shape = [n_devices, batch_size_per_device] + y_shape[1:]
        if pure_np:
            return np.reshape(y_arr, new_shape)
        return jnp.reshape(y_arr, new_shape)

    return fastmath.nested_map(f, x)


def combine_devices(x_tuple):
    def f(x):
        if len(x.shape) < 2:
            return x
        batch_size = x.shape[0] * x.shape[1]
        return jnp.reshape(x, [batch_size] + list(x.shape[2:]))

    return fastmath.nested_map(f, x_tuple)


def for_n_devices(x, n_devices):
    if n_devices <= 1:
        return x

    def f(y):
        y_np = np.asarray(y)
        return np.broadcast_to(y_np, (n_devices,) + y_np.shape)

    return fastmath.nested_map(f, x)


def on_cpu(x):
    def f(y):
        if isinstance(y, list):
            return [f(v) for v in y]
        if isinstance(y, tuple):
            return tuple(f(v) for v in y)
        if isinstance(y, dict):
            return {k: f(v) for k, v in y.items()}
        if hasattr(y, "copy_to_host_async"):
            y.copy_to_host_async()
        return np.asarray(y)

    return fastmath.nested_map(f, x)


def on_accelerator(x):
    return fastmath.nested_map(lambda y: jnp.asarray(y), x)


def jit_forward_for_eval(forward, n_devices, do_mean=True):
    if n_devices == 0:
        return fastmath.jit(forward, device=jax.devices("cpu")[0])
    if n_devices < 2:
        return fastmath.jit(forward)

    model_predict = fastmath.pmap(forward, axis_name="batch")

    def predict(x, weights, state, rng):
        res, state = model_predict(
            reshape_by_device(x, n_devices),
            weights,
            state,
            jnp.stack(fastmath.random.split(rng, n_devices)),
        )
        res = combine_devices(res)
        if do_mean:
            return fastmath.nested_map(lambda y: jnp.mean(y, axis=0), res), state
        return res, state

    return predict


class JittedLayer:
    """Layer wrapper that dispatches through a JIT/PMAP forward pass."""

    def __init__(self, layer, n_devices=None, do_mean=False):
        self._layer = layer
        self._n_devices = n_devices or fastmath.local_device_count()
        self._forward = jit_forward_for_eval(
            layer.pure_fn, self._n_devices, do_mean=do_mean
        )

    def __call__(self, x):
        y, new_state = self._forward(
            x, self._layer.weights, self._layer.state, self._layer.rng
        )
        self._layer.state = new_state
        return y

    def init(self, *args, **kwargs):
        return self._layer.init(*args, **kwargs)

    @property
    def weights(self):
        return self._layer.weights

    @weights.setter
    def weights(self, value):
        self._layer.weights = value

    @property
    def state(self):
        return self._layer.state

    @state.setter
    def state(self, value):
        self._layer.state = value

    @property
    def rng(self):
        return self._layer.rng

    @rng.setter
    def rng(self, value):
        self._layer.rng = value

    def __getattr__(self, name):
        return getattr(self._layer, name)


def wrap_layer_for_eval(layer, n_devices=None, do_mean=False):
    return JittedLayer(layer, n_devices=n_devices, do_mean=do_mean)
