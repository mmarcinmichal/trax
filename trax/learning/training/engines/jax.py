# Refactored trainers.py
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import psutil

import trax.layers as tl

from trax import fastmath
from trax.layers import combinators as cb
from trax.learning.training.utils import runtime
from trax.utils import logging as trax_logging


def _adasum_merge(a, b):
    """Compute the AdaSum of two vectors."""
    dot_val = jnp.vdot(a, b)
    a_sq = jnp.vdot(a, a)
    b_sq = jnp.vdot(b, b)
    # Handle zero-norm edge cases
    if a_sq == 0 or b_sq == 0:
        return a + b
    gamma = a_sq / (a_sq + b_sq)
    # If dot < 0, combine them scaled by gamma; else just add.
    return gamma * a + (1.0 - gamma) * b if dot_val < 0 else a + b


def _average_multidevice_gradients(gradients, adasum=False):
    """
    Averages (or Adasum-reduces) 'gradients' across devices using the axis_name='batch'.

    If adasum=False, we do a standard pmean.
    If adasum=True, we do a simple all_gather & reduce approach, for demonstration.
    """
    if not adasum:
        # Standard average via pmean
        return jax.lax.pmean(gradients, axis_name="batch")
    else:
        # Demonstration: gather all grads to each device, then reduce them.
        # (A real Adasum might do ring-based or hierarchical merges.)
        gathered = jax.lax.all_gather(gradients, axis_name="batch")

        # gathered.shape now has an extra leading dimension [n_devices].
        # We'll do a simple tree_map to accumulate them one by one.
        def adasum_reduce(g_list):
            acc = g_list[0]
            for g in g_list[1:]:
                acc = jax.tree_map(_adasum_merge, acc, g)
            return acc

        # Because we used all_gather, 'gathered' is shaped like [n_devices, ...] for each leaf
        # So we need to pass that list of leaves to adasum_reduce.
        # We'll do a small helper to slice along the 0th dimension:
        n_devices = (
            gathered[0].shape[0] if isinstance(gathered, tuple) else gathered.shape[0]
        )

        # flatten out the leading dimension for each leaf
        # to produce a python list we can fold over:
        def gather_to_list(x):
            # x shape is (n_devices, ...) -> list of n_devices leaves
            return [x[i] for i in range(n_devices)]

        # Now do adasum reduction leaf-by-leaf:
        return jax.tree_map(
            lambda arrs: adasum_reduce(arrs), jax.tree_map(gather_to_list, gathered)
        )


def _pad_batch_for_devices(batch, n_devices):
    """
    If batch_size is not divisible by n_devices, pad the leading dimension so it is.
    Returns (padded_batch, unpad_amount).

    'batch' should be a tuple/list of arrays, or a PyTree that includes arrays
    on the leading dimension for each item in the batch.
    """
    batch_size = batch[0].shape[0]  # assume batch is e.g. (input, target, ...)
    remainder = batch_size % n_devices
    if remainder == 0:
        return batch, 0

    new_size = math.ceil(batch_size / n_devices) * n_devices
    to_pad = new_size - batch_size

    def pad_fn(x):
        # x has shape [batch_size, ...]
        return jnp.pad(x, [(0, to_pad)] + [(0, 0)] * (x.ndim - 1), mode="constant")

    padded = jax.tree_map(pad_fn, batch)
    return padded, to_pad


def _unpad_batch_outputs(outputs, to_remove):
    """
    If we padded the batch by 'to_remove' examples, remove them from
    the leading dimension of the returned arrays.
    """
    if to_remove == 0:
        return outputs

    def unpad_fn(x):
        # x has leading dimension we want to slice off the last 'to_remove' elements
        return x[:-to_remove] if x.shape[0] > to_remove else x[:0]

    return jax.tree_map(unpad_fn, outputs)


def _accelerate_update_fn(forward_and_backward_fn, optimizer, n_devices, adasum):
    """
    Returns an update_fn that:
      - single-device => jitted function
      - multi-device => pmapped function that also does gradient averaging or Adasum
    """

    @jax.jit
    def single_device_update_fn(
        weights, state, opt_state, batch, rng, step_int, opt_params
    ):
        grads, loss, updated_state = forward_and_backward_fn(batch, weights, state, rng)

        new_weights, new_opt_state, metrics = optimizer.tree_update(
            step_int, grads, weights, opt_state, opt_params, store_slots=False
        )
        metrics["loss"] = loss
        return new_weights, updated_state, new_opt_state, metrics

    if n_devices <= 1:
        # Single device => just call the jitted function
        return single_device_update_fn

    # For multi-device: we pmap around single_device_update_fn
    def multi_device_update_fn(
        weights, state, opt_state, batch, rngs, step_int, opt_params
    ):
        """
        Each device runs single_device_update_fn on a shard of the batch,
        then we do gradient averaging (or Adasum).
        """

        def _per_device_step(w, s, o, b, r):
            """
            We do the forward/backward but also average grads across devices
            inside this pmap, so each device ends up with the same update.
            """
            grads, loss, st_new = forward_and_backward_fn(b, w, s, r)
            grads = _average_multidevice_gradients(grads, adasum=adasum)
            w_new, o_new, metrics = optimizer.tree_update(
                step_int, grads, w, o, opt_params, store_slots=False
            )
            metrics["loss"] = loss
            return w_new, st_new, o_new, metrics

        # We call pmap over the per-device-step
        w_updated, s_updated, o_updated, metrics = jax.pmap(
            _per_device_step, axis_name="batch"
        )(weights, state, opt_state, batch, rngs)
        return w_updated, s_updated, o_updated, metrics

    return multi_device_update_fn


class TrainingEngine:
    """A trainers that supports single- or multi-device, with optional Adasum, padding, etc."""

    def __init__(self, model_with_loss, optimizer, n_devices=None, adasum=False):
        """
        Args:
            model_with_loss: A layer that returns (loss, new_state) from pure_fn(...)
            optimizer: An optimizer with .tree_init(...) and .tree_update(...) methods
            n_devices: Number of devices to use
            adasum: Whether to do Adasum gradient reduction (instead of standard averaging)
        """
        self._model_with_loss = model_with_loss
        self._optimizer = optimizer
        self._n_devices = n_devices or jax.local_device_count()
        self._adasum = adasum

        # Initialize optimizer state from the model's initial weights
        self._slots, self._opt_params = optimizer.tree_init(
            self._model_with_loss.weights
        )

        # Build forward+backward function with value_and_grad(has_aux=True)
        def forward_and_backward_fn(batch, weights, state, rng):
            """
            Returns (gradients, loss, new_state).
            """

            def loss_fn(curr_w, curr_s):
                # Avoid caching intermediates during the backward pass, since cache
                # storage can capture tracers under JIT and trigger tracer leak
                # errors. Training recomputes values as needed, so we keep
                # use_cache disabled here.
                loss_val, new_st = model_with_loss.pure_fn(
                    batch, curr_w, curr_s, rng, use_cache=False
                )
                return loss_val, new_st

            (loss_val, new_state), grads = jax.value_and_grad(
                loss_fn, argnums=0, has_aux=True
            )(weights, state)

            return grads, loss_val, new_state

        self._forward_and_backward_fn = forward_and_backward_fn

        # Build an update function that does single vs. multi-device
        self._accelerated_update_fn = _accelerate_update_fn(
            self._forward_and_backward_fn,
            self._optimizer,
            self._n_devices,
            self._adasum,
        )

    @property
    def model_with_loss(self):
        return self._model_with_loss

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def slots(self):
        return self._slots

    @slots.setter
    def slots(self, slots):
        self._slots = slots
        self._optimizer.slots = slots

    def one_step(self, batch, rng, step=0, learning_rate=None):
        """
        1) Possibly pad the batch for multi-device
        2) Single- or multi-device forward/backward
        3) Update weights & state
        4) Unpad if needed, return loss
        """
        if learning_rate is not None:
            self._opt_params["learning_rate"] = learning_rate

        weights = self._model_with_loss.weights
        state = self._model_with_loss.state

        if self._n_devices == 1:
            # Single device => just run the function directly (already jitted).
            (new_weights, new_state, new_slots, stats,) = self._accelerated_update_fn(
                weights,
                state,
                self._slots,
                batch,
                rng,
                step,
                self._opt_params,
            )

            # Store
            self._model_with_loss.weights = new_weights
            self._model_with_loss.state = new_state
            self._slots = new_slots
            self._optimizer.slots = new_slots
            loss_value = float(stats["loss"]) if np.size(stats["loss"]) == 1 else float(
                np.mean(stats["loss"])
            )
            stats["loss"] = loss_value
            return loss_value, stats

        #
        # Multi-device => pad the batch if needed, replicate, call pmapped update
        #
        padded_batch, to_remove = _pad_batch_for_devices(batch, self._n_devices)
        padded_size = padded_batch[0].shape[0]
        batch_per_device = padded_size // self._n_devices

        # Split rng if it's just a single key
        if isinstance(rng, np.ndarray) and rng.shape == (2,):
            rng = jax.random.split(rng, self._n_devices)

        # Reshape batch for devices
        padded_batch = jax.tree_map(
            lambda x: x.reshape((self._n_devices, batch_per_device) + x.shape[1:]),
            padded_batch,
        )

        # Replicate weights/state/slots
        weights_rep = jax.tree_map(
            lambda x: np.broadcast_to(x, (self._n_devices,) + x.shape), weights
        )
        state_rep = jax.tree_map(
            lambda x: np.broadcast_to(x, (self._n_devices,) + x.shape), state
        )
        slots_rep = jax.tree_map(
            lambda x: np.broadcast_to(x, (self._n_devices,) + x.shape), self._slots
        )

        # Run the pmapped update
        (
            updated_weights_rep,
            updated_state_rep,
            updated_slots_rep,
            stats_rep,
        ) = self._accelerated_update_fn(
            weights_rep, state_rep, slots_rep, padded_batch, rng, step, self._opt_params
        )

        # Unreplicate results
        new_weights = self._unreplicate(updated_weights_rep)
        new_state = self._unreplicate(updated_state_rep)
        new_slots = self._unreplicate(updated_slots_rep)
        stats = self._unreplicate(stats_rep)
        loss_vals = stats["loss"]
        final_loss = float(loss_vals) if np.size(loss_vals) == 1 else np.mean(loss_vals)

        # Update trainers
        self._model_with_loss.weights = new_weights
        self._model_with_loss.state = new_state
        self._slots = new_slots
        self._optimizer.slots = new_slots

        # If your model returns per-example losses, you might want to unpad the output
        # after the forward pass. But here we've just got a scalar loss, so no unpadding needed
        # for the loss. If you needed to unpad e.g. a predictions array, you'd do it here.

        stats["loss"] = final_loss
        return final_loss, stats

    def _unreplicate(self, tree):
        """Return the first element of a replicated array (from shape [n_devices,...] to [...])."""
        return jax.tree_map(lambda x: x[0], tree)


class ReversibleSerialTrainer:
    """Runs an optimizer on a series of layers, reversible and not."""

    def __init__(
        self,
        blocks,
        loss_layer,
        optimizer_fn,
        n_devices=None,
        memoize_jit=True,
        free_accelerators_on_step=False,
        adasum=False,
    ):
        self._blocks = [(tl.Serial(std), rev) for (std, rev) in blocks]
        self._loss_layer = loss_layer
        self._optimizer_fn = optimizer_fn
        self._n_devices = n_devices or fastmath.local_device_count()
        self._adasum = adasum
        self._n_layers = 1 + sum([len(revs) + 1 for (_, revs) in self._blocks])
        self._n_steps_per_log = 100
        self._n_async_layers = 1
        self._jit_memory = {} if memoize_jit else None
        self._do_free = free_accelerators_on_step
        self._jit_per_device_rngs = fastmath.jit(self._per_device_rngs, backend="cpu")

        self._accelerated_layer_fns = fastmath.nested_map(
            lambda layer: self._pjit(layer.pure_fn, f"fwd {repr(layer)}"), self._blocks
        )

        def _make_optimizer(layer):
            opt = optimizer_fn()
            opt.tree_init(layer.weights)
            opt.slots = runtime.on_cpu(opt.slots)
            return opt

        self._optimizers = fastmath.nested_map(_make_optimizer, self._blocks)
        self._replicated_opt_params = fastmath.nested_map(
            lambda opt: self._replicate_cpu(opt.opt_params), self._optimizers
        )

        self._loss_opt = _make_optimizer(loss_layer)
        self._replicated_loss_opt_params = self._replicate_cpu(
            self._loss_opt.opt_params
        )

        self._fbos = []
        for i, (std_layer, rev_layers) in enumerate(self._blocks):
            (std_opt, rev_opts) = self._optimizers[i]
            std_fbo = _fbo_with_layer_and_opt(
                std_layer, std_opt, self._n_devices, adasum=self._adasum
            )
            rev_and_fbos = []
            for layer, opt in zip(rev_layers, rev_opts):
                rev_and_fbo = _reverse_and_fbo_with_layer_and_opt(
                    layer, opt, self._n_devices, self._adasum
                )
                rev_and_fbos.append(
                    self._pjit(
                        rev_and_fbo, f"rev+bwd {repr(layer)}", donate_argnums=(0, 1, 2)
                    )
                )
            jit_std_fbo = self._pjit(
                std_fbo, f"bwd {repr(std_layer)}", donate_argnums=(1, 2)
            )
            self._fbos.append((jit_std_fbo, rev_and_fbos))

        loss_fbo = _fbo_with_layer_and_opt(
            self._loss_layer, self._loss_opt, self._n_devices, "loss", self._adasum
        )
        self._loss_fbo = self._pjit(loss_fbo, donate_argnums=(1, 2))

    @property
    def loss_layer(self):
        return self._loss_layer

    @property
    def all_layers(self):
        layers = []
        for (std_layer, rev_layers) in self._blocks:
            layers.append(std_layer)
            layers.extend(rev_layers)
        layers.append(self._loss_layer)
        return layers

    @property
    def optimizer_fn(self):
        return self._optimizer_fn

    @property
    def slots(self):
        optimizers = list(self._optimizers) + [self._loss_opt]
        return fastmath.nested_map(lambda opt: opt.slots, optimizers)

    @slots.setter
    def slots(self, slots):
        for ((s_opt, r_opts), (s_slots, r_slots)) in zip(self._optimizers, slots[:-1]):
            for (opt, slot) in zip([s_opt] + r_opts, [s_slots] + r_slots):
                opt.slots = slot
        self._loss_opt.slots = slots[-1]

    def _pjit(self, f, memory_key=None, donate_argnums=()):
        should_memoize = self._jit_memory is not None and memory_key is not None
        if should_memoize and memory_key in self._jit_memory:
            trax_logging.info("Found JITed function in memory for: %s", memory_key)
            return self._jit_memory[memory_key]
        if self._n_devices == 1:
            res = fastmath.jit(f, donate_argnums=donate_argnums)
        else:
            res = fastmath.pmap(f, axis_name="batch", donate_argnums=donate_argnums)
        if should_memoize:
            self._jit_memory[memory_key] = res
        return res

    def _replicate(self, x):
        if self._n_devices > 1:
            return runtime.for_n_devices(x, self._n_devices)
        return runtime.on_accelerator(x)

    def _replicate_cpu(self, x):
        def f(x):
            if self._n_devices > 1:
                return np.broadcast_to(x, (self._n_devices,) + np.asarray(x).shape)
            else:
                return x

        return runtime.on_cpu(fastmath.nested_map(f, x))

    def _unreplicate(self, x):
        if self._n_devices == 1:
            return runtime.on_cpu(x)
        return runtime.on_cpu(fastmath.nested_map(lambda x: x[0], x))

    def _lazy_unreplicate(self, x):
        def unreplicate_and_start_async_copy(y):
            unreplicated = y if self._n_devices == 1 else y[0]
            unreplicated.copy_to_host_async()
            return unreplicated

        return fastmath.nested_map(unreplicate_and_start_async_copy, x)

    def _collect_weights(self, layer):
        layer.weights = fastmath.nested_map(np.asarray, layer.weights)

    def _free_accelerators(self, exceptions=(), keep_constants=True):
        backend = jax.lib.xla_bridge.get_backend()
        live_buffers = backend.live_buffers()
        trax_logging.info("Deleting %d live buffers.", len(live_buffers))
        exceptions_buffers = []
        for x in fastmath.tree_flatten(exceptions):
            if hasattr(x, "device_buffer"):
                exceptions_buffers.append(x.device_buffer)
            if hasattr(x, "device_buffers"):
                exceptions_buffers.extend(x.device_buffers)
        for b in live_buffers:
            should_delete = True
            for e in exceptions_buffers:
                if b is e:
                    should_delete = False
            if keep_constants and not b.shape:
                should_delete = False
            if should_delete:
                b.delete()

    def _per_device_rngs(self, rng):
        per_device_rng = fastmath.random.split(rng, self._n_devices)
        per_device_rngs = [
            fastmath.random.split(r, self._n_layers) for r in per_device_rng
        ]
        rngs = [
            jnp.stack([r[i] for r in per_device_rngs]) for i in range(self._n_layers)
        ]
        return rngs

    def one_step(self, batch, rng, step=0, learning_rate=None):
        if learning_rate is not None:
            self._replicated_loss_opt_params["learning_rate"] = self._replicate_cpu(
                learning_rate
            )
            for (std_op, rev_ops) in self._replicated_opt_params:
                std_op["learning_rate"] = self._replicate_cpu(learning_rate)
                for op in rev_ops:
                    op["learning_rate"] = self._replicate_cpu(learning_rate)

        step_int = step
        if self._n_devices > 1:
            batch = runtime.reshape_by_device(
                batch, self._n_devices, pure_np=True
            )
            step = np.repeat(step, self._n_devices)

        if self._n_devices == 1:
            rngs = fastmath.random.split(rng, self._n_layers)
        else:
            rngs = self._jit_per_device_rngs(runtime.on_cpu(rng))
        rng_blocks, rng_i = [], 0
        for _, rev_layers in self._blocks:
            l = len(rev_layers)
            rng_blocks.append((rngs[rng_i], rngs[rng_i + 1 : rng_i + l + 1]))
            rng_i += l + 1

        if self._do_free:
            self._free_accelerators()
        process = psutil.Process(os.getpid())
        if isinstance(batch, (list, tuple)):
            batch_shapes = [x.shape for x in batch]
        else:
            batch_shapes = batch.shape
        trax_logging.info("running step %d on shapes %s", step_int, str(batch_shapes))
        if step_int % self._n_steps_per_log == 1:
            trax_logging.info(
                "run fwd: cpu memory use (MB): %.2f",
                process.memory_info().rss / float(1024 * 1024),
            )

        stack = batch
        block_inputs_states = []
        for i, (std_layer, rev_layers) in enumerate(self._blocks):
            acc_std_layer_fn, acc_rev_layer_fns = self._accelerated_layer_fns[i]
            std_rng, rev_rngs = rng_blocks[i]
            stack, std_inputs, std_state = self._run_forward_standard(
                stack, std_layer, acc_std_layer_fn, std_rng, step_int
            )
            stack, rev_old_states, rev_new_states = self._run_forward_reversible(
                stack, rev_layers, acc_rev_layer_fns, rev_rngs, step_int
            )
            block_inputs_states.append(
                runtime.on_cpu(
                    ((std_inputs, std_state), (rev_old_states, rev_new_states))
                )
            )

        if step_int % self._n_steps_per_log == 1:
            trax_logging.info(
                "run loss: cpu memory use (MB): %.2f",
                process.memory_info().rss / float(1024 * 1024),
            )
        loss_state = self._replicate(self._loss_layer.state)
        loss_inputs = cb.inputs_from_stack(stack, self._loss_layer.n_in)
        loss_stats, grad_stack = self._run_backward_standard(
            None,
            step,
            self._loss_layer,
            loss_inputs,
            loss_state,
            self._loss_fbo,
            rngs[-1],
            self._loss_opt,
            self._replicated_loss_opt_params,
        )
        self._collect_weights(self._loss_layer)
        stats = [runtime.on_cpu(loss_stats)]

        if self._do_free:
            stack, grad_stack = runtime.on_cpu(stack), runtime.on_cpu(grad_stack)
            self._free_accelerators()

        if step_int % self._n_steps_per_log == 1:
            trax_logging.info(
                "run bwd: cpu memory use (MB): %.2f",
                process.memory_info().rss / float(1024 * 1024),
            )
        for i in range(len(self._blocks) - 1, -1, -1):
            std_layer, rev_layers = self._blocks[i]
            (std_inputs, std_state), (
                rev_old_states,
                rev_new_states,
            ) = block_inputs_states[i]
            std_fbo, rev_fbos = self._fbos[i]
            std_opt, rev_opts = self._optimizers[i]
            std_rng, rev_rngs = rng_blocks[i]
            repl_std_opt_params, repl_rev_opts_params = self._replicated_opt_params[i]

            stack, grad_stack, new_stats = self._run_backward_reversible(
                stack,
                grad_stack,
                step,
                rev_layers,
                rev_fbos,
                rev_old_states,
                rev_new_states,
                rev_rngs,
                rev_opts,
                repl_rev_opts_params,
            )
            stats.extend(runtime.on_cpu(new_stats))

            std_layer_stats, grad_stack = self._run_backward_standard(
                grad_stack,
                step,
                std_layer,
                std_inputs,
                std_state,
                std_fbo,
                std_rng,
                std_opt,
                repl_std_opt_params,
            )
            stack = cb.outputs_onto_stack(std_inputs, stack, std_layer.n_out)
            stats.append(runtime.on_cpu(std_layer_stats))

            for rev_layer_id in range(self._n_async_layers):
                self._collect_weights(rev_layers[rev_layer_id])
            self._collect_weights(std_layer)

        joint_stats = {}
        for i, stat in enumerate(reversed(stats)):
            for k, v in stat.items():
                joint_stats[f"layer{i}/" + k] = v
        return stats[0]["loss"], joint_stats

    def _run_forward_standard(self, stack, layer, accelerated_fn, rng, step):
        if step % self._n_steps_per_log == 1:
            trax_logging.info("running forward standard layer %s", str(layer))
        layer_inputs = cb.inputs_from_stack(stack, layer.n_in)
        layer_weights = self._replicate(layer.weights)
        layer_state = self._replicate(layer.state)
        outputs, layer_new_state = accelerated_fn(
            layer_inputs, layer_weights, layer_state, rng
        )
        stack = cb.outputs_onto_stack(outputs, stack, layer.n_in)
        return stack, layer_inputs, layer_new_state

    def _run_forward_reversible(self, stack, rev_layers, accelerated_fns, rngs, step):
        old_states, new_states = [], []
        for i, layer in enumerate(rev_layers):
            if step % self._n_steps_per_log == 1:
                trax_logging.info("running forward reversible layer %s", str(layer))
            weights = self._replicate(layer.weights)
            state = self._replicate(layer.state)
            old_states.append(state)
            inputs = cb.inputs_from_stack(stack, layer.n_in)
            outputs, new_state = accelerated_fns[i](inputs, weights, state, rngs[i])
            stack = cb.outputs_onto_stack(outputs, stack, layer.n_in)
            new_states.append(new_state)
        return stack, old_states, new_states

    def _run_backward_standard(
        self,
        grad_stack,
        step,
        layer,
        inp,
        state,
        fbo_fn,
        rng,
        optimizer,
        replicated_opt_params,
    ):
        step_int = int(step) if self._n_devices < 2 else int(step[0])
        if step_int % self._n_steps_per_log == 1:
            trax_logging.info("running backward standard layer %s", str(layer))
        if grad_stack is not None:
            grads = cb.inputs_from_stack(grad_stack, layer.n_out)
        else:
            grads = None
        slots = self._replicate(optimizer.slots)
        weights = self._replicate(layer.weights)
        state = runtime.on_accelerator(state)
        replicated_opt_params = runtime.on_accelerator(replicated_opt_params)
        rng = runtime.on_accelerator(rng)
        grads = runtime.on_accelerator(grads)
        inp = runtime.on_accelerator(inp)
        new_weights, new_state, new_slots, new_grads, stats = fbo_fn(
            inp, weights, grads, state, slots, replicated_opt_params, rng, step
        )
        layer.weights = self._lazy_unreplicate(new_weights)
        layer.state = self._unreplicate(new_state)
        optimizer.slots = self._unreplicate(new_slots)
        if grad_stack is not None:
            grad_stack = cb.outputs_onto_stack(new_grads, grad_stack, layer.n_out)
        else:
            grad_stack = new_grads
        return stats, grad_stack

    def _run_backward_reversible(
        self,
        stack,
        grad_stack,
        step,
        rev_layers,
        rev_and_fbos,
        old_states,
        new_states,
        rngs,
        optimizers,
        replicated_opt_params,
    ):
        counter = 0
        stats = []
        step_int = int(step) if self._n_devices < 2 else int(step[0])
        for layer, reverse_and_fbo, old_state, new_state, rng in reversed(
            list(zip(rev_layers, rev_and_fbos, old_states, new_states, rngs))
        ):
            if step_int % self._n_steps_per_log == 1:
                trax_logging.info("running backward reversible layer %s", str(layer))
            counter -= 1
            stack, grad_stack, layer_stats = self._run_backward_one_reversible(
                layer,
                stack,
                grad_stack,
                step,
                rng,
                optimizers[counter],
                replicated_opt_params[counter],
                reverse_and_fbo,
                old_state,
                new_state,
            )
            stats.append(layer_stats)
            if counter + self._n_async_layers < 0:
                self._collect_weights(rev_layers[counter + self._n_async_layers])
        return stack, grad_stack, stats

    def _run_backward_one_reversible(
        self,
        layer,
        stack,
        grad_stack,
        step,
        rng,
        optimizer,
        opt_params,
        reverse_and_fbo,
        old_state,
        new_state,
    ):
        outputs = cb.inputs_from_stack(stack, layer.n_out)
        grads = cb.inputs_from_stack(grad_stack, layer.n_out)
        slots = self._replicate(optimizer.slots)
        weights = self._replicate(layer.weights)
        outputs = runtime.on_accelerator(outputs)
        grads = runtime.on_accelerator(grads)
        old_state = runtime.on_accelerator(old_state)
        new_state = runtime.on_accelerator(new_state)
        opt_params = runtime.on_accelerator(opt_params)
        rng = runtime.on_accelerator(rng)
        new_weights, new_slots, inputs, grads, layer_stats = reverse_and_fbo(
            outputs, weights, grads, old_state, new_state, slots, opt_params, rng, step
        )
        layer.weights = self._lazy_unreplicate(new_weights)
        layer.state = self._unreplicate(new_state)
        optimizer.slots = self._unreplicate(new_slots)
        stack = cb.outputs_onto_stack(inputs, stack, layer.n_out)
        grad_stack = cb.outputs_onto_stack(grads, grad_stack, layer.n_out)
        return stack, grad_stack, layer_stats


def _fbo_with_layer_and_opt(layer, optimizer, n_devices, stats_name=None, adasum=False):
    def fbo(inputs, weights, grads, state, slots, opt_params, rng, step):
        def pure_fn_without_state_and_rng(x, w):
            return layer.pure_fn(x, w, state, rng)

        activations, vjp_fn, new_state = fastmath.vjp(
            pure_fn_without_state_and_rng, inputs, weights, has_aux=True
        )

        if grads is None and stats_name is not None:
            grads = jnp.ones((), dtype=activations.dtype)

        grads_inputs, grads_weights = vjp_fn(grads)

        if _is_empty_tuple(weights):
            stats = {}
            if stats_name is not None:
                stats[stats_name] = activations
            return weights, new_state, slots, grads_inputs, stats

        if n_devices > 1:
            grads_weights = _average_multidevice_gradients(
                grads_weights, adasum=adasum
            )

        new_weights, new_slots, stats = optimizer.tree_update(
            step, grads_weights, weights, slots, opt_params, store_slots=False
        )
        if stats_name is not None:
            stats[stats_name] = activations
        return new_weights, new_state, new_slots, grads_inputs, stats

    return fbo


def _reverse_and_fbo_with_layer_and_opt(layer, optimizer, n_devices, adasum):
    def reverse_and_fbo(
        output, weights, grads, state, new_state, slots, opt_params, rng, step
    ):
        inputs, (grads_inputs, grads_weights) = layer.reverse_and_grad(
            output, grads, weights, state, new_state, rng=rng
        )

        if _is_empty_tuple(weights):
            return weights, slots, inputs, grads_inputs, {}

        if n_devices > 1:
            grads_weights = _average_multidevice_gradients(
                grads_weights, adasum=adasum
            )

        new_weights, new_slots, stats = optimizer.tree_update(
            step, grads_weights, weights, slots, opt_params, store_slots=False
        )
        return new_weights, new_slots, inputs, grads_inputs, stats

    return reverse_and_fbo


def _is_empty_tuple(x):
    if not isinstance(x, (list, tuple)):
        return False
    for y in x:
        if not _is_empty_tuple(y):
            return False
    return True


def extract_reversible_blocks(layers, loss_chunk_size=0):
    def _flatten(l):
        if isinstance(l, (list, tuple)):
            return [x for layer in l for x in _flatten(layer)]
        elif isinstance(l, tl.Serial):
            return _flatten(l.sublayers)
        else:
            return [l]

    blocks, std_layers, rev_layers = [], [], []
    for layer in _flatten(layers):
        if isinstance(layer, tl.ReversibleLayer):
            rev_layers.append(layer)
        elif not rev_layers:
            std_layers.append(layer)
        else:
            blocks.append((std_layers, rev_layers))
            std_layers, rev_layers = [], []
            std_layers.append(layer)
    if rev_layers:
        raise ValueError("The final layer must be a standard loss, not reversible.")
    if loss_chunk_size > 0:
        border_layers = ["StripFromConcatenateWithPadding", "Select"]

        loss_start = None
        for index, layer in enumerate(std_layers):
            if layer.name in border_layers:
                loss_start = index + 1
        if loss_start is None:
            raise ValueError(
                "Loss layer should be preceeded by one of {}; got {}".format(
                    border_layers, [l.name for l in std_layers]
                )
            )
        if len(std_layers) - loss_start < 4:
            raise ValueError("Too short loss layer for chunking")
        last_3_names = " ".join([l.name for l in std_layers[-3:]])
        if last_3_names != "LogSoftmax _CrossEntropy _WeightedMean":
            raise ValueError(
                'Loss chunking only works with last layers being "'
                'LogSoftmax, _CrossEntropy, _WeightedMean" but got: ' + last_3_names
            )

        chunked_xent = tl.Chunk(tl.Serial(std_layers[loss_start:-1]), loss_chunk_size)

        def _reshape_to_batch_and_copy_targets(preds, targets):
            batched_preds = jnp.reshape(preds, [-1, preds.shape[-1]])
            batched_targets = jnp.reshape(targets, [-1])
            return batched_preds, batched_targets, targets

        def _reshape_xent_back(xent, targets):
            return jnp.reshape(xent, targets.shape)

        batched_xent = tl.Serial(
            tl.Fn("pre_xent_rebatch", _reshape_to_batch_and_copy_targets, n_out=3),
            chunked_xent,
            tl.Fn("after_xent_rebatch", _reshape_xent_back),
        )
        loss_layer = tl.Serial(std_layers[:loss_start] + [batched_xent], std_layers[-1])
    else:
        loss_layer = tl.Serial(std_layers)
    return blocks, loss_layer


def init_reversible_blocks(blocks, loss_layer, input_signature, rng):
    sig_stack = input_signature
    process = psutil.Process(os.getpid())
    mem_use = process.memory_info().rss
    for (std_layers, rev_layers) in blocks:
        rngs = fastmath.random.split(rng, len(std_layers) + len(rev_layers) + 1)
        rng = rngs[0]
        for layer, layer_rng in zip(std_layers + rev_layers, rngs[1:]):
            sig = cb.inputs_from_stack(sig_stack, layer.n_in)
            layer.init(sig, rng=layer_rng)
            layer.weights = runtime.on_cpu(layer.weights)
            layer.state = runtime.on_cpu(layer.state)
            trax_logging.info(
                "init: layer %s\nadded cpu memory (MB): %.2f",
                str(layer),
                (process.memory_info().rss - mem_use) / float(1024 * 1024),
            )
            mem_use = process.memory_info().rss
            trax_logging.info(
                "init: cpu memory use (MB): %.2f", mem_use / float(1024 * 1024)
            )
            out_sig = layer.output_signature(sig)
            sig_stack = cb.outputs_onto_stack(out_sig, sig_stack, layer.n_in)
    loss_layer.init(cb.inputs_from_stack(sig_stack, loss_layer.n_in), rng=rng)
    loss_layer.weights = runtime.on_cpu(loss_layer.weights)
    loss_layer.state = runtime.on_cpu(loss_layer.state)
