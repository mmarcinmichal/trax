# coding=utf-8
# Copyright 2024 The Trax Authors.
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
"""Parity checks between Trax JAX trainers and PyTorch."""

from typing import Any

import jax
import numpy as np

from numpy import floating

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from absl.testing import absltest
from learning.training.engines import jax as trainers_jax

from trax import fastmath, optimizers
from trax import layers as tl


def _flatten_l2_norms(tree) -> list[floating[Any]]:
    return [np.linalg.norm(np.asarray(leaf)) for leaf in jax.tree_util.tree_leaves(tree)]


def _flatten_deltas(new_tree, old_tree) -> list[floating[Any]]:
    deltas = jax.tree_util.tree_map(lambda n, o: np.asarray(n) - np.asarray(o), new_tree, old_tree)
    return _flatten_l2_norms(deltas)


def _copy_weights(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), tree)


def _prepare_models(batch, learning_rate, n_devices, hidden_dim=4):
    if torch is None:
        raise absltest.SkipTest("PyTorch is required for trainer parity tests.")
    torch.manual_seed(0)
    input_dim = batch[0].shape[-1]
    torch_model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1, bias=True),
    )
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
    torch_loss = torch.nn.MSELoss()

    trax_model = tl.Serial(tl.Dense(hidden_dim), tl.Relu(), tl.Dense(1), tl.L2Loss())
    trax_model.init(batch, rng=fastmath.random.get_prng(0))

    torch_linears = [m for m in torch_model.modules() if isinstance(m, torch.nn.Linear)]
    trax_denses = [m for m in trax_model.sublayers if isinstance(m, tl.Dense)]
    if len(torch_linears) != len(trax_denses):
        raise ValueError(
            "Mismatched layer counts between torch and trax: "
            f"{len(torch_linears)} vs {len(trax_denses)}"
        )
    for torch_layer, trax_layer in zip(torch_linears, trax_denses):
        torch_weight = torch_layer.weight.detach().cpu().numpy().astype(np.float32)
        torch_bias = torch_layer.bias.detach().cpu().numpy().astype(np.float32)
        trax_layer.weights = (torch_weight.T, torch_bias)
    trax_model.weights = tuple(layer.weights for layer in trax_model.sublayers)

    trax_optimizer = optimizers.SGD(learning_rate)
    trax_optimizer.tree_init(trax_model.weights)
    trax_trainer = trainers_jax.TrainingEngine(trax_model, trax_optimizer, n_devices=n_devices)

    return torch_model, torch_optimizer, torch_loss, trax_trainer


def _run_step_parity(
    batch, learning_rate, warmup_steps, measured_steps, n_devices, atol, rtol
):
    torch_model, torch_optimizer, torch_loss, trax_trainer = _prepare_models(
        batch, learning_rate, n_devices
    )

    torch_inputs = torch.from_numpy(batch[0])
    torch_targets = torch.from_numpy(batch[1])

    rng = fastmath.random.get_prng(123)
    rngs = fastmath.random.split(rng, warmup_steps + measured_steps)

    trax_losses, torch_losses = [], []
    trax_grad_norms, torch_grad_norms = [], []
    trax_deltas, torch_deltas = [], []

    def trax_loss_and_grads(weights, state, rng_key):
        def loss_fn(curr_w):
            loss_val, _ = trax_trainer.model_with_loss.pure_fn(
                batch, curr_w, state, rng_key, use_cache=False
            )
            return loss_val

        loss_value, grads = fastmath.value_and_grad(loss_fn)(weights)
        return float(loss_value), grads

    for step, step_rng in enumerate(rngs):
        # Warmup to allow JIT compilation before measurement.
        if step < warmup_steps:
            trax_trainer.one_step(batch, step_rng, step=step)
            torch_optimizer.zero_grad()
            torch_loss(torch_model(torch_inputs), torch_targets).backward()
            torch_optimizer.step()
            continue

        prev_trax_weights = _copy_weights(trax_trainer.model_with_loss.weights)
        torch_prev_params = [p.detach().clone() for p in torch_model.parameters()]

        loss_value, grads = trax_loss_and_grads(
            trax_trainer.model_with_loss.weights, trax_trainer.model_with_loss.state, step_rng
        )
        trax_grad_norms.append(_flatten_l2_norms(grads))
        trax_losses.append(loss_value)

        trax_trainer.one_step(batch, step_rng, step=step)
        trax_deltas.append(_flatten_deltas(trax_trainer.model_with_loss.weights, prev_trax_weights))

        torch_optimizer.zero_grad()
        torch_outputs = torch_model(torch_inputs)
        torch_step_loss = torch_loss(torch_outputs, torch_targets)
        torch_step_loss.backward()
        torch_optimizer.step()

        torch_losses.append(torch_step_loss.item())
        torch_grad_norms.append([p.grad.norm().item() for p in torch_model.parameters()])
        torch_deltas.append([
            (new.detach() - old).norm().item() for new, old in zip(torch_model.parameters(), torch_prev_params)
        ])

    np.testing.assert_allclose(trax_losses, torch_losses, atol=atol, rtol=rtol)
    for trax_norms, torch_norms in zip(trax_grad_norms, torch_grad_norms):
        np.testing.assert_allclose(trax_norms, torch_norms, atol=atol, rtol=rtol)
    for trax_delta, torch_delta in zip(trax_deltas, torch_deltas):
        np.testing.assert_allclose(trax_delta, torch_delta, atol=atol, rtol=rtol)


class TrainerParityTest(absltest.TestCase):
    def test_single_device_parity_with_divisible_batch(self):
        if torch is None:
            self.skipTest("PyTorch is required for trainer parity tests.")
        batch_size = 8
        inputs = np.arange(batch_size * 2, dtype=np.float32).reshape(batch_size, 2)
        targets = np.arange(batch_size, dtype=np.float32).reshape(batch_size, 1)
        weights = np.ones_like(targets, dtype=np.float32)
        batch = (inputs, targets, weights)

        _run_step_parity(
            batch,
            learning_rate=0.05,
            warmup_steps=2,
            measured_steps=5,
            n_devices=1,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_multi_device_parity_with_padding(self):
        if torch is None:
            self.skipTest("PyTorch is required for trainer parity tests.")
        n_devices = jax.device_count()
        if n_devices <= 1:
            self.skipTest("Multi-device parity test requires more than one JAX device.")

        batch_size = n_devices + 1
        inputs = np.linspace(0.0, 1.0, batch_size * 2, dtype=np.float32).reshape(batch_size, 2)
        targets = np.linspace(1.0, 2.0, batch_size, dtype=np.float32).reshape(batch_size, 1)
        weights = np.ones_like(targets, dtype=np.float32)
        batch = (inputs, targets, weights)

        _run_step_parity(
            batch,
            learning_rate=0.05,
            warmup_steps=2,
            measured_steps=4,
            n_devices=n_devices,
            atol=5e-4,
            rtol=5e-4,
        )

        divisible_batch_size = n_devices * 2
        inputs = np.linspace(0.0, 1.0, divisible_batch_size * 2, dtype=np.float32).reshape(divisible_batch_size, 2)
        targets = np.linspace(1.0, 2.0, divisible_batch_size, dtype=np.float32).reshape(divisible_batch_size, 1)
        weights = np.ones_like(targets, dtype=np.float32)
        batch = (inputs, targets, weights)

        _run_step_parity(
            batch,
            learning_rate=0.05,
            warmup_steps=2,
            measured_steps=4,
            n_devices=n_devices,
            atol=5e-4,
            rtol=5e-4,
        )


if __name__ == "__main__":
    absltest.main()
