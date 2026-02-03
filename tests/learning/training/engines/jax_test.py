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

"""Tests for JAX training engines and reversible trainers."""

import time

import numpy as np

from absl.testing import absltest

from tests.fastmath.jax.config import config
from trax import fastmath, optimizers
from trax import layers as tl
from trax.layers import base
from trax.learning.training.utils import runtime
from trax.learning.training.engines import jax as jax_engine
from trax.models.research import terraformer
from trax.utils import shapes


class TrainerTest(absltest.TestCase):
    def _assert_all_equal(self, t1, t2, tol=1e-5):
        def eq(x1, x2):
            diff = np.maximum(np.abs(x1 - x2) - tol, 0.0)
            self.assertLessEqual(
                np.sum(diff), 0.0, msg=f"\n{x1}\n !=\n{x2}\n diff:\n{x1-x2}"
            )

        fastmath.nested_map_multiarg(eq, t1, t2)

    def test_run_simple_task(self):
        inputs_batch = np.arange(8).reshape((8, 1))
        targets_batch = np.pi * np.ones_like(inputs_batch)
        labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
        loss_layer = tl.Serial(tl.Dense(1), tl.L2Loss())
        loss_layer.init(labeled_batch)
        optimizer = optimizers.SGD(0.01)
        optimizer.tree_init(loss_layer.weights)
        trainer = jax_engine.TrainingEngine(loss_layer, optimizer)
        rng = fastmath.random.get_prng(0)
        trainer.one_step(labeled_batch, rng)

    def test_run_sharded_terraformer(self):
        if fastmath.local_device_count() == 1:
            return
        inputs_batch = np.arange(8).reshape((2, 4)) + 1
        targets_batch = 2 * inputs_batch
        labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
        int_sig = shapes.ShapeDtype((2, 4), dtype=np.int32)
        input_sig = (int_sig, int_sig, int_sig)
        model = terraformer.ConfigurableTerraformer(
            20,
            d_model=8,
            d_ff=32,
            n_heads=1,
            dropout=0.0,
            n_encoder_layers=2,
            n_decoder_layers=2,
            ff_sparsity=(4, 8, 0.0, 1.0),
            encoder_attention_type=tl.Attention,
            encoder_decoder_attention_type=tl.CausalAttention,
            pos_type=None,
            reversible_encoder=True,
        )
        loss = tl.Serial(tl.LogSoftmax(), tl.CrossEntropyLoss())
        model_with_loss = tl.Serial(model, loss)
        rng_init = fastmath.random.get_prng(12)
        model_with_loss.init(input_sig, rng=rng_init)

        optimizer = optimizers.Adafactor(0.01)
        optimizer.tree_init(model_with_loss.weights)
        trainer = jax_engine.TrainingEngine(model_with_loss, optimizer)
        rng_step1 = fastmath.random.get_prng(7)
        trainer.one_step(labeled_batch, rng_step1)

    def test_run_reversible_slots(self):
        layers = [tl.Dense(4), tl.Dup()]
        rev_layers = [tl.ReversibleHalfResidual(tl.Dense(4)), tl.ReversibleSwap()]
        loss_layer = tl.Serial(
            tl.Concatenate(), tl.Dense(4), tl.LogSoftmax(), tl.CrossEntropyLoss()
        )
        trainer = jax_engine.ReversibleSerialTrainer(
            [(layers, rev_layers)], loss_layer, optimizers.Adam
        )
        slots = trainer.slots
        trainer.slots = slots
        self.assertEqual(slots, trainer.slots)

    def test_run_reversible_same_as_default_basic(self):
        inputs_batch = np.arange(8).reshape((2, 4))
        targets_batch = 2 * inputs_batch
        labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
        first_layer = tl.Serial(tl.Embedding(9, 4), tl.Dropout(0.5), tl.Dup())
        rev_layers = [
            tl.ReversibleHalfResidual(tl.Dense(4), tl.Dropout(0.2)),
            tl.ReversibleSwap(),
            tl.ReversibleHalfResidual(tl.Dropout(0.5), tl.Dense(4)),
            tl.ReversibleSwap(),
        ]
        loss_layer = tl.Serial(
            tl.Concatenate(),
            tl.Dense(19),
            tl.Dropout(0.3),
            tl.LogSoftmax(),
            tl.CrossEntropyLoss(),
        )
        model = tl.Serial([first_layer] + rev_layers + [loss_layer])
        rng_init = fastmath.random.get_prng(12)
        model.init(labeled_batch, rng=rng_init)
        optimizer_fn = optimizers.Adam

        optimizer = optimizer_fn()
        optimizer.tree_init(model.weights)
        trainer = jax_engine.TrainingEngine(model, optimizer)
        rng_step1 = fastmath.random.get_prng(7)
        rng_step2 = fastmath.random.get_prng(8)
        trainer.one_step(labeled_batch, rng_step1)
        trainer.one_step(labeled_batch, rng_step2, learning_rate=0.02)
        first_layer_weights1 = first_layer.weights
        rev_layer0_weights1 = rev_layers[0].weights
        rev_layer2_weights1 = rev_layers[2].weights
        loss_layer_weights1 = loss_layer.weights

        model.init(labeled_batch, rng=rng_init)
        trainer = jax_engine.ReversibleSerialTrainer(
            [(first_layer.sublayers, rev_layers)], loss_layer, optimizer_fn
        )
        trainer.one_step(labeled_batch, rng_step1)
        trainer.one_step(labeled_batch, rng_step2, learning_rate=0.02)

        self._assert_all_equal(loss_layer_weights1, loss_layer.weights)
        self._assert_all_equal(rev_layer2_weights1, rev_layers[2].weights)
        self._assert_all_equal(rev_layer0_weights1, rev_layers[0].weights)
        self._assert_all_equal(first_layer_weights1, first_layer.weights)

    def test_run_reversible_same_as_default_extended(self):
        inputs_batch = np.arange(8).reshape((2, 4))
        targets_batch = 2 * inputs_batch
        labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
        first_layer = tl.Serial(tl.Embedding(9, 4), tl.Dropout(0.5), tl.Dup())
        rev_layers1 = [
            tl.ReversibleHalfResidual(tl.Dense(4), tl.Dropout(0.2)),
            tl.ReversibleSwap(),
            tl.ReversibleHalfResidual(tl.Dropout(0.5), tl.Dense(4)),
            tl.ReversibleSwap(),
        ]
        mid_layer = tl.Serial(tl.Add(), tl.Dense(4), tl.Dup())
        rev_layers2 = [
            tl.ReversibleHalfResidual(tl.Dense(4), tl.Dropout(0.3)),
            tl.ReversibleSwap(),
        ]
        loss_layer = tl.Serial(
            tl.Concatenate(),
            tl.Dense(19),
            tl.Dropout(0.3),
            tl.LogSoftmax(),
            tl.CrossEntropyLoss(),
        )
        model = tl.Serial(
            [first_layer] + rev_layers1 + [mid_layer] + rev_layers2 + [loss_layer]
        )
        rng_init = fastmath.random.get_prng(12)
        model.init(labeled_batch, rng=rng_init)
        optimizer_fn = optimizers.Adam

        optimizer = optimizer_fn()
        optimizer.tree_init(model.weights)
        trainer = jax_engine.TrainingEngine(model, optimizer)
        rng_step1 = fastmath.random.get_prng(7)
        rng_step2 = fastmath.random.get_prng(8)
        rng_step3 = fastmath.random.get_prng(9)
        trainer.one_step(labeled_batch, rng_step1)
        trainer.one_step(labeled_batch, rng_step2, learning_rate=0.02)
        trainer.one_step(labeled_batch, rng_step3, learning_rate=0.03)
        first_layer_weights1 = first_layer.weights
        rev_layer12_weights1 = rev_layers1[2].weights
        mid_layer_weights1 = mid_layer.weights
        rev_layer20_weights1 = rev_layers2[0].weights
        loss_layer_weights1 = loss_layer.weights

        model.init(labeled_batch, rng=rng_init)
        trainer = jax_engine.ReversibleSerialTrainer(
            [(first_layer.sublayers, rev_layers1), (mid_layer.sublayers, rev_layers2)],
            loss_layer,
            optimizer_fn,
            memoize_jit=False,
        )
        trainer.one_step(labeled_batch, rng_step1)
        trainer.one_step(labeled_batch, rng_step2, learning_rate=0.02)
        trainer.one_step(labeled_batch, rng_step3, learning_rate=0.03)

        self._assert_all_equal(loss_layer_weights1, loss_layer.weights)
        self._assert_all_equal(rev_layer20_weights1, rev_layers2[0].weights)
        self._assert_all_equal(mid_layer_weights1, mid_layer.weights)
        self._assert_all_equal(rev_layer12_weights1, rev_layers1[2].weights)
        self._assert_all_equal(first_layer_weights1, first_layer.weights)

    def test_run_reversible_same_as_default_terraformer(self):
        inputs_batch = np.arange(8).reshape((2, 4)) + 1
        targets_batch = 2 * inputs_batch
        labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
        int_sig = shapes.ShapeDtype((2, 4), dtype=np.int32)
        input_sig = (int_sig, int_sig, int_sig)
        model = terraformer.ConfigurableTerraformer(
            20,
            d_model=8,
            d_ff=32,
            n_heads=1,
            dropout=0.0,
            n_encoder_layers=2,
            n_decoder_layers=2,
            ff_sparsity=(4, 8, 0.0, 1.0),
            pos_type=None,
            reversible_encoder=True,
        )
        loss = tl.Serial(tl.LogSoftmax(), tl.CrossEntropyLoss())
        optimizer_fn = optimizers.Adafactor
        blocks, loss_layer = jax_engine.extract_reversible_blocks(
            [model, loss], loss_chunk_size=4
        )
        blocks_serial = [(tl.Serial(std), rev) for (std, rev) in blocks]
        model_with_loss = tl.Serial(model, loss)
        rng_init = fastmath.random.get_prng(12)
        model_with_loss.init(input_sig, rng=rng_init)

        optimizer = optimizer_fn()
        optimizer.tree_init(model_with_loss.weights)
        trainer = jax_engine.TrainingEngine(model_with_loss, optimizer)
        rng_step1 = fastmath.random.get_prng(7)
        rng_step2 = fastmath.random.get_prng(8)
        rng_step3 = fastmath.random.get_prng(9)
        trainer.one_step(labeled_batch, rng_step1)
        trainer.one_step(labeled_batch, rng_step2, learning_rate=0.02)
        trainer.one_step(labeled_batch, rng_step3, learning_rate=0.03)
        first_weights = blocks_serial[0][0].weights
        first_rev_weights = blocks[0][1][0].weights
        loss_weights = loss_layer.weights

        model_with_loss.init(input_sig, rng=rng_init)
        trainer = jax_engine.ReversibleSerialTrainer(blocks, loss_layer, optimizer_fn)
        trainer.one_step(labeled_batch, rng_step1)
        trainer.one_step(labeled_batch, rng_step2, learning_rate=0.02)
        trainer.one_step(labeled_batch, rng_step3, learning_rate=0.03)

        self._assert_all_equal(loss_weights, loss_layer.weights)
        self._assert_all_equal(first_rev_weights, blocks[0][1][0].weights)
        self._assert_all_equal(first_weights, blocks_serial[0][0].weights)

    def test_run_reversible_large_weights(self):
        ram_limited = True
        if fastmath.global_device_count() == 1 and ram_limited:
            return

        inputs_batch = np.arange(8).reshape((2, 4))
        targets_batch = inputs_batch
        labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
        first_layer = tl.Serial(tl.Embedding(9, 16 * 1024), tl.Dup())
        rng_init = fastmath.random.get_prng(12)
        rng_step = fastmath.random.get_prng(13)

        first_layer.init(labeled_batch, rng=rng_init)
        n_layers = 18
        rev_layers = []
        int_shape = shapes.ShapeDtype((2, 4), dtype=np.int32)
        shape = shapes.ShapeDtype((2, 4, 16 * 1024))
        sig = (shape, shape)
        for _ in range(n_layers):
            layer = tl.ReversibleHalfResidual(tl.Dense(16 * 1024))
            layer.init(sig, rng=rng_init)
            layer.weights = jax_runtime.on_cpu(layer.weights)
            rev_layers.append(layer)
            rev_layers.append(tl.ReversibleSwap())
        loss_layer = tl.Serial(
            tl.Concatenate(), tl.Dense(9), tl.LogSoftmax(), tl.CrossEntropyLoss()
        )
        loss_layer.init((shape, shape, int_shape, int_shape))
        optimizer_fn = optimizers.Adafactor

        trainer = jax_engine.ReversibleSerialTrainer(
            [(first_layer, rev_layers)], loss_layer, optimizer_fn
        )
        loss, _ = trainer.one_step(labeled_batch, rng_step)
        self.assertLess(float(loss.sum()), 10000.0)
        run_twice = False
        if run_twice:
            t = time.time()
            loss, _ = trainer.one_step(labeled_batch, rng_step)
            self.assertLess(float(loss.sum()), 10000.0)
            print("Took %.3f seconds to run, loss %s" % (time.time() - t, loss))

    def test_run_reversible_weights_trainsfer_xprof(self):
        run_this_test = False
        if not run_this_test or fastmath.global_device_count() == 1:
            return

        inputs_batch = np.ones((1024, 128), dtype=np.int32)
        targets_batch = inputs_batch
        labeled_batch = (inputs_batch, targets_batch, np.ones_like(targets_batch))
        first_layer = tl.Serial(tl.Embedding(4, 1024), tl.Dup())
        rng_init = fastmath.random.get_prng(12)
        rng_step = fastmath.random.get_prng(13)

        first_layer.init(labeled_batch, rng=rng_init)
        n_layers = 6
        rev_layers = []
        int_shape = shapes.ShapeDtype((1024, 128), dtype=np.int32)
        shape = shapes.ShapeDtype((1024, 128, 1024))
        sig = (shape, shape)
        for _ in range(n_layers):
            layer = tl.ReversibleHalfResidual(tl.Dense(1024))
            layer.init(sig, rng=rng_init)
            layer.weights = runtime.on_cpu(layer.weights)
            rev_layers.append(layer)
            rev_layers.append(tl.ReversibleSwap())
        loss_layer = tl.Serial(
            tl.Concatenate(), tl.Dense(9), tl.LogSoftmax(), tl.CrossEntropyLoss()
        )
        loss_layer.init((shape, shape, int_shape, int_shape))
        optimizer_fn = optimizers.SGD

        trainer = jax_engine.ReversibleSerialTrainer(
            [(first_layer, rev_layers)], loss_layer, optimizer_fn
        )
        loss, _ = trainer.one_step(labeled_batch, rng_step)
        self.assertLess(float(loss.sum()), 10000.0)
        t = time.time()
        loss, _ = trainer.one_step(labeled_batch, rng_step)
        self.assertLess(float(loss.sum()), 10000.0)
        print("Took %.3f seconds to run, loss %s" % (time.time() - t, loss))


class EngineComparisonTest(absltest.TestCase):
    def _build_model_with_loss(self):
        return tl.Serial(
            tl.Parallel(
                tl.Dense(1),
                tl.Fn("Identity", lambda x: x),
                tl.Fn("Identity", lambda x: x),
            ),
            tl.L2Loss(),
        )

    def _init_model(self, model, batch):
        sig = shapes.signature(batch)
        model.init(sig, rng=fastmath.random.get_prng(0))

    def test_one_step_returns_loss_and_stats(self):
        batch_size = 8
        x = np.random.randn(batch_size, 1).astype(np.float32)
        y = np.random.randn(batch_size, 1).astype(np.float32)
        weights = np.ones((batch_size, 1), dtype=np.float32)
        batch = (x, y, weights)

        model = self._build_model_with_loss()
        self._init_model(model, batch)
        init_weights = fastmath.nested_map(np.copy, model.weights)

        opt = optimizers.SGD(0.1)
        trainer = jax_engine.TrainingEngine(model, opt, n_devices=1)

        rng = fastmath.random.get_prng(1)
        loss, stats = trainer.one_step(batch, rng, step=0)

        self.assertTrue(np.isfinite(np.asarray(loss)).all())
        if "loss" in stats:
            self.assertAlmostEqual(loss, float(stats["loss"]), places=5)
        else:
            loss_keys = [key for key in stats.keys() if key.endswith("/loss")]
            self.assertTrue(loss_keys)

        def _changed(before, after):
            before_flat = fastmath.tree_flatten(before)
            after_flat = fastmath.tree_flatten(after)
            return any(
                not np.allclose(b, a) for b, a in zip(before_flat, after_flat)
            )

        self.assertTrue(_changed(init_weights, model.weights))

    def test_multidevice_one_step_matches_api(self):
        if fastmath.local_device_count() < 2:
            self.skipTest("Multi-device test requires at least 2 devices.")

        n_devices = 2
        batch_size = n_devices * 2
        x = np.random.randn(batch_size, 1).astype(np.float32)
        y = np.random.randn(batch_size, 1).astype(np.float32)
        weights = np.ones((batch_size, 1), dtype=np.float32)
        batch = (x, y, weights)

        model = self._build_model_with_loss()
        self._init_model(model, batch)
        opt = optimizers.SGD(0.1)
        trainer = jax_engine.TrainingEngine(model, opt, n_devices=n_devices)

        rng = fastmath.random.get_prng(2)
        loss, stats = trainer.one_step(batch, rng, step=0)

        self.assertTrue(np.isfinite(np.asarray(loss)).all())
        if "loss" not in stats:
            loss_keys = [key for key in stats.keys() if key.endswith("/loss")]
            self.assertTrue(loss_keys)

    def test_reversible_trainer_one_step(self):
        batch_size = 4
        x = np.random.randn(batch_size, 1).astype(np.float32)
        y = np.random.randn(batch_size, 1).astype(np.float32)
        weights = np.ones((batch_size, 1), dtype=np.float32)
        batch = (x, y, weights)

        model = tl.Parallel(
            tl.Dense(1),
            tl.Fn("Identity", lambda x: x),
            tl.Fn("Identity", lambda x: x),
        )
        loss_layer = tl.L2Loss()
        sig = shapes.signature(batch)
        blocks, loss = jax_engine.extract_reversible_blocks(
            [model, loss_layer], loss_chunk_size=0
        )
        jax_engine.init_reversible_blocks(
            blocks, loss, sig, fastmath.random.get_prng(4)
        )

        def optimizer_fn(_=None):
            return optimizers.SGD(0.1)

        trainer = jax_engine.ReversibleSerialTrainer(
            blocks, loss, optimizer_fn, n_devices=1
        )

        rng = fastmath.random.get_prng(5)
        loss_value, stats = trainer.one_step(batch, rng, step=0)

        self.assertTrue(np.isfinite(np.asarray(loss_value)).all())
        if "loss" not in stats:
            loss_keys = [key for key in stats.keys() if key.endswith("/loss")]
            self.assertTrue(loss_keys)


if __name__ == "__main__":
    config.config_with_absl()
    absltest.main()
