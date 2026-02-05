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

"""Speed and parity tests for attention implementations.

Run with:
  TRAX_BENCHMARKS=1 python -m pytest tests/layers/attention/speed_test.py -q

If you want GPU numbers, run in an environment with CUDA-enabled JAX and set
JAX_PLATFORMS=cuda before starting pytest.
"""

import os
import time

import jax
import numpy as np

from absl.testing import absltest

import importlib

base_attention = importlib.import_module("trax.layers.attention.base")
efficient_attention = importlib.import_module("trax.layers.attention.efficient")
flash_attention = importlib.import_module("trax.layers.attention.flash")
rel_attention = importlib.import_module("trax.layers.attention.rel")
from trax.utils import shapes


def _time_call(fn, warmup=2, iters=20):
    for _ in range(warmup):
        out = fn()
        jax.device_get(out)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        jax.device_get(out)
        samples.append(time.perf_counter() - start)
    return samples


class AttentionSpeedTest(absltest.TestCase):
    def test_padding_parity_against_base(self):
        if not os.getenv("TRAX_BENCHMARKS"):
            self.skipTest("Set TRAX_BENCHMARKS=1 to enable speed/parity tests.")

        rng = np.random.default_rng(0)
        b, h, l, d = 2, 2, 8, 4
        q = rng.normal(size=(b, h, l, d)).astype(np.float32)
        k = rng.normal(size=(b, h, l, d)).astype(np.float32)
        v = rng.normal(size=(b, h, l, d)).astype(np.float32)
        mask = np.ones((b, l), dtype=np.int32)

        ref_layer = base_attention.DotProductAttention(dropout=0.0, mode="eval")
        ref_layer.init(
            (
                shapes.signature(q),
                shapes.signature(k),
                shapes.signature(v),
                shapes.signature(mask[:, None, None, :]),
            )
        )
        ref = ref_layer((q, k, v, mask[:, None, None, :]))

        for impl in ("naive", "jax", "flash"):
            out = flash_attention.scaled_dot_product_attention(
                q,
                k,
                v,
                implementation=impl,
                attention_mask=mask,
                causal=False,
                dropout=0.0,
                mode="eval",
                block_q=4,
                block_k=4,
            )
            np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)

    def test_attention_speed(self):
        if not os.getenv("TRAX_BENCHMARKS"):
            self.skipTest("Set TRAX_BENCHMARKS=1 to enable speed/parity tests.")

        rng = np.random.default_rng(0)
        b, l, d, h = 4, 128, 256, 8
        x = rng.normal(size=(b, l, d)).astype(np.float32)
        mask_11l = np.ones((b, 1, 1, l), dtype=bool)
        mask_bl = np.ones((b, l), dtype=bool)

        # Base attention (includes projections)
        base_layer = base_attention.Attention(d_feature=d, n_heads=h, dropout=0.0, mode="eval")
        base_layer.init((shapes.signature(x), shapes.signature(mask_11l)))

        # Efficient attention (masked=True expects boolean mask [b, l])
        eff_layer = efficient_attention.SelfAttention(
            n_heads=h,
            d_qk=d // h,
            d_v=d // h,
            masked=True,
            causal=False,
            attention_dropout=0.0,
            output_dropout=0.0,
            mode="eval",
        )
        eff_layer.init((shapes.signature(x), shapes.signature(mask_bl)))

        # Flash attention (three implementations)
        flash_layers = {
            impl: flash_attention.FlashSelfAttention(
                d_model=d,
                n_heads=h,
                block_size=64,
                attention_dropout=0.0,
                use_rope=False,
                use_segment_mask=False,
                implementation=impl,
                mode="eval",
            )
            for impl in ("naive", "jax", "flash")
        }
        for layer in flash_layers.values():
            layer.init(
                (
                    shapes.signature(x),
                    shapes.signature(mask_bl),
                    shapes.signature(np.asarray([[0, l]] * b, dtype=np.int32)),
                    shapes.signature(np.asarray([l] * b, dtype=np.int32)),
                )
            )

        # Relative attention layer (q,k,v interface)
        context_bias_layer, location_bias_layer = rel_attention.get_rel_att_inputs(d, h)
        rel_layer = rel_attention.RelativeAttentionLayer(
            d,
            context_bias_layer,
            location_bias_layer,
            total_kv_pooling=1,
            separate_cls=False,
            n_heads=h,
            dropout=0.0,
            mode="eval",
        )
        q = rng.normal(size=(b, l, d)).astype(np.float32)
        k = rng.normal(size=(b, l, d)).astype(np.float32)
        v = rng.normal(size=(b, l, d)).astype(np.float32)
        rel_layer.init(
            (
                shapes.signature(q),
                shapes.signature(k),
                shapes.signature(v),
                shapes.signature(mask_11l),
            )
        )

        def _run_base():
            return base_layer((x, mask_11l))[0]

        def _run_eff():
            return eff_layer((x, mask_bl))

        def _run_flash(impl):
            layer = flash_layers[impl]
            cu = np.asarray([[0, l]] * b, dtype=np.int32)
            max_len = np.asarray([l] * b, dtype=np.int32)
            return layer((x, mask_bl.astype(np.int32), cu, max_len))

        def _run_rel():
            return rel_layer((q, k, v, mask_11l))[0]

        timings = {
            "base": _time_call(_run_base),
            "efficient": _time_call(_run_eff),
            "flash_naive": _time_call(lambda: _run_flash("naive")),
            "flash_jax": _time_call(lambda: _run_flash("jax")),
            "flash": _time_call(lambda: _run_flash("flash")),
            "relative": _time_call(_run_rel),
        }

        backend = jax.default_backend()
        header = (
            "Attention timings (backend=%s, b=%d, l=%d, d=%d, h=%d):"
            % (backend, b, l, d, h)
        )
        lines = [header]
        for name, samples in timings.items():
            arr = np.asarray(samples)
            lines.append(
                f"  {name}: n={arr.size} "
                f"mean={arr.mean():.6f}s "
                f"median={np.median(arr):.6f}s "
                f"min={arr.min():.6f}s "
                f"max={arr.max():.6f}s"
            )
        report = "\n" + "\n".join(lines)
        print(report)

        report_path = "/tmp/trax_attention_speed.txt"
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write(report + "\n")

        self.assertTrue(all(all(s > 0.0 for s in samples) for samples in timings.values()))


if __name__ == "__main__":
    absltest.main()
