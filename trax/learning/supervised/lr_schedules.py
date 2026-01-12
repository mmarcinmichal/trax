# coding=utf-8
"""Compatibility shim for learning rate schedules.

Implementation lives in :mod:`trax.learning.training.lr_schedules`.
"""

import gin

from trax.learning.training.utils import lr_schedules as _base

constant = gin.external_configurable(
    _base.constant, module="trax.learning.supervised.lr_schedules"
)
warmup = gin.external_configurable(
    _base.warmup, module="trax.learning.supervised.lr_schedules"
)
warmup_and_rsqrt_decay = gin.external_configurable(
    _base.warmup_and_rsqrt_decay, module="trax.learning.supervised.lr_schedules"
)
multifactor = gin.external_configurable(
    _base.multifactor, module="trax.learning.supervised.lr_schedules"
)

_BodyAndTail = _base._BodyAndTail
_CosineSawtoothTail = _base._CosineSawtoothTail
_rsqrt = _base._rsqrt

__all__ = [
    "constant",
    "warmup",
    "warmup_and_rsqrt_decay",
    "multifactor",
    "_BodyAndTail",
    "_CosineSawtoothTail",
    "_rsqrt",
]
