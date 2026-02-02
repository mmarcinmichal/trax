# coding=utf-8
"""Shared logging helpers for Trax.

Use these wrappers to keep logging behavior consistent across the codebase.
"""

import sys

from absl import logging as _absl_logging


def _format(msg, args):
    if args:
        return msg % args
    return msg


def info(msg, *args, stdout=False, flush=True, also_log=True):
    if also_log:
        _absl_logging.info(msg, *args)
    if stdout:
        print(_format(msg, args))
        if flush:
            sys.stdout.flush()


def warning(msg, *args, stdout=False, flush=True, also_log=True):
    if also_log:
        _absl_logging.warning(msg, *args)
    if stdout:
        print(_format(msg, args))
        if flush:
            sys.stdout.flush()


def error(msg, *args, stdout=False, flush=True, also_log=True):
    if also_log:
        _absl_logging.error(msg, *args)
    if stdout:
        print(_format(msg, args))
        if flush:
            sys.stdout.flush()


def debug(msg, *args, stdout=False, flush=True, also_log=True):
    if also_log:
        _absl_logging.debug(msg, *args)
    if stdout:
        print(_format(msg, args))
        if flush:
            sys.stdout.flush()


INFO = _absl_logging.INFO
WARNING = _absl_logging.WARNING
ERROR = _absl_logging.ERROR
DEBUG = _absl_logging.DEBUG


def set_verbosity(level):
    _absl_logging.set_verbosity(level)


def vlog_is_on(level):
    return _absl_logging.vlog_is_on(level)
