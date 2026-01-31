# coding=utf-8
# Lightweight helper that builds tf.data.Datasets from local JSON files
# Conventions:
#   data_dir/raw_json/<dataset_id>/train/*.json  -> train split (JSONL or one-object-per-file)
#   data_dir/raw_json/<dataset_id>/eval/*.json   -> eval split (optional)
#
# Each line (JSONL) should be a JSON object with fields, e.g. {"text": "...", "label": "..."}
# The produced examples are dicts with keys "inputs" and "targets".
from __future__ import annotations

import glob
import json
import os

from typing import List, Optional, Tuple

import tensorflow as tf


def _parse_json_line(line: tf.Tensor, input_key: str = "text", target_key: str = "label", single_file: bool = False):
    """Parse a single JSON line (tf.string) into a dict {'inputs': ..., 'targets': ...}.

    Uses tf.py_function for simplicity (parsing in Python). Keeps shapes as scalars.
    """
    def _to_features(s):
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        # If single_file is True, s is the entire file content. We expect either a single object
        # or a JSON array; for simplicity we support single-object files by returning that object.
        try:
            obj = json.loads(s)
        except Exception:
            # If parsing fails, return empty strings to avoid crashing the pipeline.
            return "", ""
        # If the file contained a list, it's ambiguous; fall back to first element if present.
        if single_file and isinstance(obj, list) and obj:
            obj = obj[0]
        inp = obj.get(input_key, "")
        tgt = obj.get(target_key, "")
        # Ensure returned types are str
        return str(inp), str(tgt)

    inp, tgt = tf.py_function(_to_features, [line], [tf.string, tf.string])
    inp.set_shape([])
    tgt.set_shape([])
    return {"inputs": inp, "targets": tgt}

def _dataset_from_files(
    files: List[str],
    jsonl: bool = True,
    input_key: str = "text",
    target_key: str = "label",
    shuffle_files: bool = True,
):
    if not files:
        return None
    files = sorted(files)
    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle_files:
        ds = ds.shuffle(buffer_size=max(1, len(files)))
    # Read file contents: either line-by-line (jsonl) or whole file as single example
    def _file_to_lines(fname):
        fname = tf.cast(fname, tf.string)
        if jsonl:
            return tf.data.TextLineDataset(fname)
        else:
            content = tf.io.read_file(fname)
            return tf.data.Dataset.from_tensors(content)
    ds = ds.interleave(_file_to_lines, cycle_length=4, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda line: _parse_json_line(line, input_key=input_key, target_key=target_key, single_file=not jsonl),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds

def load_raw_json_datasets(
    dataset_id: str,
    data_dir: Optional[str],
    jsonl: bool = True,
    train_shuffle_files: bool = True,
    eval_shuffle_files: bool = False,
    input_key: str = "text",
    target_key: str = "label",
) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Tuple[List[str], List[str]]]:
    """Load train/eval tf.data.Datasets from local JSON files using a simple convention.

    Returns (train_ds, eval_ds, supervised_keys) where supervised_keys is (['inputs'], ['targets'])
    to be compatible with the existing DatasetLoader/DatasetStreams interface.
    """
    base = data_dir or os.getcwd()
    base = os.path.join(base, "raw_json", dataset_id)
    train_dir = os.path.join(base, "train")
    eval_dir = os.path.join(base, "eval")

    train_files = glob.glob(os.path.join(train_dir, "*.json")) if os.path.isdir(train_dir) else []
    eval_files = glob.glob(os.path.join(eval_dir, "*.json")) if os.path.isdir(eval_dir) else []

    train_ds = _dataset_from_files(train_files, jsonl=jsonl, input_key=input_key, target_key=target_key, shuffle_files=train_shuffle_files)
    eval_ds = _dataset_from_files(eval_files, jsonl=jsonl, input_key=input_key, target_key=target_key, shuffle_files=eval_shuffle_files)

    supervised_keys = (["inputs"], ["targets"])
    return train_ds, eval_ds, supervised_keys
