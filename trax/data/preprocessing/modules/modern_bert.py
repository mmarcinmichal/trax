"""ModernBERT-style preprocessing components."""

from collections import deque
from typing import Callable, Iterable, List, Optional

import gin
import numpy as np

from trax.data.encoder import encoder as trax_encoder


@gin.configurable(module="trax.data")
def ModernBertTokenize(
    input_key="text",
    output_key="input_ids",
    vocab_file="answerdotai/ModernBERT-base",
    vocab_dir=None,
    n_reserved_ids=0,
    append_bos=False,
    append_eos=False,
):
    """Tokenize text using the built-in ModernBERT encoder."""

    def _tokenize(stream):
        encoder = trax_encoder._get_vocab(
            vocab_type="modernbert", vocab_file=vocab_file, vocab_dir=vocab_dir
        )
        tokenized = trax_encoder.tokenize(
            stream,
            keys=[input_key] if input_key is not None else None,
            vocab_type="modernbert",
            vocab_file=vocab_file,
            vocab_dir=vocab_dir,
            n_reserved_ids=n_reserved_ids,
        )
        for example in tokenized:
            if not isinstance(example, dict):
                raise ValueError("ModernBertTokenize expects dict examples.")
            if input_key not in example:
                raise KeyError(f"ModernBertTokenize missing key: {input_key}")
            tokens = np.asarray(example[input_key], dtype=np.int32)
            if append_bos:
                tokens = np.concatenate(
                    [np.asarray([encoder.bos_id], dtype=np.int32), tokens]
                )
            if append_eos:
                tokens = np.concatenate(
                    [tokens, np.asarray([encoder.eos_id], dtype=np.int32)]
                )
            yield {output_key: tokens}

    return _tokenize


@gin.configurable(module="trax.data")
def SelectTextField(field="text", output_key="text"):
    """Select a single text field from dict examples."""

    def _select(stream):
        for example in stream:
            if not isinstance(example, dict):
                raise ValueError("SelectTextField expects dict examples.")
            if field not in example:
                raise KeyError(f"SelectTextField missing field: {field}")
            yield {output_key: example[field]}

    return _select


@gin.configurable(module="trax.data")
def TokenizerEncode(
    tokenizer=None,
    encode_fn: Optional[Callable[[str], List[int]]] = None,
    input_key="text",
    output_key="input_ids",
    add_special_tokens=False,
    truncation=False,
    padding=False,
    max_length=None,
):
    """Encode text into token ids using a tokenizer or encode_fn."""

    if tokenizer is None and encode_fn is None:
        raise ValueError("TokenizerEncode requires tokenizer or encode_fn.")

    def _encode_text(text):
        if encode_fn is not None:
            return encode_fn(text)
        encoded = tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
        )
        if isinstance(encoded, dict) and "input_ids" in encoded:
            return encoded["input_ids"]
        return encoded

    def _tokenize(stream):
        for example in stream:
            if not isinstance(example, dict):
                raise ValueError("TokenizerEncode expects dict examples.")
            text = example.get(input_key)
            if text is None:
                raise KeyError(f"TokenizerEncode missing key: {input_key}")
            tokens = _encode_text(text)
            tokens = np.asarray(tokens, dtype=np.int32)
            yield {output_key: tokens}

    return _tokenize


@gin.configurable(module="trax.data")
def AppendDocBoundaryTokens(
    input_key="input_ids",
    output_key="input_ids",
    bos_tokens=None,
    eos_tokens=None,
):
    """Append optional BOS/EOS tokens to the sequence."""
    bos_tokens = list(bos_tokens or [])
    eos_tokens = list(eos_tokens or [])

    def _append(stream):
        for example in stream:
            if not isinstance(example, dict):
                raise ValueError("AppendDocBoundaryTokens expects dict examples.")
            tokens = example.get(input_key)
            if tokens is None:
                raise KeyError(f"AppendDocBoundaryTokens missing key: {input_key}")
            tokens = np.asarray(tokens, dtype=np.int32)
            if bos_tokens or eos_tokens:
                tokens = np.concatenate(
                    [np.asarray(bos_tokens, dtype=np.int32), tokens, np.asarray(eos_tokens, dtype=np.int32)]
                )
            yield {output_key: tokens}

    return _append


@gin.configurable(module="trax.data")
def ModernBertAppendDocBoundaryTokens(
    input_key="input_ids",
    output_key="input_ids",
    vocab_file="answerdotai/ModernBERT-base",
    vocab_dir=None,
    add_bos=False,
    add_eos=True,
):
    """Append BOS/EOS tokens using ModernBERT encoder ids."""
    encoder = trax_encoder._get_vocab(
        vocab_type="modernbert", vocab_file=vocab_file, vocab_dir=vocab_dir
    )
    bos_tokens = [encoder.bos_id] if add_bos else []
    eos_tokens = [encoder.eos_id] if add_eos else []

    def _append(stream):
        for example in stream:
            if not isinstance(example, dict):
                raise ValueError("ModernBertAppendDocBoundaryTokens expects dict examples.")
            tokens = example.get(input_key)
            if tokens is None:
                raise KeyError(f"ModernBertAppendDocBoundaryTokens missing key: {input_key}")
            tokens = np.asarray(tokens, dtype=np.int32)
            if bos_tokens or eos_tokens:
                tokens = np.concatenate(
                    [
                        np.asarray(bos_tokens, dtype=np.int32),
                        tokens,
                        np.asarray(eos_tokens, dtype=np.int32),
                    ]
                )
            yield {output_key: tokens}

    return _append


@gin.configurable(module="trax.data")
def ChunkFixedLength(
    max_seq_len: int,
    input_key="input_ids",
    output_key="input_ids",
    no_wrap: bool = False,
):
    """Concatenate tokens across docs and emit fixed-length chunks."""
    if max_seq_len <= 0:
        raise ValueError("ChunkFixedLength requires max_seq_len > 0.")

    def _chunk(stream):
        buffer = []
        for example in stream:
            if not isinstance(example, dict):
                raise ValueError("ChunkFixedLength expects dict examples.")
            tokens = example.get(input_key)
            if tokens is None:
                raise KeyError(f"ChunkFixedLength missing key: {input_key}")
            tokens = np.asarray(tokens, dtype=np.int32).tolist()
            if tokens:
                buffer.extend(tokens)
            while len(buffer) >= max_seq_len:
                chunk = np.asarray(buffer[:max_seq_len], dtype=np.int32)
                buffer = buffer[max_seq_len:]
                yield {output_key: chunk}
        if no_wrap:
            buffer.clear()

    return _chunk


@gin.configurable(module="trax.data")
def BatchDict(
    batch_size: int,
    keys=("input_ids",),
    pad_to: Optional[int] = None,
    pad_value: Optional[dict] = None,
):
    """Batch dict examples into a single dict of numpy arrays."""
    if batch_size <= 0:
        raise ValueError("BatchDict requires batch_size > 0.")
    keys = tuple(keys or [])
    pad_value = pad_value or {}

    def _pad_array(arr, length, value):
        if length is None:
            return arr
        if arr.shape[0] == length:
            return arr
        if arr.shape[0] > length:
            return arr[:length]
        pad_width = length - arr.shape[0]
        return np.pad(arr, (0, pad_width), constant_values=value)

    def _batch(stream):
        batch = []
        for example in stream:
            if not isinstance(example, dict):
                raise ValueError("BatchDict expects dict examples.")
            batch.append(example)
            if len(batch) < batch_size:
                continue
            out = {}
            for key in keys:
                vals = []
                for item in batch:
                    if key not in item:
                        raise KeyError(f"BatchDict missing key: {key}")
                    arr = np.asarray(item[key], dtype=np.int32)
                    arr = _pad_array(arr, pad_to, pad_value.get(key, 0))
                    vals.append(arr)
                out[key] = np.stack(vals, axis=0)
            batch = []
            yield out

    return _batch


@gin.configurable(module="trax.data")
def BatchDictList(batch_size: int):
    """Batch dict examples into a list without padding."""
    if batch_size <= 0:
        raise ValueError("BatchDictList requires batch_size > 0.")

    def _batch(stream):
        batch = []
        for example in stream:
            if not isinstance(example, dict):
                raise ValueError("BatchDictList expects dict examples.")
            batch.append(example)
            if len(batch) < batch_size:
                continue
            yield batch
            batch = []

    return _batch


def _find_best_fit(remaining_spaces, seq_len, seq_counts, max_sequences_per_pack):
    valid = remaining_spaces >= seq_len
    if max_sequences_per_pack is not None:
        valid = valid & (seq_counts < max_sequences_per_pack)
    if not np.any(valid):
        return -1
    valid_space_sizes = remaining_spaces[valid]
    best_fit_idx = np.argmin(valid_space_sizes)
    return np.arange(len(remaining_spaces))[valid][best_fit_idx]


@gin.configurable(module="trax.data")
def ModernBertSequencePacker(
    src_batch_size: int,
    src_max_seq_len: int,
    micro_batch_size: int,
    pad_token_id: Optional[int],
    mask_token_id: Optional[int],
    ignore_token_id: int = -100,
    mask_prob: float = 0.3,
    seed: int = 42,
    suppress_masking: bool = False,
    buffer_size: Optional[int] = None,
    max_sequences_per_pack: Optional[int] = None,
    unpad_input: bool = False,
    pad_cu_seqlens_to: Optional[int] = None,
    pad_cu_seqlens_value: int = -1,
    input_key: str = "input_ids",
    vocab_file: str = "answerdotai/ModernBERT-base",
    vocab_dir: Optional[str] = None,
):
    """Greedy best-fit sequence packer matching ModernBERT behavior."""
    if src_batch_size % micro_batch_size != 0:
        raise ValueError("src_batch_size must be divisible by micro_batch_size.")
    out_batch_size = src_batch_size // micro_batch_size
    out_pseq_len = micro_batch_size * src_max_seq_len
    buffer_size = buffer_size or (5 * src_batch_size)

    encoder = None
    if pad_token_id is None or mask_token_id is None:
        encoder = trax_encoder._get_vocab(
            vocab_type="modernbert", vocab_file=vocab_file, vocab_dir=vocab_dir
        )
    if pad_token_id is None:
        pad_token_id = int(encoder.pad_id)
    if mask_token_id is None:
        mask_token_id = int(encoder.mask_id)

    def _mlm_masking(seq, np_rng):
        labels = np.where(seq == pad_token_id, ignore_token_id, seq)
        rand = np_rng.random(seq.shape)
        mask_mask = rand < mask_prob * 0.8
        random_mask = (rand >= mask_prob * 0.8) & (rand < mask_prob * 0.9)
        keep_mask = (rand >= mask_prob * 0.9) & (rand < mask_prob)
        labels = np.where(mask_mask | random_mask | keep_mask, labels, ignore_token_id)
        seq = np.where(mask_mask, mask_token_id, seq)
        random_words = np_rng.integers(0, np.max(seq) + 1, size=seq.shape)
        seq = np.where(random_mask, random_words, seq)
        return seq, labels

    def _trim_pad(arr):
        if not unpad_input:
            return arr
        if arr.size == 0:
            return arr
        last = arr.shape[0]
        while last > 0 and arr[last - 1] == pad_token_id:
            last -= 1
        return arr[:last]

    def _create_batch(buffer):
        batch = np.full((out_batch_size, out_pseq_len), pad_token_id, dtype=np.int32)
        seq_counts = np.zeros(out_batch_size, dtype=np.int32)
        cum_seq_lens = [[0] for _ in range(out_batch_size)]
        remaining_spaces = np.full((out_batch_size,), out_pseq_len, dtype=np.int32)
        temp_buffer = []

        while True:
            if not buffer:
                break
            seq = buffer.popleft()
            seq_len = len(seq)
            if seq_len == 0:
                continue
            best_fit_idx = _find_best_fit(
                remaining_spaces, seq_len, seq_counts, max_sequences_per_pack
            )
            if best_fit_idx != -1:
                end_pos = out_pseq_len - remaining_spaces[best_fit_idx]
                batch[best_fit_idx, end_pos : end_pos + seq_len] = seq
                seq_counts[best_fit_idx] += 1
                remaining_spaces[best_fit_idx] -= seq_len
                cum_seq_lens[best_fit_idx].append(
                    cum_seq_lens[best_fit_idx][-1] + seq_len
                )
            else:
                temp_buffer.append(seq)

        buffer.extendleft(temp_buffer)
        if np.all(seq_counts > 0):
            for x in cum_seq_lens:
                if x[-1] != out_pseq_len:
                    x.append(out_pseq_len)
            return batch, cum_seq_lens
        return None

    def _pack(stream):
        buffer = deque()
        np_rng = np.random.default_rng(seed)
        for example in stream:
            if isinstance(example, list):
                batch_items = example
            elif isinstance(example, dict):
                if input_key not in example:
                    raise KeyError(f"ModernBertSequencePacker missing key: {input_key}")
                batch_items = example[input_key]
            else:
                raise ValueError("ModernBertSequencePacker expects list or dict batch input.")

            if isinstance(batch_items, np.ndarray):
                if batch_items.ndim != 2:
                    raise ValueError("ModernBertSequencePacker expects 2D batch input.")
                if batch_items.shape[0] > src_batch_size:
                    raise ValueError("Incoming batch larger than src_batch_size.")
                for row in batch_items:
                    buffer.append(_trim_pad(np.asarray(row, dtype=np.int32)))
            else:
                if len(batch_items) > src_batch_size:
                    raise ValueError("Incoming batch larger than src_batch_size.")
                for item in batch_items:
                    if isinstance(item, dict):
                        if input_key not in item:
                            raise KeyError(f"ModernBertSequencePacker missing key: {input_key}")
                        seq = np.asarray(item[input_key], dtype=np.int32)
                    else:
                        seq = np.asarray(item, dtype=np.int32)
                    buffer.append(_trim_pad(seq))
            while True:
                created = _create_batch(buffer)
                if created is None:
                    break
                packed, cu_seqlens = created
                cu_arr = []
                max_seqlen = []
                for entry in cu_seqlens:
                    diffs = np.diff(entry)
                    max_len = int(np.max(diffs)) if diffs.size else 0
                    max_seqlen.append(max_len)
                    if pad_cu_seqlens_to is not None:
                        if len(entry) > pad_cu_seqlens_to:
                            raise ValueError("cu_seqlens exceeds pad_cu_seqlens_to.")
                        padded = np.full(
                            (pad_cu_seqlens_to,),
                            pad_cu_seqlens_value,
                            dtype=np.int32,
                        )
                        padded[: len(entry)] = np.asarray(entry, dtype=np.int32)
                        cu_arr.append(padded)
                    else:
                        cu_arr.append(np.asarray(entry, dtype=np.int32))
                cu_arr = np.stack(cu_arr, axis=0)
                max_seqlen = np.asarray(max_seqlen, dtype=np.int32)

                attention_mask = (packed != pad_token_id).astype(np.int64)
                if suppress_masking:
                    yield {
                        "input_ids": packed,
                        "labels": None,
                        "cu_seqlens": cu_arr,
                        "max_seqlen": max_seqlen,
                        "attention_mask": attention_mask,
                    }
                else:
                    masked, labels = _mlm_masking(packed, np_rng)
                    yield {
                        "input_ids": masked,
                        "labels": labels,
                        "cu_seqlens": cu_arr,
                        "max_seqlen": max_seqlen,
                        "attention_mask": attention_mask,
                    }

    return _pack


@gin.configurable(module="trax.data")
def AddPositionIds(seq_key="input_ids", out_key="position_ids", dtype="int64"):
    """Add position_ids for packed sequences."""
    dtype = np.int64 if dtype == "int64" else np.int32

    def _add(stream):
        for example in stream:
            if not isinstance(example, dict):
                raise ValueError("AddPositionIds expects dict examples.")
            seq = example.get(seq_key)
            if seq is None:
                raise KeyError(f"AddPositionIds missing key: {seq_key}")
            seq = np.asarray(seq)
            seq_len = seq.shape[-1]
            position_ids = np.arange(seq_len, dtype=dtype)
            position_ids = np.broadcast_to(position_ids, seq.shape)
            out = dict(example)
            out[out_key] = position_ids
            yield out

    return _add


@gin.configurable(module="trax.data")
def ToModelTuple(inputs_keys, labels_key="labels"):
    """Convert dict to ((inputs...), labels) tuple."""
    inputs_keys = tuple(inputs_keys)

    def _convert(stream):
        for example in stream:
            if not isinstance(example, dict):
                raise ValueError("ToModelTuple expects dict examples.")
            inputs = tuple(example[k] for k in inputs_keys)
            labels = example.get(labels_key)
            yield inputs, labels

    return _convert
