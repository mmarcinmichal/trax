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

"""Encoders for text data.

* TextEncoder: base class
* ByteTextEncoder: for ascii text
* TokenTextEncoder: with user-supplied vocabulary file
* SubwordTextEncoder: invertible
* BertEncoder: for compatible tokenizers with original bert
"""
import collections
import itertools
import math
import os
import re
import sys
import tempfile
import time
import unicodedata

import gin
import numpy as np
import six
import tensorflow as tf
import tensorflow_text as tft

from absl import logging

from trax.data.debugger import data_pipeline as debug_data_pipeline

# Reserved tokens for things like padding and EOS symbols.
PAD = "<pad>"
EOS = "<EOS>"
RESERVED_TOKENS = [PAD, EOS]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)  # Normally 0
EOS_ID = RESERVED_TOKENS.index(EOS)  # Normally 1
RESERVED_TOKENS_BYTES = [bytes(PAD, "ascii"), bytes(EOS, "ascii")]

# Regular expression for unescaping token strings.
# '\u' is converted to '_'
# '\\' is converted to '\'
# '\213;' is converted to unichr(213)
_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")
_ESCAPE_CHARS = set("\\_u;0123456789")

# This set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i)
    for i in range(sys.maxunicode)
    if (
        unicodedata.category(six.unichr(i)).startswith("L")
        or unicodedata.category(six.unichr(i)).startswith("N")
    )
)


# Unicode utility functions that work with Python 2 and 3
def native_to_unicode(s):
    if is_unicode(s):
        return s
    try:
        return to_unicode(s)
    except UnicodeDecodeError:
        res = to_unicode(s, ignore_errors=True)
        logging.info("Ignoring Unicode error, outputting: %s", res)
        return res


def is_unicode(s):
    return isinstance(s, six.text_type)


def to_unicode(s, ignore_errors=False):
    if is_unicode(s):
        return s
    error_mode = "ignore" if ignore_errors else "strict"
    return s.decode("utf-8", errors=error_mode)


def to_unicode_ignore_errors(s):
    return to_unicode(s, ignore_errors=True)


def to_unicode_utf8(s):
    return s.decode("utf-8")


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def strip_ids(ids, ids_to_strip):
    """Strip ids_to_strip from the end IDs."""
    ids = list(ids)
    while ids and ids[-1] in ids_to_strip:
        ids.pop()
    return ids


def _escape_token(token, alphabet):
    """Escape away underscores and OOV characters and append '_'.

    This allows the token to be expressed as the concatenation of a list
    of subtokens from the vocabulary. The underscore acts as a sentinel
    which allows us to invertibly concatenate multiple such lists.

    Args:
      token: A unicode string to be escaped.
      alphabet: A set of all characters in the vocabulary's alphabet.

    Returns:
      escaped_token: An escaped unicode string.

    Raises:
      ValueError: If the provided token is not unicode.
    """
    if not isinstance(token, six.text_type):
        raise ValueError("Expected string type for token, got %s" % type(token))

    token = token.replace("\\", "\\\\").replace("_", "\\u")
    ret = [c if c in alphabet and c != "\n" else r"\%d;" % ord(c) for c in token]
    return "".join(ret) + "_"


def _unescape_token(escaped_token):
    """Inverse of _escape_token().

    Args:
      escaped_token: a unicode string

    Returns:
      token: a unicode string
    """

    def match(m):
        if m.group(1) is None:
            return "_" if m.group(0) == "\\u" else "\\"

        try:
            return six.unichr(int(m.group(1)))
        except (ValueError, OverflowError) as _:
            return "\u3013"  # Unicode for undefined character.

    trimmed = escaped_token[:-1] if escaped_token.endswith("_") else escaped_token
    return _UNESCAPE_REGEX.sub(match, trimmed)


def _bert_is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _bert_is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _bert_is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _read_filepattern(filepattern, max_lines=None, split_on_newlines=True):
    """Reads files matching a wildcard pattern, yielding the contents.

    Args:
      filepattern: A wildcard pattern matching one or more files.
      max_lines: If set, stop reading after reading this many lines.
      split_on_newlines: A boolean. If true, then split files by lines and strip
          leading and trailing whitespace from each line. Otherwise, treat each
          file as a single string.

    Yields:
      The contents of the files as lines, if split_on_newlines is True, or
      the entire contents of each file if False.
    """
    filenames = sorted(tf.io.gfile.glob(filepattern))
    lines_read = 0
    for filename in filenames:
        with tf.io.gfile.GFile(filename) as f:
            if split_on_newlines:
                for line in f:
                    yield line.strip()
                    lines_read += 1
                    if max_lines and lines_read >= max_lines:
                        return

            else:
                if max_lines:
                    doc = []
                    for line in f:
                        doc.append(line)
                        lines_read += 1
                        if max_lines and lines_read >= max_lines:
                            yield "".join(doc)
                            return
                    yield "".join(doc)

                else:
                    yield f.read()


def corpus_token_counts(text_filepattern, corpus_max_lines, split_on_newlines=True):
    """Read the corpus and compute a dictionary of token counts.

    Args:
      text_filepattern: A pattern matching one or more files.
      corpus_max_lines: An integer; maximum total lines to read.
      split_on_newlines: A boolean. If true, then split files by lines and strip
          leading and trailing whitespace from each line. Otherwise, treat each
          file as a single string.

    Returns:
      a dictionary mapping token to count.
    """
    counts = collections.Counter()
    for doc in _read_filepattern(
        text_filepattern,
        max_lines=corpus_max_lines,
        split_on_newlines=split_on_newlines,
    ):
        counts.update(encode(doc))

    return counts


def vocab_token_counts(text_filepattern, max_lines):
    """Read a vocab file and return a dictionary of token counts.

    Reads a two-column CSV file of tokens and their frequency in a dataset. The
    tokens are presumed to be generated by encode() or the equivalent.

    Args:
      text_filepattern: A pattern matching one or more files.
      max_lines: An integer; maximum total lines to read.

    Returns:
      a dictionary mapping token to count.
    """
    ret = {}
    for i, line in enumerate(_read_filepattern(text_filepattern, max_lines=max_lines)):
        if "," not in line:
            logging.warning("Malformed vocab line #%d '%s'", i, line)
            continue

        token, count = line.rsplit(",", 1)
        ret[token] = int(count)

    return ret


def _get_vocab(vocab_type="subword", vocab_file=None, vocab_dir=None, extra_ids=0):
    """Gets the vocabulary object for tokenization; see tokenize for details."""
    if vocab_type not in ["char", "subword", "sentencepiece", "bert", "bert-lowercase"]:
        raise ValueError(
            'vocab_type must be "subword", "char", "sentencepiece", "bert" or "bert-lowercase" '
            f"but got {vocab_type}"
        )

    if vocab_type == "char":
        # Note that we set num_reserved_ids=0 below. We could instead pass
        # the value n_reserved_ids from tokenize here -- ByteTextEncoder does
        # exactly the same thing as tokenize above, ie., adds num_reserved_ids.
        return ByteTextEncoder(num_reserved_ids=0)

    vocab_dir = vocab_dir or "gs://trax-ml/vocabs/"
    path = os.path.join(vocab_dir, vocab_file)

    if vocab_type == "subword":
        return SubwordTextEncoder(path)

    if vocab_type == "bert":
        return BertEncoder(path, do_lower_case=False)

    if vocab_type == "bert-lowercase":
        return BertEncoder(path, do_lower_case=True)

    if vocab_type == "sentencepiece":
        return SentencePieceEncoder(path, extra_ids=extra_ids)


def vocab_size(vocab_type="subword", vocab_file=None, vocab_dir=None, n_reserved_ids=0):
    """Returns the size of the vocabulary (number of symbols used).

    This function can be used to set the size of the final layers of a model that
    needs to predict symbols from a given vocabulary. More precisely, if this
    function returns N then the last layer size should be set to at least N (it
    can be more). Note that this function does take reserved IDs into account.

    Args:
      vocab_type: Type of vocabulary, one of: 'subword', 'sentencepiece', 'char'.
      vocab_file: Name of the vocabulary file.
      vocab_dir: Directory which contains the vocabulary file.
      n_reserved_ids: An int, offset added so 0, ..., n_reserved_ids-1 are unused.

    Returns:
      An integer, the number of symbols used (including reserved IDs).
    """
    vocab = _get_vocab(vocab_type, vocab_file, vocab_dir)
    return vocab.vocab_size + n_reserved_ids


"""
    Encoders and decoders
"""


def encode(text):
    """Encode a unicode string as a list of tokens.

    Args:
      text: a unicode string
    Returns:
      a list of tokens as Unicode strings
    """
    if not text:
        return []
    ret = []
    token_start = 0
    # Classify each character in the input string
    is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
    for pos in range(1, len(text)):
        if is_alnum[pos] != is_alnum[pos - 1]:
            token = text[token_start:pos]
            if token != " " or token_start == 0:
                ret.append(token)
            token_start = pos
    final_token = text[token_start:]
    ret.append(final_token)
    return ret


def decode(tokens):
    """Decode a list of tokens to a unicode string.

    Args:
      tokens: a list of Unicode strings
    Returns:
      a unicode string
    """
    token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
    ret = []
    for i, token in enumerate(tokens):
        if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
            ret.append(" ")
        ret.append(token)
    return "".join(ret)


class TextEncoder:
    """Base class for converting from ints to/from human readable strings."""

    def __init__(self, num_reserved_ids=NUM_RESERVED_TOKENS):
        self._num_reserved_ids = num_reserved_ids

    @property
    def num_reserved_ids(self):
        return self._num_reserved_ids

    def encode(self, s):
        """Transform a human-readable string into a sequence of int IDs.

        The IDs should be in the range [num_reserved_ids, vocab_size). IDs [0,
        num_reserved_ids) are reserved.

        EOS is not appended.

        Args:
          s: human-readable string to be converted.

        Returns:
          ids: list of integers
        """
        return [int(w) + self._num_reserved_ids for w in s.split()]

    def decode(self, ids, strip_extraneous=False):
        """Transform a sequence of int IDs into a human-readable string.

        EOS is not expected in IDs.

        Args:
          ids: list of integers to be converted.
          strip_extraneous: bool, whether to strip off extraneous tokens (EOS and
            PAD).

        Returns:
          s: human-readable string.
        """
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        return " ".join(self.decode_list(ids))

    def decode_list(self, ids):
        """Transform a sequence of int IDs into a their string versions.

        This method supports transforming individual input/output IDs to their
        string versions so that sequence to/from text conversions can be visualized
        in a human readable format.

        Args:
          ids: list of integers to be converted.

        Returns:
          strs: list of human-readable string.
        """
        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < self._num_reserved_ids:
                decoded_ids.append(RESERVED_TOKENS[int(id_)])
            else:
                decoded_ids.append(id_ - self._num_reserved_ids)
        return [str(d) for d in decoded_ids]

    @property
    def vocab_size(self):
        raise NotImplementedError()


class ByteTextEncoder(TextEncoder):
    """Encodes each byte to an id. For 8-bit strings only."""

    def encode(self, s):
        numres = self._num_reserved_ids
        # Python3: explicitly convert to UTF-8
        return [c + numres for c in s.encode("utf-8")]

    def decode(self, ids, strip_extraneous=False):
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        numres = self._num_reserved_ids
        decoded_ids = []
        int2byte = six.int2byte
        for id_ in ids:
            if 0 <= id_ < numres:
                decoded_ids.append(RESERVED_TOKENS_BYTES[int(id_)])
            else:
                decoded_ids.append(int2byte(id_ - numres))
        # Python3: join byte arrays and then decode string
        return b"".join(decoded_ids).decode("utf-8", "replace")

    def decode_list(self, ids):
        numres = self._num_reserved_ids
        decoded_ids = []
        int2byte = six.int2byte
        for id_ in ids:
            if 0 <= id_ < numres:
                decoded_ids.append(RESERVED_TOKENS_BYTES[int(id_)])
            else:
                decoded_ids.append(int2byte(id_ - numres))
        # Python3: join byte arrays and then decode string
        return decoded_ids

    @property
    def vocab_size(self):
        return 2**8 + self._num_reserved_ids


class ClassLabelEncoder(TextEncoder):
    """Encoder for class labels."""

    def __init__(self, class_labels=None, class_labels_fname=None):
        super(ClassLabelEncoder, self).__init__(num_reserved_ids=0)

        if class_labels_fname:
            with tf.io.gfile.GFile(class_labels_fname) as f:
                class_labels = [label.strip() for label in f.readlines()]

        assert class_labels
        self._class_labels = class_labels

    def encode(self, s):
        label_str = s
        return self._class_labels.index(label_str)

    def decode(self, ids, strip_extraneous=False):
        del strip_extraneous
        label_id = ids
        if isinstance(label_id, list):
            assert len(label_id) == 1
            (label_id,) = label_id
        if isinstance(label_id, np.ndarray):
            label_id = np.squeeze(label_id)
        return self._class_labels[label_id]

    def decode_list(self, ids):
        return [self._class_labels[i] for i in ids]

    @property
    def vocab_size(self):
        return len(self._class_labels)


class OneHotClassLabelEncoder(ClassLabelEncoder):
    """One-hot encoder for class labels."""

    def encode(self, label_str, on_value=1, off_value=0):  # pylint: disable=arguments-differ
        e = np.full(self.vocab_size, off_value, dtype=np.int32)
        e[self._class_labels.index(label_str)] = on_value
        return e.tolist()

    def decode(self, ids, strip_extraneous=False):
        del strip_extraneous
        label_id = ids
        if isinstance(label_id, np.ndarray):
            label_id = np.squeeze(label_id).astype(np.int8).tolist()
        assert isinstance(label_id, list)
        assert len(label_id) == self.vocab_size
        return self._class_labels[label_id.index(1)]

    @property
    def vocab_size(self):
        return len(self._class_labels)


class TokenTextEncoder(TextEncoder):
    """Encoder based on a user-supplied vocabulary (file or list)."""

    def __init__(
        self,
        vocab_filename,
        reverse=False,
        vocab_list=None,
        replace_oov=None,
        num_reserved_ids=NUM_RESERVED_TOKENS,
    ):
        """Initialize from a file or list, one token per line.

        Handling of reserved tokens works as follows:
        - When initializing from a list, we add reserved tokens to the vocab.
        - When initializing from a file, we do not add reserved tokens to the vocab.
        - When saving vocab files, we save reserved tokens to the file.

        Args:
          vocab_filename: If not None, the full filename to read vocab from. If this
            is not None, then vocab_list should be None.
          reverse: Boolean indicating if tokens should be reversed during encoding
            and decoding.
          vocab_list: If not None, a list of elements of the vocabulary. If this is
            not None, then vocab_filename should be None.
          replace_oov: If not None, every out-of-vocabulary token seen when encoding
            will be replaced by this string (which must be in vocab).
          num_reserved_ids: Number of IDs to save for reserved tokens like <EOS>.
        """
        super(TokenTextEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        if vocab_filename:
            self._init_vocab_from_file(vocab_filename)
        else:
            assert vocab_list is not None
            self._init_vocab_from_list(vocab_list)

    def encode(self, s):
        """Converts a space-separated string of tokens to a list of ids."""
        sentence = s
        tokens = sentence.strip().split()
        if self._replace_oov is not None:
            tokens = [
                t if t in self._token_to_id else self._replace_oov for t in tokens
            ]
        ret = [self._token_to_id[tok] for tok in tokens]
        return ret[::-1] if self._reverse else ret

    def decode(self, ids, strip_extraneous=False):
        return " ".join(self.decode_list(ids))

    def decode_list(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return [self._safe_id_to_token(i) for i in seq]

    @property
    def vocab_size(self):
        return len(self._id_to_token)

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, "ID_%d" % idx)

    def _init_vocab_from_file(self, filename):
        """Load vocab from a file.

        Args:
          filename: The file to load vocabulary from.
        """
        with tf.io.gfile.GFile(filename) as f:
            tokens = [token.strip() for token in f.readlines()]

        def token_gen():
            for token in tokens:
                yield token

        self._init_vocab(token_gen(), add_reserved_tokens=False)

    def _init_vocab_from_list(self, vocab_list):
        """Initialize tokens from a list of tokens.

        It is ok if reserved tokens appear in the vocab list. They will be
        removed. The set of tokens in vocab_list should be unique.

        Args:
          vocab_list: A list of tokens.
        """

        def token_gen():
            for token in vocab_list:
                if token not in RESERVED_TOKENS:
                    yield token

        self._init_vocab(token_gen())

    def _init_vocab(self, token_generator, add_reserved_tokens=True):
        """Initialize vocabulary with tokens from token_generator."""

        self._id_to_token = {}
        non_reserved_start_index = 0

        if add_reserved_tokens:
            self._id_to_token.update(enumerate(RESERVED_TOKENS))
            non_reserved_start_index = len(RESERVED_TOKENS)

        self._id_to_token.update(
            enumerate(token_generator, start=non_reserved_start_index)
        )

        # _token_to_id is the reverse of _id_to_token
        self._token_to_id = dict((v, k) for k, v in six.iteritems(self._id_to_token))

    def store_to_file(self, filename):
        """Write vocab file to disk.

        Vocab files have one token per line. The file ends in a newline. Reserved
        tokens are written to the vocab file as well.

        Args:
          filename: Full path of the file to store the vocab to.
        """
        with tf.io.gfile.GFile(filename, "w") as f:
            for i in range(len(self._id_to_token)):
                f.write(self._id_to_token[i] + "\n")


class SubwordTextEncoder(TextEncoder):
    """Class for invertibly encoding text using a limited vocabulary.

    Invertibly encodes a native string as a sequence of subtokens from a limited
    vocabulary.

    A SubwordTextEncoder is built from a corpus (so it is tailored to the text in
    the corpus), and stored to a file. See subword.py.

    It can then be loaded and used to encode/decode any text.

    Encoding has four phases:

    1. Tokenize into a list of tokens.  Each token is a unicode string of either
       all alphanumeric characters or all non-alphanumeric characters.  We drop
       tokens consisting of a single space that are between two alphanumeric
       tokens.

    2. Escape each token.  This escapes away special and out-of-vocabulary
       characters, and makes sure that each token ends with an underscore, and
       has no other underscores.

    3. Represent each escaped token as a the concatenation of a list of subtokens
       from the limited vocabulary.  Subtoken selection is done greedily from
       beginning to end.  That is, we construct the list in order, always picking
       the longest subtoken in our vocabulary that matches a prefix of the
       remaining portion of the encoded token.

    4. Concatenate these lists.  This concatenation is invertible due to the
       fact that the trailing underscores indicate when one list is finished.

    """

    def __init__(self, filename=None):
        """Initialize and read from a file, if provided.

        Args:
          filename: filename from which to read vocab. If None, do not load a vocab
        """
        self._alphabet = set()
        self.filename = filename
        if filename is not None:
            self._load_from_file(filename)
        super(SubwordTextEncoder, self).__init__()

    def encode(self, s):
        """Converts a native string to a list of subtoken IDs.

        Args:
          s: a native string.

        Returns:
          a list of integers in the range [0, vocab_size)
        """
        return self._tokens_to_subtoken_ids(encode(native_to_unicode(s)))

    def encode_without_tokenizing(self, token_text):
        """Converts string to list of subtoken IDs without calling tokenizer.

        This treats `token_text` as a single token and directly converts it
        to subtoken IDs. This may be useful when the default tokenizer doesn't
        do what we want (e.g., when encoding text with tokens composed of lots of
        nonalphanumeric characters). It is then up to the caller to make sure that
        raw text is consistently converted into tokens. Only use this if you are
        sure that `encode` doesn't suit your needs.

        Args:
          token_text: A native string representation of a single token.

        Returns:
          A list of subword token IDs; i.e., integers in the range [0, vocab_size).
        """
        return self._tokens_to_subtoken_ids([native_to_unicode(token_text)])

    def decode(self, ids, strip_extraneous=False):
        """Converts a sequence of subtoken IDs to a native string.

        Args:
          ids: a list of integers in the range [0, vocab_size)
          strip_extraneous: bool, whether to strip off extraneous tokens (EOS and
            PAD).

        Returns:
          a native string
        """
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        return decode(self._subtoken_ids_to_tokens(ids))

    def decode_list(self, ids):
        return [self._subtoken_id_to_subtoken_string(s) for s in ids]

    @property
    def vocab_size(self):
        """The subtoken vocabulary size."""
        return len(self._all_subtoken_strings)

    def _tokens_to_subtoken_ids(self, tokens):
        """Converts a list of tokens to a list of subtoken IDs.

        Args:
          tokens: a list of strings.

        Returns:
          a list of integers in the range [0, vocab_size)
        """
        ret = []
        for token in tokens:
            ret.extend(self._token_to_subtoken_ids(token))
        return ret

    def _token_to_subtoken_ids(self, token):
        """Converts token to a list of subtoken IDs.

        Args:
          token: a string.

        Returns:
          a list of integers in the range [0, vocab_size)
        """
        cache_location = hash(token) % self._cache_size
        cache_key, cache_value = self._cache[cache_location]
        if cache_key == token:
            return cache_value
        ret = self._escaped_token_to_subtoken_ids(_escape_token(token, self._alphabet))
        self._cache[cache_location] = (token, ret)
        return ret

    def _subtoken_ids_to_tokens(self, subtokens):
        """Converts a list of subtoken IDs to a list of tokens.

        Args:
          subtokens: a list of integers in the range [0, vocab_size)

        Returns:
          a list of strings.
        """
        concatenated = "".join(
            [self._subtoken_id_to_subtoken_string(s) for s in subtokens]
        )
        split = concatenated.split("_")
        ret = []
        for t in split:
            if t:
                unescaped = _unescape_token(t + "_")
                if unescaped:
                    ret.append(unescaped)
        return ret

    def _subtoken_id_to_subtoken_string(self, subtoken):
        """Converts a subtoken integer ID to a subtoken string."""
        if 0 <= subtoken < self.vocab_size:
            return self._all_subtoken_strings[subtoken]
        return ""

    def _escaped_token_to_subtoken_strings(self, escaped_token):
        """Converts an escaped token string to a list of subtoken strings.

        Args:
          escaped_token: An escaped token as a unicode string.

        Returns:
          A list of subtokens as unicode strings.
        """
        # NOTE: This algorithm is greedy; it won't necessarily produce the "best"
        # list of subtokens.
        ret = []
        start = 0
        token_len = len(escaped_token)
        while start < token_len:
            for end in range(min(token_len, start + self._max_subtoken_len), start, -1):
                subtoken = escaped_token[start:end]
                if subtoken in self._subtoken_string_to_id:
                    ret.append(subtoken)
                    start = end
                    break

            else:  # Did not break
                # If there is no possible encoding of the escaped token then one of the
                # characters in the token is not in the alphabet. This should be
                # impossible and would be indicative of a bug.
                assert False, "Token substring not found in subtoken vocabulary."

        return ret

    def _escaped_token_to_subtoken_ids(self, escaped_token):
        """Converts an escaped token string to a list of subtoken IDs.

        Args:
          escaped_token: An escaped token as a unicode string.

        Returns:
          A list of subtoken IDs as integers.
        """
        return [
            self._subtoken_string_to_id[subtoken]
            for subtoken in self._escaped_token_to_subtoken_strings(escaped_token)
        ]

    @classmethod
    def build_from_generator(
        cls, generator, target_size, max_subtoken_length=None, reserved_tokens=None
    ):
        """Builds a SubwordTextEncoder from the generated text.

        Args:
          generator: yields text.
          target_size: int, approximate vocabulary size to create.
          max_subtoken_length: Maximum length of a subtoken. If this is not set,
            then the runtime and memory use of creating the vocab is quadratic in
            the length of the longest token. If this is set, then it is instead
            O(max_subtoken_length * length of longest token).
          reserved_tokens: List of reserved tokens. The global variable
            `RESERVED_TOKENS` must be a prefix of `reserved_tokens`. If this
            argument is `None`, it will use `RESERVED_TOKENS`.

        Returns:
          SubwordTextEncoder with `vocab_size` approximately `target_size`.
        """
        token_counts = collections.defaultdict(int)
        for item in generator:
            for tok in encode(native_to_unicode(item)):
                token_counts[tok] += 1
        encoder = cls.build_to_target_size(
            target_size,
            token_counts,
            1,
            1e3,
            max_subtoken_length=max_subtoken_length,
            reserved_tokens=reserved_tokens,
        )
        return encoder

    @classmethod
    def build_to_target_size(
        cls,
        target_size,
        token_counts,
        min_val,
        max_val,
        max_subtoken_length=None,
        reserved_tokens=None,
        num_iterations=4,
    ):
        """Builds a SubwordTextEncoder that has `vocab_size` near `target_size`.

        Uses simple recursive binary search to find a minimum token count that most
        closely matches the `target_size`.

        Args:
          target_size: Desired vocab_size to approximate.
          token_counts: A dictionary of token counts, mapping string to int.
          min_val: An integer; lower bound for the minimum token count.
          max_val: An integer; upper bound for the minimum token count.
          max_subtoken_length: Maximum length of a subtoken. If this is not set,
            then the runtime and memory use of creating the vocab is quadratic in
            the length of the longest token. If this is set, then it is instead
            O(max_subtoken_length * length of longest token).
          reserved_tokens: List of reserved tokens. The global variable
            `RESERVED_TOKENS` must be a prefix of `reserved_tokens`. If this
            argument is `None`, it will use `RESERVED_TOKENS`.
          num_iterations: An integer; how many iterations of refinement.

        Returns:
          A SubwordTextEncoder instance.

        Raises:
          ValueError: If `min_val` is greater than `max_val`.
        """
        if min_val > max_val:
            raise ValueError(
                "Lower bound for the minimum token count "
                "is greater than the upper bound."
            )
        if target_size < 1:
            raise ValueError("Target size must be positive.")

        if reserved_tokens is None:
            reserved_tokens = RESERVED_TOKENS

        def bisect(min_val, max_val):
            """Bisection to find the right size."""
            present_count = (max_val + min_val) // 2
            logging.info("Trying min_count %d", present_count)
            subtokenizer = cls()
            subtokenizer.build_from_token_counts(
                token_counts,
                present_count,
                num_iterations,
                max_subtoken_length=max_subtoken_length,
                reserved_tokens=reserved_tokens,
            )

            # Being within 1% of the target size is ok.
            is_ok = abs(subtokenizer.vocab_size - target_size) * 100 < target_size
            # If min_val == max_val, we can't do any better than this.
            if is_ok or min_val >= max_val or present_count < 2:
                return subtokenizer

            if subtokenizer.vocab_size > target_size:
                other_subtokenizer = bisect(present_count + 1, max_val)
            else:
                other_subtokenizer = bisect(min_val, present_count - 1)

            if other_subtokenizer is None:
                return subtokenizer

            if abs(other_subtokenizer.vocab_size - target_size) < abs(
                subtokenizer.vocab_size - target_size
            ):
                return other_subtokenizer
            return subtokenizer

        return bisect(min_val, max_val)

    def build_from_token_counts(
        self,
        token_counts,
        min_count,
        num_iterations=4,
        reserved_tokens=None,
        max_subtoken_length=None,
    ):
        """Train a SubwordTextEncoder based on a dictionary of word counts.

        Args:
          token_counts: a dictionary of Unicode strings to int.
          min_count: an integer - discard subtokens with lower counts.
          num_iterations: an integer.  how many iterations of refinement.
          reserved_tokens: List of reserved tokens. The global variable
            `RESERVED_TOKENS` must be a prefix of `reserved_tokens`. If this
            argument is `None`, it will use `RESERVED_TOKENS`.
          max_subtoken_length: Maximum length of a subtoken. If this is not set,
            then the runtime and memory use of creating the vocab is quadratic in
            the length of the longest token. If this is set, then it is instead
            O(max_subtoken_length * length of longest token).

        Raises:
          ValueError: if reserved is not 0 or len(RESERVED_TOKENS). In this case, it
            is not clear what the space is being reserved for, or when it will be
            filled in.
        """
        if reserved_tokens is None:
            reserved_tokens = RESERVED_TOKENS
        else:
            # There is not complete freedom in replacing RESERVED_TOKENS.
            for default, proposed in zip(RESERVED_TOKENS, reserved_tokens):
                if default != proposed:
                    raise ValueError(
                        "RESERVED_TOKENS must be a prefix of " "reserved_tokens."
                    )

        # Initialize the alphabet. Note, this must include reserved tokens or it can
        # result in encoding failures.
        alphabet_tokens = itertools.chain(
            six.iterkeys(token_counts), [native_to_unicode(t) for t in reserved_tokens]
        )

        self._init_alphabet_from_tokens(alphabet_tokens)

        # Bootstrap the initial list of subtokens with the characters from the
        # alphabet plus the escaping characters.
        self._init_subtokens_from_list(
            list(self._alphabet), reserved_tokens=reserved_tokens
        )

        # We build iteratively.  On each iteration, we segment all the words,
        # then count the resulting potential subtokens, keeping the ones
        # with high enough counts for our new vocabulary.
        if min_count < 1:
            min_count = 1
        for i in range(num_iterations):
            logging.info("Iteration %d", i)

            # Collect all substrings of the encoded token that break along current
            # subtoken boundaries.
            subtoken_counts = collections.defaultdict(int)
            for token, count in six.iteritems(token_counts):
                iter_start_time = time.time()
                escaped_token = _escape_token(token, self._alphabet)
                subtokens = self._escaped_token_to_subtoken_strings(escaped_token)
                start = 0
                for subtoken in subtokens:
                    last_position = len(escaped_token) + 1
                    if max_subtoken_length is not None:
                        last_position = min(last_position, start + max_subtoken_length)

                    for end in range(start + 1, last_position):
                        new_subtoken = escaped_token[start:end]
                        subtoken_counts[new_subtoken] += count
                    start += len(subtoken)
                iter_time_secs = time.time() - iter_start_time
                if iter_time_secs > 0.1:
                    logging.info(
                        "Processing token [%s] took {%d} seconds, consider "
                        "setting Text2TextProblem.max_subtoken_length to a "
                        "smaller value.",
                        token,
                        iter_time_secs,
                    )

            # Array of sets of candidate subtoken strings, by length.
            len_to_subtoken_strings = []
            for subtoken_string, count in six.iteritems(subtoken_counts):
                lsub = len(subtoken_string)
                if count >= min_count:
                    while len(len_to_subtoken_strings) <= lsub:
                        len_to_subtoken_strings.append(set())
                    len_to_subtoken_strings[lsub].add(subtoken_string)

            # Consider the candidates longest to shortest, so that if we accept
            # a longer subtoken string, we can decrement the counts of its prefixes.
            new_subtoken_strings = []
            for lsub in range(len(len_to_subtoken_strings) - 1, 0, -1):
                subtoken_strings = len_to_subtoken_strings[lsub]
                for subtoken_string in subtoken_strings:
                    count = subtoken_counts[subtoken_string]
                    if count >= min_count:
                        # Exclude alphabet tokens here, as they must be included later,
                        # explicitly, regardless of count.
                        if subtoken_string not in self._alphabet:
                            new_subtoken_strings.append((count, subtoken_string))
                        for l in range(1, lsub):
                            subtoken_counts[subtoken_string[:l]] -= count

            # Include the alphabet explicitly to guarantee all strings are encodable.
            new_subtoken_strings.extend(
                (subtoken_counts.get(a, 0), a) for a in self._alphabet
            )
            new_subtoken_strings.sort(reverse=True)

            # Reinitialize to the candidate vocabulary.
            new_subtoken_strings = [subtoken for _, subtoken in new_subtoken_strings]
            if reserved_tokens:
                escaped_reserved_tokens = [
                    _escape_token(native_to_unicode(t), self._alphabet)
                    for t in reserved_tokens
                ]
                new_subtoken_strings = escaped_reserved_tokens + new_subtoken_strings

            self._init_subtokens_from_list(new_subtoken_strings)
            logging.info("vocab_size = %d", self.vocab_size)

    @property
    def all_subtoken_strings(self):
        return tuple(self._all_subtoken_strings)

    def dump(self):
        """Debugging dump of the current subtoken vocabulary."""
        subtoken_strings = [
            (i, s) for s, i in six.iteritems(self._subtoken_string_to_id)
        ]
        print(
            ", ".join("{0} : '{1}'".format(i, s) for i, s in sorted(subtoken_strings))
        )

    def _init_subtokens_from_list(self, subtoken_strings, reserved_tokens=None):
        """Initialize token information from a list of subtoken strings.

        Args:
          subtoken_strings: a list of subtokens
          reserved_tokens: List of reserved tokens. We must have `reserved_tokens`
            as None or the empty list, or else the global variable `RESERVED_TOKENS`
            must be a prefix of `reserved_tokens`.

        Raises:
          ValueError: if reserved is not 0 or len(RESERVED_TOKENS). In this case, it
            is not clear what the space is being reserved for, or when it will be
            filled in.
        """
        if reserved_tokens is None:
            reserved_tokens = []

        if reserved_tokens:
            self._all_subtoken_strings = reserved_tokens + subtoken_strings
        else:
            self._all_subtoken_strings = subtoken_strings

        # we remember the maximum length of any subtoken to avoid having to
        # check arbitrarily long strings.
        self._max_subtoken_len = max([len(s) for s in subtoken_strings])
        self._subtoken_string_to_id = {
            s: i + len(reserved_tokens) for i, s in enumerate(subtoken_strings) if s
        }
        # Initialize the cache to empty.
        self._cache_size = 2**20
        self._cache = [(None, None)] * self._cache_size

    def _init_alphabet_from_tokens(self, tokens):
        """Initialize alphabet from an iterable of token or subtoken strings."""
        # Include all characters from all tokens in the alphabet to guarantee that
        # any token can be encoded. Additionally, include all escaping characters.
        self._alphabet = {c for token in tokens for c in token}  # pylint: disable=g-complex-comprehension
        self._alphabet |= _ESCAPE_CHARS

    def _load_from_file_object(self, f):
        """Load from a file object.

        Args:
          f: File object to load vocabulary from
        """
        subtoken_strings = []
        for line in f:
            s = line.rstrip()
            # Some vocab files wrap words in single quotes, but others don't
            if (s.startswith("'") and s.endswith("'")) or (
                s.startswith('"') and s.endswith('"')
            ):
                s = s[1:-1]
            subtoken_strings.append(native_to_unicode(s))
        self._init_subtokens_from_list(subtoken_strings)
        self._init_alphabet_from_tokens(subtoken_strings)

    def _load_from_file(self, filename):
        """Load from a vocab file."""
        if not tf.io.gfile.exists(filename):
            raise ValueError("File %s not found" % filename)
        with tf.io.gfile.GFile(filename) as f:
            self._load_from_file_object(f)

    def store_to_file(self, filename, add_single_quotes=True):
        with tf.io.gfile.GFile(filename, "w") as f:
            for subtoken_string in self._all_subtoken_strings:
                if add_single_quotes:
                    f.write("'" + subtoken_string + "'\n")
                else:
                    f.write(subtoken_string + "\n")


class ImageEncoder:
    """Encoder class for saving and loading images."""

    def __init__(self, num_reserved_ids=0, height=None, width=None, channels=3):
        assert num_reserved_ids == 0
        self._height = height
        self._width = width
        self._channels = channels

    @property
    def num_reserved_ids(self):
        return 0

    def encode(self, s):
        """Transform a string with a filename into a list of RGB integers.

        Args:
          s: path to the file with an image.

        Returns:
          ids: list of integers
        """
        try:
            import matplotlib.image as im  # pylint: disable=g-import-not-at-top
        except ImportError as e:
            logging.warning(
                "Reading an image requires matplotlib to be installed: %s", e
            )
            raise NotImplementedError("Image reading not implemented.")
        return im.imread(s)

    def decode(self, ids, strip_extraneous=False):
        """Transform a sequence of int IDs into an image file.

        Args:
          ids: list of integers to be converted.
          strip_extraneous: unused

        Returns:
          Path to the temporary file where the image was saved.

        Raises:
          ValueError: if the IDs are not of the appropriate size.
        """
        del strip_extraneous
        _, tmp_file_path = tempfile.mkstemp("_decode.png")
        if self._height is None or self._width is None:
            size = int(math.sqrt(len(ids) / self._channels))
            length = size * size * self._channels
        else:
            size = None
            length = self._height * self._width * self._channels
        if len(ids) != length:
            raise ValueError(
                "Length of ids (%d) must be height (%d) x width (%d) x "
                "channels (%d); %d != %d.\n Ids: %s"
                % (
                    len(ids),
                    self._height,
                    self._width,
                    self._channels,
                    len(ids),
                    length,
                    " ".join([str(i) for i in ids]),
                )
            )
        with tf.Graph().as_default():
            raw = tf.constant(ids, dtype=tf.uint8)
            if size is None:
                img = tf.reshape(raw, [self._height, self._width, self._channels])
            else:
                img = tf.reshape(raw, [size, size, self._channels])
            png = tf.image.encode_png(img)
            op = tf.write_file(tmp_file_path, png)
            with tf.Session() as sess:
                sess.run(op)
        return tmp_file_path

    def decode_list(self, ids):
        """Transform a sequence of int IDs into an image file.

        Args:
          ids: list of integers to be converted.

        Returns:
          Singleton list: path to the temporary file where the image was saved.
        """
        return [self.decode(ids)]

    @property
    def vocab_size(self):
        return 256


class RealEncoder:
    """Encoder class for saving and loading float values."""

    def encode(self, s):
        """Transform a string (space separated float values) into a float array.

        Args:
          s: space separated float values.

        Returns:
          Array of float values.
        """
        return [float(w) for w in s.split()]

    def decode(self, ids, strip_extraneous=False):
        """Transform sequence of float values into string (float values).

        Args:
          ids: array of floats to be converted.
          strip_extraneous: unused

        Returns:
          String having space separated float values.

        Raises:
          ValueError: if the IDs are not of the appropriate size.
        """
        del strip_extraneous
        return " ".join([str(i) for i in ids])


class BertEncoder:
    """Encoder Class that is compatible with models trained in original BERT library."""

    def __init__(self, vocab_file, do_lower_case=True):
        self._vocab = self.load_vocab(vocab_file)
        self._inv_vocab = {v: k for k, v in self._vocab.items()}
        self._basic_tokenizer = BertBasicEncoder(do_lower_case=do_lower_case)
        self._wordpiece_tokenizer = BertWordpieceTokenizer(vocab=self._vocab)

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        index = 0
        with tf.io.gfile.GFile(vocab_file, "r") as reader:
            while True:
                token = native_to_unicode(reader.readline())
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

    def encode(self, text):
        return self._convert_tokens_to_ids(self.tokenize(text))

    # Note: Because encoding by BertEncoder is not unique text decoded
    # from token ids is not unique.
    def decode(self, ids):
        """Returns a text that encoded would yield provided ids."""
        tokens = self._convert_ids_to_tokens(ids)
        if not tokens:
            return ""
        retarr = [tokens[0]]
        for token in tokens[1:]:
            if token.startswith("##"):
                retarr.append(token.lstrip("#"))
            else:
                retarr.append(" ")
                retarr.append(token)
        return "".join(retarr)

    @property
    def vocab_size(self):
        return len(self._vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self._basic_tokenizer.tokenize(text):
            for sub_token in self._wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def _convert_tokens_to_ids(self, tokens):
        return [self._vocab[token] for token in tokens]

    def _convert_ids_to_tokens(self, ids):
        return [self._inv_vocab[token_id] for token_id in ids]


class BertBasicEncoder:
    """Part of BertEncoder; tokenization (punctuation splitting, lower casing)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = native_to_unicode(text)
        text = self._clean_text(text)

        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _bert_is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _bert_is_control(char):
                continue
            if _bert_is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class SentencePieceEncoder:
    """SentencePiece tokenizer with support for extra_ids like in T5."""

    def __init__(self, spm_path, extra_ids=0):
        with tf.io.gfile.GFile(spm_path, "rb") as f:
            sp_model = f.read()
        self.tokenizer = tft.SentencepieceTokenizer(model=sp_model)
        self.extra_ids = extra_ids
        # Note: We assume vocab size includes EOS, PAD, etc.
        self.vocab_size = self.tokenizer.vocab_size().numpy()
        self.total_vocab_size = self.vocab_size + self.extra_ids

    def encode(self, text):
        # Tokenize the text into base SentencePiece IDs
        tokens = self.tokenizer.tokenize([text])
        # Back to numpy from tf
        print(text)
        return tokens.flat_values

    def decode(self, ids):
        # Convert IDs back to text, handling extra_ids if needed
        ids = [i for i in ids if i < self.vocab_size]
        ids_tensor = tf.constant([ids], dtype=tf.int32)
        text = self.tokenizer.detokenize(ids_tensor)
        # Back to numpy from tf
        print(text)
        return text.numpy()[0].decode("utf-8")


"""A simple invertible tokenizer.

Converts from a unicode string to a list of tokens
(represented as Unicode strings).

This tokenizer has the following desirable properties:
 - It is invertible.
 - Alphanumeric characters are broken away from non-alphanumeric characters.
 - A single space between words does not produce an extra token.
 - The full Unicode punctuation and separator set is recognized.

The tokenization algorithm is as follows:

1.  Split the text into a list of tokens, splitting at every boundary of an
    alphanumeric character and a non-alphanumeric character.  This produces
    a list which alternates between "alphanumeric tokens"
    (strings of alphanumeric characters) and "non-alphanumeric tokens"
    (strings of non-alphanumeric characters).

2.  Remove every token consisting of a single space, unless it is
    the very first or very last token in the list.  These tokens are now
    implied by the fact that there are two adjacent alphanumeric tokens.

e.g.  u"Dude - that's so cool."
        -> [u"Dude", u" - ", u"that", u"'", u"s", u"so", u"cool", u"."]
"""


# Tokenization.
@debug_data_pipeline.debug_pipeline
def tokenize(
    stream,
    keys=None,
    vocab_type="subword",
    vocab_file=None,
    vocab_dir=None,
    n_reserved_ids=0,
):
    """Tokenize examples from the stream.

    This function assumes that `stream` generates either strings or tuples/dicts
    containing strings at some `keys`. This function maps these strings to
    numpy arrays of integers -- the tokenized version of each string.

    Args:
      stream: A python generator yielding strings, tuples or dicts.
      keys: which keys of the tuple/dict to tokenize (by default: all)
      vocab_type: Type of vocabulary, one of: 'subword', 'sentencepiece', 'char'.
      vocab_file: Name of the vocabulary file.
      vocab_dir: Directory which contains the vocabulary file.
      n_reserved_ids: An int, offset added so 0, ..., n_reserved_ids-1 are unused;
        This is common for example when reserving the 0 for padding and 1 for EOS,
        but it's only needed if these symbols are not already included (and thus
        reserved) in the vocab_file.

    Yields:
      Examples from stream with strings at `keys` replaced by np.arrays of
      integers -- the tokenized version of these strings.
    """
    vocab = _get_vocab(vocab_type, vocab_file, vocab_dir)
    for example in stream:
        if isinstance(example, (list, tuple)):
            new_example = []
            for i, x in enumerate(example):
                if keys is None or i in keys:
                    new_example.append(np.array(vocab.encode(x)) + n_reserved_ids)
                else:
                    new_example.append(x)
            output = tuple(new_example)
            yield output
        elif isinstance(example, dict):
            new_example = {}
            for k in example:
                if keys is None or k in keys:
                    new_example[k] = np.array(vocab.encode(example[k])) + n_reserved_ids
                else:
                    new_example[k] = example[k]
            yield new_example
        else:
            output = np.array(vocab.encode(example)) + n_reserved_ids
            yield output


@gin.configurable(module="trax.data")
def Tokenize(  # pylint: disable=invalid-name
    keys=None,
    vocab_type="subword",  # pylint: disable=invalid-name
    vocab_file=None,
    vocab_dir=None,
    n_reserved_ids=0,
):
    """Returns a function that maps text to integer arrays; see `tokenize`."""
    return lambda g: tokenize(  # pylint: disable=g-long-lambda
        g,
        keys=keys,
        vocab_type=vocab_type,
        vocab_file=vocab_file,
        vocab_dir=vocab_dir,
        n_reserved_ids=n_reserved_ids,
    )


def detokenize(
    x, vocab_type="subword", vocab_file=None, vocab_dir=None, n_reserved_ids=0
):
    """Maps integer arrays to text; the opposite of `tokenize`.

    In many cases (all char- and subword-type vocabularies and most sentencepiece
    ones) the tokenization is invertible, so detokenize(tokenize(x)) = x. In some
    more rare cases this can remove some spacing, but it is still often useful
    to run detokenize to get a readable version for a tokenized string.

    Args:
      x: a list or numpy array of integers.
      vocab_type: Type of vocabulary, one of: 'subword', 'sentencepiece', 'char'.
      vocab_file: Name of the vocabulary file.
      vocab_dir: Directory which contains the vocabulary file.
      n_reserved_ids: An int, offset added so 0, ..., n_reserved_ids-1 are unused;
        This is common for example when reserving the 0 for padding and 1 for EOS,
        but it's only needed if these symbols are not already included (and thus
        reserved) in the vocab_file.

    Returns:
      A string corresponding to the de-tokenized version of x.
    """
    vocab = _get_vocab(vocab_type, vocab_file, vocab_dir)
    x_unreserved = np.array(x) - n_reserved_ids
    return str(vocab.decode(x_unreserved.tolist()))


@gin.configurable(module="trax.data")
def SentencePieceTokenizer(spm_path=None, extra_ids=0):
    """
    Returns a generator function that tokenizes a stream of text using
    SentencePiece and supports extra IDs.

    Args:
      spm_path: Path to the SentencePiece model file. Must be provided.
      extra_ids: Number of extra IDs to reserve.

    Returns:
      A function that takes a generator of text examples and yields tokenized
      numpy arrays.
    """
    if spm_path is None:
        raise ValueError("spm_path must be provided.")

    def tokenize(stream, spm_path, extra_ids):
        vocab_file = os.path.basename(spm_path)
        vocab_dir = os.path.dirname(spm_path)
        vocab = _get_vocab(
            vocab_type="sentencepiece",
            vocab_file=vocab_file,
            vocab_dir=vocab_dir,
            extra_ids=extra_ids,
        )
        for example in stream:
            # Optionally replace print with logging.debugger
            # logging.debugger("Tokenizing example: %s", example)
            if isinstance(example, tuple):
                example = example[0]
            yield np.array(vocab.encode(example), dtype=np.int64)

    return lambda g: tokenize(g, spm_path=spm_path, extra_ids=extra_ids)


class BertWordpieceTokenizer:
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        text = native_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
