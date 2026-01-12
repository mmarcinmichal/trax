# coding: utf-8
"""
Preprocessing w stylu TextGCN (remove_words.py) dla 20NG.

- clean_str jak w CNN_sentence / TextGCN.
- stopwords z NLTK.
- filtr globalny: min_freq=5 na całym korpusie (train+test).
"""

import re

from collections import Counter
from typing import Any, Iterable, List

import nltk

from absl import logging

logging.set_verbosity(logging.INFO)

try:
    from nltk.corpus import stopwords
    nltk.download('stopwords')

    EN_STOPWORDS = set(stopwords.words("english"))
except ValueError:
    logging.info("Install nltk for stopwords")


_CLEAN_PATTERNS = [
    # 1) dokładnie jak w CNN_sentence / TextGCN:
    #    zostawiamy A–Z, a–z, cyfry, nawiasy, , ! ? ' `
    (r"[^A-Za-z0-9(),!?\'\`]", " "),
    (r"\'s", " 's"),
    (r"\'ve", " 've"),
    (r"n\'t", " n't"),
    (r"\'re", " 're"),
    (r"\'d", " 'd"),
    (r"\'ll", " 'll"),
    (r",", " , "),
    (r"!", " ! "),
    (r"\(", " ( "),
    (r"\)", " ) "),
    (r"\?", " ? "),
]


def clean_str_textgcn(s: str) -> str:
    """
    Oryginalne clean_str jak w TextGCN / CNN_sentence.
    """
    s = s.strip().strip('"')
    for pattern, repl in _CLEAN_PATTERNS:
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip().lower()


def _tokenize_textgcn(s: str) -> List[str]:
    """clean_str + split na spacje."""
    return clean_str_textgcn(s).split()


def preprocess_newsgroups_textgcn(
    train_texts: Iterable[str],
    test_texts: Iterable[str],
    min_freq: int = 5,
) -> tuple[list[str], list[str], Counter[Any]]:
    """
    Dokładny odpowiednik remove_words.py dla 20NG:

    - liczymy częstości słów na całym korpusie (train+test),
    - wyrzucamy stopwords i słowa o freq < min_freq,
    - zwracamy oczyszczone teksty jako "word1 word2 ...".
    """
    # Zamiana na listy, żeby można było indeksować
    train_texts = list(train_texts)
    test_texts = list(test_texts)

    all_raw = train_texts + test_texts

    # 1. clean_str + tokenizacja
    all_tokens: List[List[str]] = []
    for txt in all_raw:
        toks = _tokenize_textgcn(txt)
        all_tokens.append(toks)

    # 2. globalne częstości (po usunięciu stopwords)
    freq = Counter()
    for toks in all_tokens:
        for tok in toks:
            if tok and tok not in EN_STOPWORDS:
                freq[tok] += 1

    # 3. filtracja dokumentów
    def _filter_doc(toks: List[str]) -> List[str]:
        return [
            tok
            for tok in toks
            if tok not in EN_STOPWORDS and freq[tok] >= min_freq
        ]

    cleaned_docs: List[str] = []
    for toks in all_tokens:
        kept = _filter_doc(toks)
        cleaned_docs.append(" ".join(kept))

    n_tr = len(train_texts)

    return cleaned_docs[:n_tr], cleaned_docs[n_tr:], freq
