import numpy as np

from absl import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import trax

from trax.data.loader.raw.base import RawDataset, Splits

logging.set_verbosity(logging.INFO)

try:
    from trax.data.builder.graph.textgcn import (
        newsgroups as cfg,
    )

    MAX_TRAIN_DOCS = getattr(cfg, "MAX_TRAIN_DOCS")
    MAX_TEST_DOCS = getattr(cfg, "MAX_TEST_DOCS")
    SHUFFLE_SEED = getattr(cfg, "SHUFFLE_SEED")

    MAX_VOCAB_SIZE = getattr(cfg, "MAX_VOCAB_SIZE")
    MIN_DF = getattr(cfg, "MIN_DF")
    MAX_DF = getattr(cfg, "MAX_DF")
    STOP_WORDS = getattr(cfg, "STOP_WORDS")
    VAL_RATIO = getattr(cfg, "VAL_RATIO")
except ValueError:
    # Fallback defaults if import fails
    MAX_TRAIN_DOCS = None
    MAX_TEST_DOCS = None
    SHUFFLE_SEED = 0

    MAX_VOCAB_SIZE = 20000
    MIN_DF = 5
    MAX_DF = 0.5
    STOP_WORDS = "english"
    VAL_RATIO = 0.1


def main():
    train_texts, train_labels = trax.data.loader.raw.base.load_dataset(
        RawDataset.NG.value, Splits.TRAIN.value
    )
    test_texts, test_labels = trax.data.loader.raw.base.load_dataset(
        RawDataset.NG.value, Splits.TEST.value
    )

    if MAX_TRAIN_DOCS is not None:
        train_texts = train_texts[:MAX_TRAIN_DOCS]
        train_labels = train_labels[:MAX_TRAIN_DOCS]

    if MAX_TEST_DOCS is not None:
        test_texts = test_texts[:MAX_TEST_DOCS]
        test_labels = test_labels[:MAX_TEST_DOCS]

    rng = np.random.RandomState(SHUFFLE_SEED)
    idx = np.arange(len(train_texts))
    rng.shuffle(idx)

    n_val = int(round(len(idx) * VAL_RATIO))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    tr_texts = [train_texts[i] for i in tr_idx]
    tr_y = train_labels[tr_idx]

    val_texts = [train_texts[i] for i in val_idx]
    val_y = train_labels[val_idx]

    logging.info(
        f"train/val/test sizes: {len(tr_texts)} / {len(val_texts)} / {len(test_texts)}"
    )
    logging.info(
        f"labels min/max/classes: {train_labels.min()} / {train_labels.max()} / {len(np.unique(train_labels))}"
    )

    vec = TfidfVectorizer(
        lowercase=True,
        max_features=MAX_VOCAB_SIZE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        stop_words=STOP_WORDS,
        norm="l2",
        smooth_idf=True,
        sublinear_tf=True,
    )

    X_tr = vec.fit_transform(tr_texts)
    X_val = vec.transform(val_texts)
    X_te = vec.transform(test_texts)

    nnz_tr = np.asarray(X_tr.getnnz(axis=1)).reshape(-1)
    logging.info(f"TF-IDF vocab size: {len(vec.vocabulary_)}")
    logging.info(f"train docs with 0 terms: {(nnz_tr == 0).sum()} / {len(nnz_tr)}")

    clf = LogisticRegression(
        max_iter=3_000,
        solver="lbfgs",
        n_jobs=-1,
        C=4.0,  # mild regularization; try 1.0/2.0/4.0/8.0 if you want
        random_state=SHUFFLE_SEED,
    )

    clf.fit(X_tr, tr_y)

    val_pred = clf.predict(X_val)
    te_pred = clf.predict(X_te)

    val_acc = accuracy_score(val_y, val_pred)
    te_acc = accuracy_score(test_labels, te_pred)

    logging.info(f"TFIDF+LR – val accuracy:  {val_acc:.4f}")
    logging.info(f"TFIDF+LR – test accuracy: {te_acc:.4f}")


if __name__ == "__main__":
    main()
