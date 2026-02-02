"""Build a global TextGCN graph for 20 Newsgroups."""

from pathlib import Path

from trax.utils import logging as trax_logging

from resources.examples.python.base import find_project_root
from trax.data.builder.graph.textgcn.base import TextGraphConfig, build_textgcn_graph
from trax.data.loader.raw.base import RawDataset

trax_logging.set_verbosity(trax_logging.INFO)

# --- CONFIG ---
MAX_TRAIN_DOCS = None
MAX_TEST_DOCS = None
SHUFFLE_SEED = 0

MAX_VOCAB_SIZE = None
MIN_DF = 1
MAX_DF = 1.0
STOP_WORDS = None

PMI_WINDOW_SIZE = 20
VAL_RATIO = 0.01

PROJECT_ROOT = find_project_root(Path(__file__).resolve())
OUTPUT_FILE = PROJECT_ROOT / "resources" / "data" / "serialized" / "graphs" / "20_newsgroups_bydate.npz"


def main():
    trax_logging.info(f"Build sparse matrix and save it into: {OUTPUT_FILE}")

    config = TextGraphConfig(
        dataset_name=RawDataset.NG.value,  # "SetFit/20_newsgroups",
        is_local=True,
        dataset_config=None,
        train_split="train",
        test_split="test",
        text_field="text",
        label_field="label",
        max_train_docs=MAX_TRAIN_DOCS,
        max_test_docs=MAX_TEST_DOCS,
        shuffle_seed=SHUFFLE_SEED,
        max_vocab_size=MAX_VOCAB_SIZE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        stop_words=STOP_WORDS,
        pmi_window_size=PMI_WINDOW_SIZE,
        val_ratio=VAL_RATIO,
    )

    build_textgcn_graph(config, OUTPUT_FILE)


if __name__ == "__main__":
    main()