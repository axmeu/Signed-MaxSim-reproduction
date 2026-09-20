from datasets import Dataset, concatenate_datasets, load_dataset
from pylate import evaluation as pylate_evaluation

from .config import SEED


def nevir_row_to_triplets(example: dict) -> list[dict]:
    return [
        {"query": example["q1"], "positive": example["doc1"], "negative": example["doc2"]},
        {"query": example["q2"], "positive": example["doc2"], "negative": example["doc1"]},
    ]


def load_nevir_splits(dev_fraction: float = 0.1):
    nevir_train_full = load_dataset("orionweller/NevIR", split="train")
    split = nevir_train_full.train_test_split(test_size=dev_fraction, seed=SEED)
    nevir_train, nevir_dev = split["train"], split["test"]

    triplets = []
    for row in nevir_train:
        triplets.extend(nevir_row_to_triplets(row))
    nevir_triplets_dataset = Dataset.from_list(triplets)

    nevir_test = load_dataset("orionweller/NevIR", split="test")
    return nevir_triplets_dataset, nevir_dev, nevir_test


def load_general_train_pool(dataset_name: str = "sentence-transformers/msmarco-bm25"):
    general_train_full = load_dataset(dataset_name, "triplet", split="train")
    return general_train_full.shuffle(seed=SEED)


def build_train_dataset(nevir_triplets_dataset, general_train_full,
                        n_general_samples: int, nevir_oversample: int = 1):
    parts = [nevir_triplets_dataset] * max(nevir_oversample, 1)
    if n_general_samples > 0:
        n = min(n_general_samples, len(general_train_full))
        parts.append(general_train_full.select(range(n)))
    return concatenate_datasets(parts).shuffle(seed=SEED)


def load_beir_task(task: str):
    documents, queries, qrels = pylate_evaluation.load_beir(task, split="test")
    return documents, queries, qrels
