import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .config import RESULTS_PATH


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def atomic_write_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def atomic_savez_compressed(path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".", suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def atomic_torch_save(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_results(path: str = RESULTS_PATH) -> dict:
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_result(key: str, value: float, path: str = RESULTS_PATH) -> None:
    results = load_results(path)
    results[key] = value
    atomic_write_json(results, path)
    log(f"Saved: {key} = {value:.3f}")


def save_run_dict(run_dict: dict, path) -> None:
    query_ids = list(run_dict.keys())
    doc_ids = list(next(iter(run_dict.values())).keys())
    matrix = np.array([[run_dict[q][d] for d in doc_ids] for q in query_ids], dtype=np.float32)
    atomic_savez_compressed(path, matrix=matrix, query_ids=query_ids, doc_ids=doc_ids)


def load_run_dict(path) -> dict:
    data = np.load(path, allow_pickle=True)
    query_ids, doc_ids, matrix = data["query_ids"], data["doc_ids"], data["matrix"]
    return {q: {d: float(matrix[i, j]) for j, d in enumerate(doc_ids)} for i, q in enumerate(query_ids)}
