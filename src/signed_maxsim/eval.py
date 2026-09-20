from pathlib import Path

import torch
from ranx import Qrels, Run, evaluate

from .config import DEVICE
from .io_utils import atomic_write_json, load_run_dict, log, save_run_dict
from .model import signed_maxsim
import json


def pairwise_accuracy_nevir(score_fn, dataset, checkpoint_path=None, checkpoint_every: int = 20) -> float:
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    state = {"processed": 0, "correct": 0}
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            state = json.load(f)
        log(f"    Pairwise accuracy: {state['processed']}/{len(dataset)} already done")

    n = len(dataset)
    for i in range(state["processed"], n):
        ex = dataset[i]
        docs = [ex["doc1"], ex["doc2"]]
        scores_q1 = score_fn(ex["q1"], docs)
        scores_q2 = score_fn(ex["q2"], docs)
        if scores_q1[0] > scores_q1[1] and scores_q2[1] > scores_q2[0]:
            state["correct"] += 1
        state["processed"] += 1
        if checkpoint_path and state["processed"] % checkpoint_every == 0:
            atomic_write_json(state, checkpoint_path)

    if checkpoint_path:
        atomic_write_json(state, checkpoint_path)
    return state["correct"] / state["processed"]


@torch.no_grad()
def score_query_docs_pylate(pylate_model, query, docs):
    q_emb = pylate_model.encode([query], is_query=True)
    scores = []
    for doc in docs:
        d_emb = pylate_model.encode([doc], is_query=False)
        scores.append(float(pylate_model.similarity(q_emb, d_emb).squeeze()))
    return scores


@torch.no_grad()
def score_query_docs_signed(signed_model, tokenizer, query, docs, device: str = DEVICE):
    signed_model.eval()
    q_enc = tokenizer([query], padding=True, truncation=True, max_length=32, return_tensors="pt")
    d_enc = tokenizer(docs, padding=True, truncation=True, max_length=180, return_tensors="pt")
    q_mag, q_sign, q_mask = signed_model.encode(q_enc["input_ids"].to(device), q_enc["attention_mask"].to(device))
    d_mag, d_sign, d_mask = signed_model.encode(d_enc["input_ids"].to(device), d_enc["attention_mask"].to(device))
    scores = []
    for j in range(len(docs)):
        s = signed_maxsim(q_mag, q_sign, q_mask, d_mag[j:j + 1], d_sign[j:j + 1], d_mask[j:j + 1])
        scores.append(s.item())
    return scores


@torch.no_grad()
def encode_corpus_pylate_exact(pylate_model, texts, batch_size: int = 32, cache_path=None):
    import os
    import pickle

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            log(f"    Loading corpus cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    embs = pylate_model.encode(texts, batch_size=batch_size, is_query=False, show_progress_bar=True)
    if cache_path is not None:
        tmp = Path(str(cache_path) + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(embs, f)
        os.replace(tmp, cache_path)
    return embs


@torch.no_grad()
def score_query_vs_corpus_pylate(pylate_model, query, corpus_embs):
    q_emb = pylate_model.encode([query], is_query=True)
    scores = []
    for doc_emb in corpus_embs:
        s = pylate_model.similarity(q_emb, [doc_emb])
        scores.append(float(s[0][0]))
    return scores


@torch.no_grad()
def encode_corpus_signed(signed_model, tokenizer, texts, batch_size: int = 32, device: str = DEVICE):
    signed_model.eval()
    all_mag, all_sign, all_mask = [], [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        mag, sign, mask = signed_model.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        all_mag.append(mag)
        all_sign.append(sign)
        all_mask.append(mask)
    return all_mag, all_sign, all_mask


@torch.no_grad()
def score_query_vs_corpus_signed(signed_model, tokenizer, query, corpus_mag, corpus_sign, corpus_mask, device: str = DEVICE):
    signed_model.eval()
    q_enc = tokenizer([query], truncation=True, max_length=32, return_tensors="pt")
    q_mag, q_sign, q_mask = signed_model.encode(q_enc["input_ids"].to(device), q_enc["attention_mask"].to(device))
    scores = []
    for mag_b, sign_b, mask_b in zip(corpus_mag, corpus_sign, corpus_mask):
        Bc = mag_b.shape[0]
        s = signed_maxsim(
            q_mag.expand(Bc, -1, -1), q_sign.expand(Bc, -1, -1), q_mask.expand(Bc, -1),
            mag_b, sign_b, mask_b,
        )
        scores.extend(s.cpu().tolist())
    return scores


def evaluate_beir_pylate(pylate_model, documents, queries, qrels, cache_path=None, run_path=None) -> float:
    doc_ids = [str(d["id"]) for d in documents]
    doc_texts = [d["text"] for d in documents]
    corpus_embs = encode_corpus_pylate_exact(pylate_model, doc_texts, cache_path=cache_path)

    run_dict = {}
    run_path = Path(run_path) if run_path else None
    if run_path and run_path.exists():
        run_dict = load_run_dict(run_path)
        log(f"    BEIR: {len(run_dict)} requests already scored")

    for qid, qtext in queries.items():
        qid = str(qid)
        if qid in run_dict:
            continue
        scores = score_query_vs_corpus_pylate(pylate_model, qtext, corpus_embs)
        run_dict[qid] = {doc_ids[j]: scores[j] for j in range(len(doc_ids))}
        if run_path and len(run_dict) % 20 == 0:
            save_run_dict(run_dict, run_path)

    if run_path:
        save_run_dict(run_dict, run_path)

    qrels_dict = {str(qid): {str(did): rel for did, rel in docs.items()} for qid, docs in qrels.items()}
    return evaluate(Qrels(qrels_dict), Run(run_dict), "ndcg@10", make_comparable=True)


def evaluate_beir_signed(signed_model, tokenizer, documents, queries, qrels, run_path=None, device: str = DEVICE) -> float:
    doc_ids = [str(d["id"]) for d in documents]
    doc_texts = [d["text"] for d in documents]
    corpus_mag, corpus_sign, corpus_mask = encode_corpus_signed(signed_model, tokenizer, doc_texts, batch_size=32, device=device)

    run_dict = {}
    run_path = Path(run_path) if run_path else None
    if run_path and run_path.exists():
        run_dict = load_run_dict(run_path)
        log(f"    BEIR: {len(run_dict)} requests already scored")

    for qid, qtext in queries.items():
        qid = str(qid)
        if qid in run_dict:
            continue
        scores = score_query_vs_corpus_signed(signed_model, tokenizer, qtext, corpus_mag, corpus_sign, corpus_mask, device=device)
        run_dict[qid] = {doc_ids[j]: scores[j] for j in range(len(doc_ids))}
        if run_path and len(run_dict) % 20 == 0:
            save_run_dict(run_dict, run_path)

    if run_path:
        save_run_dict(run_dict, run_path)

    qrels_dict = {str(qid): {str(did): rel for did, rel in docs.items()} for qid, docs in qrels.items()}
    return evaluate(Qrels(qrels_dict), Run(run_dict), "ndcg@10", make_comparable=True)
