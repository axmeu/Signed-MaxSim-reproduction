import gc
from pathlib import Path
import torch
from pylate import models
from transformers import AutoTokenizer
from .config import BACKBONES, BACKBONE_LR_DEFAULT, BEIR_TASKS, DEVICE, EMBEDDING_DIM, VARIANTS
from .data import load_beir_task
from .eval import (
    evaluate_beir_pylate,
    evaluate_beir_signed,
    pairwise_accuracy_nevir,
    score_query_docs_pylate,
    score_query_docs_signed,
)
from .io_utils import atomic_torch_save, load_results, log, save_result
from .model import SignedColBERT
from .train import train_signed_model


def run_backbone(backbone_key: str, train_dataset, nevir_dev, nevir_test, device: str = DEVICE) -> None:
    cfg = BACKBONES[backbone_key]
    checkpoint = cfg["checkpoint"]
    trust_remote_code = cfg.get("trust_remote_code", False)
    pylate_kwargs = cfg.get("pylate_kwargs", {})

    log(f"=== Backbone : {backbone_key} ({checkpoint}) ===")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)

    all_expected_keys = []

    baseline_pairwise_key = f"{backbone_key}_baseline_pairwise_acc"
    baseline_beir_keys = [f"{backbone_key}_baseline_{task}_beir" for task in BEIR_TASKS]
    all_expected_keys += [baseline_pairwise_key] + baseline_beir_keys

    if any(k not in load_results() for k in [baseline_pairwise_key] + baseline_beir_keys):
        pylate_model = models.ColBERT(model_name_or_path=checkpoint, **pylate_kwargs)
        pylate_model.to(device)
        score_fn = lambda q, docs: score_query_docs_pylate(pylate_model, q, docs)

        if baseline_pairwise_key not in load_results():
            log("  NevIR pairwise accuracy — baseline")
            save_result(baseline_pairwise_key, pairwise_accuracy_nevir(
                score_fn, nevir_test, checkpoint_path=f"nevir_ckpt_{backbone_key}_baseline.json"
            ))

        for task in BEIR_TASKS:
            key = f"{backbone_key}_baseline_{task}_beir"
            if key not in load_results():
                log(f"  BEIR {task} — baseline")
                documents, queries, qrels = load_beir_task(task)
                save_result(key, evaluate_beir_pylate(
                    pylate_model, documents, queries, qrels,
                    cache_path=f"corpus_emb_{backbone_key}_baseline_{task}.pkl",
                    run_path=f"beir_run_{backbone_key}_baseline_{task}.npz",
                ))

        del pylate_model, score_fn
        gc.collect()
        torch.cuda.empty_cache()

    for variant in VARIANTS:
        vname = variant["name"]
        pairwise_key = f"{backbone_key}_{vname}_pairwise_acc"
        beir_keys = [f"{backbone_key}_{vname}_{task}_beir" for task in BEIR_TASKS]
        all_expected_keys += [pairwise_key] + beir_keys

        if all(k in load_results() for k in [pairwise_key] + beir_keys):
            continue

        signed_model = SignedColBERT(
            checkpoint, embedding_dim=EMBEDDING_DIM,
            use_sign=variant["use_sign"], trust_remote_code=trust_remote_code,
        ).to(device)
        ckpt_path = Path(f"signed_model_{backbone_key}_{vname}.pt")

        if ckpt_path.exists():
            log(f"  [{vname}] Chargement du modèle déjà entraîné : {ckpt_path}")
            signed_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            log(f"  [{vname}] Fine-tuning (use_sign={variant['use_sign']}, backbone_lr={variant['backbone_lr']})...")
            train_signed_model(signed_model, tokenizer, train_dataset,
                                backbone_lr=variant["backbone_lr"], dev_dataset=nevir_dev)
            atomic_torch_save(signed_model.state_dict(), ckpt_path)

        score_fn = lambda q, docs: score_query_docs_signed(signed_model, tokenizer, q, docs, device)

        if pairwise_key not in load_results():
            log(f"  [{vname}] NevIR pairwise accuracy")
            save_result(pairwise_key, pairwise_accuracy_nevir(
                score_fn, nevir_test, checkpoint_path=f"nevir_ckpt_{backbone_key}_{vname}.json"
            ))

        for task in BEIR_TASKS:
            key = f"{backbone_key}_{vname}_{task}_beir"
            if key not in load_results():
                log(f"  [{vname}] BEIR {task}")
                documents, queries, qrels = load_beir_task(task)
                save_result(key, evaluate_beir_signed(
                    signed_model, tokenizer, documents, queries, qrels,
                    run_path=f"beir_run_{backbone_key}_{vname}_{task}.npz",
                ))

        del signed_model, score_fn
        gc.collect()
        torch.cuda.empty_cache()

    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    if all(k in load_results() for k in all_expected_keys):
        patterns = [f"nevir_ckpt_{backbone_key}_baseline.json"]
        for task in BEIR_TASKS:
            patterns.append(f"corpus_emb_{backbone_key}_baseline_{task}.pkl")
            patterns.append(f"beir_run_{backbone_key}_baseline_{task}.npz")
        for variant in VARIANTS:
            vname = variant["name"]
            patterns.append(f"nevir_ckpt_{backbone_key}_{vname}.json")
            for task in BEIR_TASKS:
                patterns.append(f"beir_run_{backbone_key}_{vname}_{task}.npz")
        for pattern in patterns:
            p = Path(pattern)
            if p.exists():
                p.unlink()
        log(f"  {backbone_key} done, tmp files deleted.")
    else:
        missing = [k for k in all_expected_keys if k not in load_results()]
        log(f"  {backbone_key} : incomplete ({len(missing)} missing keys) - restart to continue")
