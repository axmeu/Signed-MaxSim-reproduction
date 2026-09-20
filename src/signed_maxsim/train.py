from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    DEVICE,
    EVAL_EVERY,
    HEAD_LR,
    MAX_STEPS,
    MIN_STEPS_BEFORE_STOP,
    PATIENCE_EVALS,
    WEIGHT_DECAY,
)
from .eval import pairwise_accuracy_nevir, score_query_docs_signed
from .io_utils import log
from .model import infonce_loss, score_all_pairs

import torch


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        q_enc = tokenizer([ex["query"] for ex in batch], padding=True, truncation=True, max_length=32, return_tensors="pt")
        d_enc = tokenizer([ex["positive"] for ex in batch], padding=True, truncation=True, max_length=180, return_tensors="pt")
        return q_enc, d_enc
    return collate_fn


def train_signed_model(signed_model, tokenizer, train_dataset, backbone_lr,
                        dev_dataset=None, history=None, device: str = DEVICE):
    frozen = backbone_lr == 0.0
    for p in signed_model.backbone.parameters():
        p.requires_grad = not frozen

    param_groups = [
        {"params": signed_model.magnitude_head.parameters(), "lr": HEAD_LR},
        {"params": signed_model.sign_head.parameters(), "lr": HEAD_LR},
    ]
    if not frozen:
        param_groups.append({"params": signed_model.backbone.parameters(), "lr": backbone_lr})

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=make_collate_fn(tokenizer))
    optimizer = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_dev_acc = -1.0
    evals_without_improvement = 0
    best_state = None
    step = 0
    last_loss = None

    signed_model.train()
    while step < MAX_STEPS:
        for q_enc, d_enc in train_loader:
            if step >= MAX_STEPS:
                break

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                scores = score_all_pairs(signed_model, q_enc, d_enc, device)
                loss = infonce_loss(scores)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            last_loss = loss.item()

            if step % 10 == 0:
                log(f"      step {step}/{MAX_STEPS} loss {last_loss:.4f}")

            if dev_dataset is not None and step % EVAL_EVERY == 0:
                signed_model.eval()
                score_fn = lambda q, docs: score_query_docs_signed(signed_model, tokenizer, q, docs, device)
                dev_acc = pairwise_accuracy_nevir(score_fn, dev_dataset)
                signed_model.train()
                log(f"      step {step} — dev pairwise acc : {dev_acc:.3f}")
                if history is not None:
                    history.append((step, last_loss, dev_acc))

                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    best_state = {k: v.clone() for k, v in signed_model.state_dict().items()}
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1
                    if evals_without_improvement >= PATIENCE_EVALS and step >= MIN_STEPS_BEFORE_STOP:
                        log(f"      Early stop at step {step} (best dev acc = {best_dev_acc:.3f})")
                        if best_state is not None:
                            signed_model.load_state_dict(best_state)
                        return signed_model

    if best_state is not None:
        signed_model.load_state_dict(best_state)
    return signed_model
