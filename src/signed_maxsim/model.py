import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .config import DROPOUT, EMBEDDING_DIM


class SignedColBERT(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        embedding_dim: int = EMBEDDING_DIM,
        use_sign: bool = True,
        dropout: float = DROPOUT,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name, trust_remote_code=trust_remote_code)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.magnitude_head = nn.Linear(hidden_size, embedding_dim)
        self.sign_head = nn.Linear(hidden_size, embedding_dim)
        self.use_sign = use_sign

    def encode(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = self.dropout(out.last_hidden_state)

        magnitude = F.normalize(self.magnitude_head(token_embeddings), p=2, dim=-1)
        if self.use_sign:
            sign = torch.tanh(self.sign_head(token_embeddings))
        else:
            sign = torch.ones_like(magnitude)
        return magnitude, sign, attention_mask

    def forward(self, query_ids, query_mask, doc_ids, doc_mask):
        q_mag, q_sign, q_mask = self.encode(query_ids, query_mask)
        d_mag, d_sign, d_mask = self.encode(doc_ids, doc_mask)
        return signed_maxsim(q_mag, q_sign, q_mask, d_mag, d_sign, d_mask)


def signed_maxsim(q_mag, q_sign, q_mask, d_mag, d_sign, d_mask):
    magnitude_sim = torch.einsum("bqd,bkd->bqk", q_mag, d_mag)  # (B, Lq, Ld)

    doc_pad_mask = (d_mask == 0).unsqueeze(1)
    magnitude_sim_masked = magnitude_sim.masked_fill(doc_pad_mask, float("-inf"))

    max_magnitude, best_doc_idx = magnitude_sim_masked.max(dim=-1)  # (B, Lq) chacun

    sign_sim = torch.einsum("bqd,bkd->bqk", q_sign, d_sign) / q_sign.shape[-1]  # (B, Lq, Ld)
    sign_at_best_match = torch.gather(sign_sim, dim=-1, index=best_doc_idx.unsqueeze(-1)).squeeze(-1)  # (B, Lq)

    per_token_score = max_magnitude * sign_at_best_match
    per_token_score = per_token_score.masked_fill(q_mask == 0, 0.0)
    return per_token_score.sum(dim=-1)


def score_all_pairs(model: SignedColBERT, q_enc, d_enc, device: str):
    q_mag, q_sign, q_mask = model.encode(q_enc["input_ids"].to(device), q_enc["attention_mask"].to(device))
    d_mag, d_sign, d_mask = model.encode(d_enc["input_ids"].to(device), d_enc["attention_mask"].to(device))

    magnitude_sim = torch.einsum("aid,bjd->abij", q_mag, d_mag)  # (B, B, Lq, Ld)
    doc_pad_mask = (d_mask == 0)[None, :, None, :]
    magnitude_sim_masked = magnitude_sim.masked_fill(doc_pad_mask, float("-inf"))

    max_magnitude, best_doc_idx = magnitude_sim_masked.max(dim=-1)  # (B, B, Lq) chacun

    sign_sim = torch.einsum("aid,bjd->abij", q_sign, d_sign) / q_sign.shape[-1]  # (B, B, Lq, Ld)
    sign_at_best_match = torch.gather(sign_sim, dim=-1, index=best_doc_idx.unsqueeze(-1)).squeeze(-1)  # (B, B, Lq)

    per_token_score = max_magnitude * sign_at_best_match
    query_pad_mask = (q_mask == 0)[:, None, :]
    per_token_score = per_token_score.masked_fill(query_pad_mask, 0.0)

    return per_token_score.sum(dim=-1)  # (B, B), scores[a, b] = query a vs doc b


def infonce_loss(scores_matrix: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    labels = torch.arange(scores_matrix.shape[0], device=scores_matrix.device)
    return F.cross_entropy(scores_matrix / temperature, labels)
