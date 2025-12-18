from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =============================
# Config
# =============================

@dataclass
class BCConfig:
    # Max ICIs stored in CSV per row (ICI1..ICIK)
    max_ici_cols: int = 28

    # Special values in the CSV representation
    eos_value: float = 0.0          # end-of-coda marker in CSV padding
    soc_value: float = -1.0         # start-of-coda marker we add (never a real ICI)

    # Token types
    # (we don't rely on numeric ICI value alone to identify special tokens)
    TOK_PAD: int = 0
    TOK_ICI: int = 1
    TOK_EOS: int = 2
    TOK_SOC: int = 3

    # Normalization / transforms
    use_log_ici: bool = True        # model in log-space for stability
    clamp_min_ici: float = 1e-4     # when taking log / exp safety

    # Model hyperparams
    whale_emb_dim: int = 32
    type_emb_dim: int = 16
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.0

    # Loss weights
    w_ici: float = 1.0
    w_eos: float = 1.0

    # Training
    grad_clip_norm: float = 1.0

    # If whale IDs in CSV need remapping, handle externally; we assume non-negative ints in column.
    # (Safer than offset hacks.)
    pass


# =============================
# Utilities: load + parse rows
# =============================

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Always sort within conversation by timestamp to ensure correct history order.
    # This also ensures rows of the same REC are contiguous, which simplifies logic elsewhere.
    df = df.sort_values(["REC", "TsTo"], ascending=True).reset_index(drop=True)

    return df

def build_conversations(df: pd.DataFrame) -> Dict[str, List[int]]:
    conv: Dict[str, List[int]] = {}
    # df is assumed sorted by ["REC","TsTo"]
    for idx, (rec, _) in enumerate(zip(df["REC"].tolist(), df["TsTo"].tolist())):
        conv.setdefault(rec, []).append(idx)
    return conv


def split_recs(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.0,
    seed: int = 0,
) -> Tuple[List[str], List[str], List[str]]:
    rng = np.random.default_rng(seed)
    recs = df["REC"].unique().tolist()
    rng.shuffle(recs)

    n = len(recs)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test_recs = recs[:n_test]
    val_recs = recs[n_test:n_test + n_val]
    train_recs = recs[n_test + n_val:]

    return train_recs, val_recs, test_recs


def extract_coda_ici(df: pd.DataFrame, row_idx: int, cfg: BCConfig) -> List[float]:
    """
    Extract the ICI sequence for a row, stopping at EOS padding (=0).
    EOS itself is NOT included in returned list.
    """
    seq: List[float] = []
    for k in range(1, cfg.max_ici_cols + 1):
        v = float(df.at[row_idx, f"ICI{k}"])
        if v == cfg.eos_value:
            break
        seq.append(v)
    return seq


def compute_log_stats(df: pd.DataFrame, cfg: BCConfig) -> Tuple[float, float]:
    """
    Compute mean/std of log(ICIs) over all non-zero ICIs in df.
    Used to normalize ICI tokens. Special tokens use 0 feature.
    """
    vals: List[float] = []
    for k in range(1, cfg.max_ici_cols + 1):
        col = df[f"ICI{k}"].astype(float).values
        col = col[col > cfg.eos_value]  # strictly > 0
        if col.size:
            vals.append(col)
    if not vals:
        # fallback
        return 0.0, 1.0

    all_ici = np.concatenate(vals, axis=0)
    all_ici = np.clip(all_ici, cfg.clamp_min_ici, None)

    if cfg.use_log_ici:
        feats = np.log(all_ici)
    else:
        feats = all_ici

    mean = float(feats.mean())
    std = float(feats.std()) if float(feats.std()) > 1e-8 else 1.0
    return mean, std


def ici_to_feat(ici: float, cfg: BCConfig, mean: float, std: float) -> float:
    """
    Convert a raw ICI (>0) to a normalized feature.
    For special tokens (SOC/EOS/PAD), caller should pass 0.0 and rely on type embedding.
    """
    x = max(float(ici), cfg.clamp_min_ici)
    if cfg.use_log_ici:
        x = math.log(x)
    return float((x - mean) / std)


def feat_to_ici(feat: float, cfg: BCConfig, mean: float, std: float) -> float:
    """
    Inverse of ici_to_feat (for regression head output).
    """
    x = float(feat) * std + mean
    if cfg.use_log_ici:
        x = math.exp(x)
    return float(max(x, 0.0))


# =============================
# Dataset construction
# =============================

class WhaleBCDataset(Dataset):
    """
    Each dataset item corresponds to one row (one coda episode).
    Input tokens = [history tokens] + [SOC] + [ICI... ] + [EOS]
    Model predicts, at each position t, properties of token t+1:
      - next_eos (binary): whether token t+1 is EOS
      - next_ici_feat (regression): only when token t+1 is ICI
    Loss is applied ONLY on the suffix (current coda), i.e. positions whose "next token"
    is within [SOC..EOS] of the current episode. History is context only.
    """
    def __init__(
        self,
        csv_path: str,
        cfg: Optional[BCConfig] = None,
        ici_mean: Optional[float] = None,
        ici_std: Optional[float] = None,
        recs: Optional[List[str]] = None,
    ):
        self.cfg = cfg or BCConfig()
        self.df = load_csv(csv_path)
        if recs is not None:
            self.df = self.df[self.df["REC"].isin(recs)].reset_index(drop=True)

        # Remap whale IDs to a dense range 0..n-1 for embedding efficiency.
        raw_whales = self.df["Whale"].astype(int).values
        uniq = np.unique(raw_whales)
        self.whale_id_map = {int(w): i for i, w in enumerate(uniq)}
        self.inv_whale_id_map = {i: int(w) for w, i in self.whale_id_map.items()}
        self.df["Whale"] = self.df["Whale"].astype(int).map(self.whale_id_map).astype(int)

        self.n_whales = len(uniq)

        self.conversations = build_conversations(self.df)

        # Normalization stats: compute if not provided (TRAIN ONLY).
        if ici_mean is None or ici_std is None:
            self.ici_mean, self.ici_std = compute_log_stats(self.df, self.cfg)
        else:
            self.ici_mean, self.ici_std = float(ici_mean), float(ici_std)

        # Episodes: (REC, pos_in_convo, row_idx)
        self.episodes: List[Tuple[str, int, int]] = []
        for rec, rows in self.conversations.items():
            for pos, row_idx in enumerate(rows):
                self.episodes.append((rec, pos, row_idx))

    def __len__(self) -> int:
        return len(self.episodes)

    def _build_history_tokens(self, conv_rows: List[int], upto_pos: int) -> Tuple[List[int], List[float], List[int]]:
        """
        Build tokens for all prior codas:
          [ICI..., EOS] for each prior coda, with whale_id repeated for those tokens.
        Returns:
          whale_ids: List[int]
          ici_feats: List[float] (0.0 for EOS)
          tok_types: List[int]
        """
        cfg = self.cfg
        whale_ids: List[int] = []
        ici_feats: List[float] = []
        tok_types: List[int] = []

        for p in range(upto_pos):
            row_idx = conv_rows[p]
            w = int(self.df.at[row_idx, "Whale"])
            seq = extract_coda_ici(self.df, row_idx, cfg)

            for ici in seq:
                whale_ids.append(w)
                ici_feats.append(ici_to_feat(ici, cfg, self.ici_mean, self.ici_std))
                tok_types.append(cfg.TOK_ICI)

            # EOS marker between codas (type distinguishes EOS vs PAD)
            whale_ids.append(w)
            ici_feats.append(0.0)
            tok_types.append(cfg.TOK_EOS)

        return whale_ids, ici_feats, tok_types

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cfg = self.cfg
        rec, pos, row_idx = self.episodes[idx]
        conv_rows = self.conversations[rec]

        # History tokens
        h_w, h_f, h_t = self._build_history_tokens(conv_rows, upto_pos=pos)
        hist_len = len(h_w)

        # Current episode (current coda)
        cur_w = int(self.df.at[row_idx, "Whale"])
        cur_seq = extract_coda_ici(self.df, row_idx, cfg)  # excludes EOS

        # Build suffix tokens: SOC, ICI..., EOS
        s_w: List[int] = [cur_w]
        s_f: List[float] = [0.0]             # SOC uses 0 feature; type embedding carries meaning
        s_t: List[int] = [cfg.TOK_SOC]

        for ici in cur_seq:
            s_w.append(cur_w)
            s_f.append(ici_to_feat(ici, cfg, self.ici_mean, self.ici_std))
            s_t.append(cfg.TOK_ICI)

        # Append EOS token at end (important for correct alignment and EOS learning)
        s_w.append(cur_w)
        s_f.append(0.0)
        s_t.append(cfg.TOK_EOS)

        # Full input sequence
        whale_ids = h_w + s_w
        ici_feats = h_f + s_f
        tok_types = h_t + s_t
        T = len(whale_ids)

        # Targets predict the *next token* (shifted left)
        # For position t, next token is at t+1
        next_is_eos = [0] * T
        next_ici_feat = [0.0] * T
        loss_mask_eos = [0] * T
        loss_mask_ici = [0] * T

        # We only compute loss for transitions whose next-token lies in the suffix
        # Define suffix index range in the full sequence
        suffix_start = hist_len  # index of SOC
        suffix_end = T - 1       # index of final EOS (last token)
        # transitions that predict token in [suffix_start .. suffix_end] come from positions
        # [suffix_start-1 .. suffix_end-1]. But we only want to train on suffix internal behavior,
        # so we start at suffix_start (SOC's own prediction onward). That matches your intention.
        # (History->SOC prediction is not learned; SOC is a given control token.)
        train_from = suffix_start  # position of SOC
        train_to = suffix_end - 1  # last position that has a valid next token inside suffix

        # Fill targets & masks
        for t in range(T - 1):
            nxt_type = tok_types[t + 1]
            if t < train_from or t > train_to:
                continue

            # EOS head: always supervised on suffix transitions
            loss_mask_eos[t] = 1
            next_is_eos[t] = 1 if nxt_type == cfg.TOK_EOS else 0

            # ICI head: only supervised when next token is an ICI
            if nxt_type == cfg.TOK_ICI:
                loss_mask_ici[t] = 1
                next_ici_feat[t] = ici_feats[t + 1]

        attn_mask = [1] * T  # all real tokens; padding happens in collate

        return {
            "whale_ids": whale_ids,
            "ici_feats": ici_feats,
            "tok_types": tok_types,
            "attn_mask": attn_mask,
            "next_is_eos": next_is_eos,
            "next_ici_feat": next_ici_feat,
            "loss_mask_eos": loss_mask_eos,
            "loss_mask_ici": loss_mask_ici,
        }


def collate_bc(batch: List[Dict[str, Any]], cfg: Optional[BCConfig] = None):
    """
    Pads variable-length sequences. Returns tensors:
      whale_ids      (B,T) long
      ici_feats      (B,T) float
      tok_types      (B,T) long
      attn_mask      (B,T) bool
      next_is_eos    (B,T) float (0/1)
      next_ici_feat  (B,T) float
      loss_mask_eos  (B,T) bool
      loss_mask_ici  (B,T) bool
    """
    cfg = cfg or BCConfig()
    B = len(batch)
    T = max(len(ex["whale_ids"]) for ex in batch)

    whale_ids = torch.zeros((B, T), dtype=torch.long)
    ici_feats = torch.zeros((B, T), dtype=torch.float32)
    tok_types = torch.full((B, T), cfg.TOK_PAD, dtype=torch.long)
    attn_mask = torch.zeros((B, T), dtype=torch.bool)

    next_is_eos = torch.zeros((B, T), dtype=torch.float32)
    next_ici_feat = torch.zeros((B, T), dtype=torch.float32)
    loss_mask_eos = torch.zeros((B, T), dtype=torch.bool)
    loss_mask_ici = torch.zeros((B, T), dtype=torch.bool)

    for i, ex in enumerate(batch):
        t = len(ex["whale_ids"])
        whale_ids[i, :t] = torch.tensor(ex["whale_ids"], dtype=torch.long)
        ici_feats[i, :t] = torch.tensor(ex["ici_feats"], dtype=torch.float32)
        tok_types[i, :t] = torch.tensor(ex["tok_types"], dtype=torch.long)
        attn_mask[i, :t] = True

        next_is_eos[i, :t] = torch.tensor(ex["next_is_eos"], dtype=torch.float32)
        next_ici_feat[i, :t] = torch.tensor(ex["next_ici_feat"], dtype=torch.float32)
        loss_mask_eos[i, :t] = torch.tensor(ex["loss_mask_eos"], dtype=torch.bool)
        loss_mask_ici[i, :t] = torch.tensor(ex["loss_mask_ici"], dtype=torch.bool)

    return (
        whale_ids, ici_feats, tok_types, attn_mask,
        next_is_eos, next_ici_feat, loss_mask_eos, loss_mask_ici
    )


# =============================
# Model
# =============================

class GRUBCModel(nn.Module):
    """
    GRU sequence model with two heads:
      - EOS head: predicts whether next token is EOS
      - ICI head: predicts next ICI feature (normalized log-space), only meaningful if next isn't EOS
    """

    def __init__(self, n_whales: int, cfg: Optional[BCConfig] = None):
        super().__init__()
        self.cfg = cfg or BCConfig()

        self.whale_emb = nn.Embedding(n_whales, self.cfg.whale_emb_dim)
        self.type_emb = nn.Embedding(4, self.cfg.type_emb_dim)  # PAD/ICI/EOS/SOC

        in_dim = self.cfg.whale_emb_dim + self.cfg.type_emb_dim + 1  # + ici_feat scalar
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            batch_first=True,
            dropout=self.cfg.dropout if self.cfg.num_layers > 1 else 0.0,
        )

        self.head_eos = nn.Linear(self.cfg.hidden_size, 1)  # logits
        self.head_ici = nn.Linear(self.cfg.hidden_size, 1)  # regression in feature space

    def forward(
        self,
        whale_ids: torch.Tensor,   # (B,T) long
        ici_feats: torch.Tensor,   # (B,T) float
        tok_types: torch.Tensor,   # (B,T) long
        attn_mask: torch.Tensor,   # (B,T) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          eos_logits: (B,T) logits for "next is EOS"
          ici_pred:   (B,T) predicted next ICI feature (normalized)
        """
        w = self.whale_emb(whale_ids)          # (B,T,Ew)
        t = self.type_emb(tok_types)           # (B,T,Et)
        x = torch.cat([w, t, ici_feats.unsqueeze(-1)], dim=-1)

        lengths = attn_mask.long().sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths=lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,T,H)

        eos_logits = self.head_eos(out).squeeze(-1)  # (B,T)
        ici_pred = self.head_ici(out).squeeze(-1)    # (B,T)
        return eos_logits, ici_pred


# =============================
# Training / Evaluation
# =============================

def compute_losses(
    eos_logits: torch.Tensor,
    ici_pred: torch.Tensor,
    next_is_eos: torch.Tensor,
    next_ici_feat: torch.Tensor,
    loss_mask_eos: torch.Tensor,
    loss_mask_ici: torch.Tensor,
    cfg: BCConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    EOS loss: BCEWithLogits on suffix transitions
    ICI loss: MSE on suffix transitions where next token is ICI
    """
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    eos_loss_all = bce(eos_logits, next_is_eos)  # (B,T)
    eos_loss = (eos_loss_all[loss_mask_eos]).mean() if loss_mask_eos.any() else torch.tensor(0.0, device=eos_logits.device)

    ici_loss_all = mse(ici_pred, next_ici_feat)  # (B,T)
    ici_loss = (ici_loss_all[loss_mask_ici]).mean() if loss_mask_ici.any() else torch.tensor(0.0, device=ici_pred.device)

    total = cfg.w_eos * eos_loss + cfg.w_ici * ici_loss
    stats = {
        "loss_total": float(total.detach().cpu().item()),
        "loss_eos": float(eos_loss.detach().cpu().item()),
        "loss_ici": float(ici_loss.detach().cpu().item()),
    }
    return total, stats


@torch.no_grad()
def evaluate_bc(model: nn.Module, loader: DataLoader, device: str = "cpu") -> Dict[str, float]:
    model.eval()
    totals = {"loss_total": 0.0, "loss_eos": 0.0, "loss_ici": 0.0}
    n = 0

    cfg: BCConfig = model.cfg if hasattr(model, "cfg") else BCConfig()

    for batch in loader:
        (
            whale_ids, ici_feats, tok_types, attn_mask,
            next_is_eos, next_ici_feat, loss_mask_eos, loss_mask_ici
        ) = batch

        whale_ids = whale_ids.to(device)
        ici_feats = ici_feats.to(device)
        tok_types = tok_types.to(device)
        attn_mask = attn_mask.to(device)

        next_is_eos = next_is_eos.to(device)
        next_ici_feat = next_ici_feat.to(device)
        loss_mask_eos = loss_mask_eos.to(device)
        loss_mask_ici = loss_mask_ici.to(device)

        eos_logits, ici_pred = model(whale_ids, ici_feats, tok_types, attn_mask)
        total, stats = compute_losses(
            eos_logits, ici_pred,
            next_is_eos, next_ici_feat,
            loss_mask_eos, loss_mask_ici,
            cfg,
        )

        for k in totals:
            totals[k] += stats[k]
        n += 1

    if n == 0:
        return totals
    return {k: v / n for k, v in totals.items()}


def train_bc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    show_progress: bool = True,
    log_every: int = 50,
) -> List[Dict[str, float]]:
    model.to(device)
    cfg: BCConfig = model.cfg if hasattr(model, "cfg") else BCConfig()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[Dict[str, float]] = []

    for ep in range(1, epochs + 1):
        model.train()

        running_total = 0.0
        running_eos = 0.0
        running_ici = 0.0
        n_steps = 0

        pbar = _get_pbar(train_loader, enabled=show_progress, desc=f"BC train ep {ep}/{epochs}")

        for step, batch in enumerate(pbar, start=1):
            (
                whale_ids, ici_feats, tok_types, attn_mask,
                next_is_eos, next_ici_feat, loss_mask_eos, loss_mask_ici
            ) = batch

            whale_ids = whale_ids.to(device)
            ici_feats = ici_feats.to(device)
            tok_types = tok_types.to(device)
            attn_mask = attn_mask.to(device)

            next_is_eos = next_is_eos.to(device)
            next_ici_feat = next_ici_feat.to(device)
            loss_mask_eos = loss_mask_eos.to(device)
            loss_mask_ici = loss_mask_ici.to(device)

            eos_logits, ici_pred = model(whale_ids, ici_feats, tok_types, attn_mask)
            loss, stats = compute_losses(
                eos_logits, ici_pred,
                next_is_eos, next_ici_feat,
                loss_mask_eos, loss_mask_ici,
                cfg,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            opt.step()

            # Running averages
            running_total += stats["loss_total"]
            running_eos += stats["loss_eos"]
            running_ici += stats["loss_ici"]
            n_steps += 1

            if show_progress and hasattr(pbar, "set_postfix"):
                if (step % max(1, log_every)) == 0 or step == 1:
                    pbar.set_postfix({
                        "loss": f"{running_total/n_steps:.4f}",
                        "eos": f"{running_eos/n_steps:.4f}",
                        "ici": f"{running_ici/n_steps:.4f}",
                    })

        row: Dict[str, float] = {
            "epoch": float(ep),
            "train_loss_total": running_total / max(1, n_steps),
            "train_loss_eos": running_eos / max(1, n_steps),
            "train_loss_ici": running_ici / max(1, n_steps),
        }

        if val_loader is not None:
            val_stats = evaluate_bc(model, val_loader, device=device)
            row.update({f"val_{k}": v for k, v in val_stats.items()})

        history.append(row)

        # Optional epoch summary print (works even without tqdm)
        if show_progress:
            msg = (
                f"[BC] ep {ep}/{epochs} "
                f"train_total={row['train_loss_total']:.4f} "
                f"train_eos={row['train_loss_eos']:.4f} "
                f"train_ici={row['train_loss_ici']:.4f}"
            )
            if val_loader is not None:
                msg += (
                    f" | val_total={row.get('val_loss_total', float('nan')):.4f} "
                    f"val_eos={row.get('val_loss_eos', float('nan')):.4f} "
                    f"val_ici={row.get('val_loss_ici', float('nan')):.4f}"
                )
            print(msg)

    return history


# =============================
# Rollout / Generation
# =============================

@torch.no_grad()
def rollout_coda(
    model: GRUBCModel,
    history_tokens: List[Tuple[int, int, float]],
    current_whale: int,
    max_len: int,
    ici_mean: float,
    ici_std: float,
    eos_threshold: float = 0.5,
    device: str = "cpu",
) -> List[float]:
    """
    Generate a single coda autoregressively.
    history_tokens: list of (whale_id, tok_type, ici_feat) representing prior conversation
                   (including EOS tokens between codas if you want).
    We append [SOC] for current_whale and then generate ICI until EOS or max_len.

    Returns: list of generated raw ICIs (excluding EOS).
    """
    cfg = model.cfg
    model.eval()
    model.to(device)

    seq_w: List[int] = [w for (w, _, _) in history_tokens]
    seq_t: List[int] = [tt for (_, tt, _) in history_tokens]
    seq_f: List[float] = [f for (_, _, f) in history_tokens]

    # append SOC
    seq_w.append(int(current_whale))
    seq_t.append(cfg.TOK_SOC)
    seq_f.append(0.0)

    generated: List[float] = []

    for _ in range(max_len):
        whale_ids = torch.tensor([seq_w], dtype=torch.long, device=device)
        tok_types = torch.tensor([seq_t], dtype=torch.long, device=device)
        ici_feats = torch.tensor([seq_f], dtype=torch.float32, device=device)
        attn_mask = torch.ones_like(ici_feats, dtype=torch.bool, device=device)

        eos_logits, ici_pred = model(whale_ids, ici_feats, tok_types, attn_mask)
        eos_prob = torch.sigmoid(eos_logits[0, -1]).item()

        if eos_prob >= eos_threshold:
            # terminate coda
            break

        # predicted next ICI feature -> raw ICI
        feat = float(ici_pred[0, -1].item())
        ici = feat_to_ici(feat, cfg, ici_mean, ici_std)
        ici = max(ici, cfg.clamp_min_ici)

        generated.append(ici)

        # append generated ICI token
        seq_w.append(int(current_whale))
        seq_t.append(cfg.TOK_ICI)
        seq_f.append(ici_to_feat(ici, cfg, ici_mean, ici_std))

    return generated


# =============================
# Helper: build history for rollout from dataframe
# =============================

def build_history_tokens_from_df(
    df: pd.DataFrame,
    cfg: BCConfig,
    ici_mean: float,
    ici_std: float,
    rec: str,
    upto_row_pos: int,
) -> List[Tuple[int, int, float]]:
    """
    Create (whale_id, tok_type, ici_feat) tokens for all rows in REC before upto_row_pos.
    Useful for evaluation/rollout in notebooks.
    Assumes df already sorted by TsTo, and you have rows positions per REC externally.
    """
    # Build row indices in rec order
    rows = df.index[df["REC"] == rec].tolist()
    rows = rows[:upto_row_pos]

    tokens: List[Tuple[int, int, float]] = []
    for row_idx in rows:
        w = int(df.at[row_idx, "Whale"])
        seq = extract_coda_ici(df, row_idx, cfg)
        for ici in seq:
            tokens.append((w, cfg.TOK_ICI, ici_to_feat(ici, cfg, ici_mean, ici_std)))
        tokens.append((w, cfg.TOK_EOS, 0.0))
    return tokens


# =============================
# Discriminator (Evaluation)
# =============================

@dataclass
class DiscConfig:
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.0
    whale_emb_dim: int = 32
    type_emb_dim: int = 16
    grad_clip_norm: float = 1.0


class GRUDiscriminator(nn.Module):
    """
    Sequence discriminator: predicts whether a full (history + SOC + coda + EOS) sequence
    is real (from dataset) or fake (generated by the BC model).

    Output: logit for "real" (1=real, 0=fake)
    """
    def __init__(self, n_whales: int, cfg: Optional[DiscConfig] = None, bc_cfg: Optional[BCConfig] = None):
        super().__init__()
        self.cfg = cfg or DiscConfig()
        self.bc_cfg = bc_cfg or BCConfig()

        self.whale_emb = nn.Embedding(n_whales, self.cfg.whale_emb_dim)
        self.type_emb = nn.Embedding(4, self.cfg.type_emb_dim)  # PAD/ICI/EOS/SOC

        in_dim = self.cfg.whale_emb_dim + self.cfg.type_emb_dim + 1
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            batch_first=True,
            dropout=self.cfg.dropout if self.cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(self.cfg.hidden_size, 1)

    def forward(
        self,
        whale_ids: torch.Tensor,   # (B,T) long
        ici_feats: torch.Tensor,   # (B,T) float
        tok_types: torch.Tensor,   # (B,T) long
        attn_mask: torch.Tensor,   # (B,T) bool
    ) -> torch.Tensor:
        """
        Returns logits (B,) for "real"
        """
        w = self.whale_emb(whale_ids)
        t = self.type_emb(tok_types)
        x = torch.cat([w, t, ici_feats.unsqueeze(-1)], dim=-1)

        lengths = attn_mask.long().sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.gru(packed)  # h_n: (num_layers, B, H)
        h_last = h_n[-1]  # (B,H)

        logits = self.head(h_last).squeeze(-1)  # (B,)
        return logits


def _tokens_to_tensors(
    tokens: List[Tuple[int, int, float]],
    cfg: BCConfig,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert token triples -> single-sequence tensors of shape (1,T).
    """
    w = torch.tensor([[t[0] for t in tokens]], dtype=torch.long, device=device)
    tt = torch.tensor([[t[1] for t in tokens]], dtype=torch.long, device=device)
    f = torch.tensor([[t[2] for t in tokens]], dtype=torch.float32, device=device)
    m = torch.ones_like(f, dtype=torch.bool, device=device)
    return w, f, tt, m


def _build_episode_tokens_real(
    ds: WhaleBCDataset,
    episode_index: int,
) -> List[Tuple[int, int, float]]:
    """
    Build full token list for the REAL episode:
      [history tokens] + [SOC] + [ICI... ] + [EOS]
    Returns list of (whale_id, tok_type, ici_feat)
    """
    ex = ds[episode_index]
    tokens: List[Tuple[int, int, float]] = []
    for w, tt, f in zip(ex["whale_ids"], ex["tok_types"], ex["ici_feats"]):
        tokens.append((int(w), int(tt), float(f)))
    return tokens


@torch.no_grad()
def _build_episode_tokens_fake(
    ds: WhaleBCDataset,
    episode_index: int,
    bc_model: GRUBCModel,
    max_len: int,
    eos_threshold: float = 0.5,
    device: str = "cpu",
) -> List[Tuple[int, int, float]]:
    """
    Build full token list for the FAKE episode:
      [history tokens] + [SOC] + [generated ICI...] + [EOS]
    Conditioned on the same history and same current whale as the real episode.
    """
    cfg = ds.cfg
    bc_model.eval()
    bc_model.to(device)

    ex = ds[episode_index]
    whale_ids = ex["whale_ids"]
    tok_types = ex["tok_types"]
    ici_feats = ex["ici_feats"]

    # Find SOC index (start of suffix)
    try:
        soc_idx = tok_types.index(cfg.TOK_SOC)
    except ValueError:
        raise RuntimeError("Could not find SOC token in example. Check dataset construction.")

    # History tokens are everything before SOC
    history_tokens: List[Tuple[int, int, float]] = []
    for w, tt, f in zip(whale_ids[:soc_idx], tok_types[:soc_idx], ici_feats[:soc_idx]):
        history_tokens.append((int(w), int(tt), float(f)))

    # Current whale is the whale at SOC token
    current_whale = int(whale_ids[soc_idx])

    # Generate with your rollout helper (returns raw ICIs)
    gen_ici_raw = rollout_coda(
        model=bc_model,
        history_tokens=history_tokens,
        current_whale=current_whale,
        max_len=max_len,
        ici_mean=ds.ici_mean,
        ici_std=ds.ici_std,
        eos_threshold=eos_threshold,
        device=device,
    )

    # Convert generated raw ICIs -> tokens
    tokens = list(history_tokens)
    tokens.append((current_whale, cfg.TOK_SOC, 0.0))
    for ici in gen_ici_raw:
        tokens.append((current_whale, cfg.TOK_ICI, ici_to_feat(ici, cfg, ds.ici_mean, ds.ici_std)))
    tokens.append((current_whale, cfg.TOK_EOS, 0.0))
    return tokens


def _pad_disc_batch(
    batch_tokens: List[List[Tuple[int, int, float]]],
    cfg: BCConfig,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad a list of token sequences into (B,T) tensors.
    """
    B = len(batch_tokens)
    T = max(len(x) for x in batch_tokens)

    whale_ids = torch.zeros((B, T), dtype=torch.long, device=device)
    tok_types = torch.full((B, T), cfg.TOK_PAD, dtype=torch.long, device=device)
    ici_feats = torch.zeros((B, T), dtype=torch.float32, device=device)
    attn_mask = torch.zeros((B, T), dtype=torch.bool, device=device)

    for i, seq in enumerate(batch_tokens):
        t = len(seq)
        whale_ids[i, :t] = torch.tensor([a for (a, _, _) in seq], dtype=torch.long, device=device)
        tok_types[i, :t] = torch.tensor([b for (_, b, _) in seq], dtype=torch.long, device=device)
        ici_feats[i, :t] = torch.tensor([c for (_, _, c) in seq], dtype=torch.float32, device=device)
        attn_mask[i, :t] = True

    return whale_ids, ici_feats, tok_types, attn_mask


def _binary_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    AUROC for binary labels without sklearn.
    probs: predicted probability of label==1 (real)
    labels: 0/1
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    # Sort by prob ascending
    order = np.argsort(probs)
    labels_sorted = labels[order]

    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank-sum method (Mann–Whitney U)
    ranks = np.arange(1, len(labels_sorted) + 1, dtype=np.float64)
    sum_ranks_pos = ranks[labels_sorted == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def make_disc_loader_from_dataset(
    ds: WhaleBCDataset,
    bc_model: GRUBCModel,
    num_pairs: int,
    batch_size: int,
    max_len: int,
    eos_threshold: float = 0.5,
    shuffle: bool = True,
    seed: int = 0,
    device_for_generation: str = "cpu",
) -> DataLoader:
    """
    Build a DataLoader of discriminator examples by sampling `num_pairs` episodes.
    For each sampled episode i, we create:
      - one REAL example (label 1)
      - one FAKE example (label 0) generated by bc_model conditioned on same history + whale

    Returns DataLoader yielding:
      whale_ids, ici_feats, tok_types, attn_mask, labels
    """

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=num_pairs, replace=(num_pairs > len(ds))).tolist()

    cfg = ds.cfg
    sequences: List[List[Tuple[int, int, float]]] = []
    labels: List[int] = []

    for i in idxs:
        real_tokens = _build_episode_tokens_real(ds, i)
        fake_tokens = _build_episode_tokens_fake(
            ds, i, bc_model=bc_model, max_len=max_len, eos_threshold=eos_threshold, device=device_for_generation
        )
        sequences.append(real_tokens); labels.append(1)
        sequences.append(fake_tokens); labels.append(0)

    # Optionally shuffle examples (real/fake mixed)
    if shuffle:
        perm = rng.permutation(len(sequences)).tolist()
        sequences = [sequences[p] for p in perm]
        labels = [labels[p] for p in perm]

    class _DiscMemDataset(Dataset):
        def __len__(self): return len(sequences)
        def __getitem__(self, i):
            return sequences[i], labels[i]

    def _collate(batch):
        seqs = [b[0] for b in batch]
        labs = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        whale_ids, ici_feats, tok_types, attn_mask = _pad_disc_batch(seqs, cfg=cfg, device="cpu")
        return whale_ids, ici_feats, tok_types, attn_mask, labs

    return DataLoader(_DiscMemDataset(), batch_size=batch_size, shuffle=False, collate_fn=_collate)


def train_discriminator(
    disc: GRUDiscriminator,
    loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
    show_progress: bool = True,
    log_every: int = 50,
) -> List[Dict[str, float]]:
    disc.to(device)
    disc.train()

    opt = torch.optim.Adam(disc.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    hist: List[Dict[str, float]] = []

    for ep in range(1, epochs + 1):
        disc.train()
        total_loss = 0.0
        n_batches = 0

        pbar = _get_pbar(loader, enabled=show_progress, desc=f"DISC train ep {ep}/{epochs}")

        for step, (whale_ids, ici_feats, tok_types, attn_mask, labels) in enumerate(pbar, start=1):
            whale_ids = whale_ids.to(device)
            ici_feats = ici_feats.to(device)
            tok_types = tok_types.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            logits = disc(whale_ids, ici_feats, tok_types, attn_mask)
            loss = bce(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if disc.cfg.grad_clip_norm is not None and disc.cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(disc.parameters(), disc.cfg.grad_clip_norm)
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            n_batches += 1

            if show_progress and hasattr(pbar, "set_postfix"):
                if (step % max(1, log_every)) == 0 or step == 1:
                    pbar.set_postfix({"loss": f"{total_loss/max(1,n_batches):.4f}"})

        row = {"epoch": float(ep), "disc_loss": total_loss / max(1, n_batches)}
        hist.append(row)

        if show_progress:
            print(f"[DISC] ep {ep}/{epochs} loss={row['disc_loss']:.4f}")

    return hist


@torch.no_grad()
def evaluate_discriminator(
    disc: GRUDiscriminator,
    loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Returns accuracy and AUROC on loader.
    """
    disc.eval()
    disc.to(device)

    all_probs: List[float] = []
    all_labels: List[int] = []

    for whale_ids, ici_feats, tok_types, attn_mask, labels in loader:
        whale_ids = whale_ids.to(device)
        ici_feats = ici_feats.to(device)
        tok_types = tok_types.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        logits = disc(whale_ids, ici_feats, tok_types, attn_mask)
        probs = torch.sigmoid(logits)

        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())

    probs_np = np.asarray(all_probs, dtype=np.float64)
    labels_np = np.asarray(all_labels, dtype=np.int64)
    preds_np = (probs_np >= 0.5).astype(np.int64)

    acc = float((preds_np == labels_np).mean()) if len(labels_np) else float("nan")
    auc = _binary_auc(probs_np, labels_np) if len(labels_np) else float("nan")

    return {"disc_acc": acc, "disc_auc": auc}

# =============================
# Progress helper (optional tqdm)
# =============================

def _get_pbar(iterable, enabled: bool, desc: str = ""):
    """
    Wrap an iterable with tqdm if available and enabled; otherwise return iterable.
    """
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, leave=False)
    except Exception:
        return iterable