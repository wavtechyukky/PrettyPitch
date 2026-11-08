"""Frame expansion without a Python loop — shared by training, inference and ONNX export.

Every function takes a ``frames`` template [B, T] (any dtype; only its shape is used)
instead of an integer T, so the frame count stays dynamic when the code is traced into
an ONNX graph. Durations may be zero: a token of length 0 owns no frame.

  frame_index        token index of every frame (scatter into T+1 slots, then cumsum)
  length_regulate    token features -> frame features, zeros past each item's length
  position_in_token  0 -> 1 ramp inside each token (linspace(0, 1, d); 0 when d == 1)
"""
from __future__ import annotations

import torch


def frames_for(durations: torch.Tensor) -> torch.Tensor:
    """A [B, T] template for the longest item (T = max total duration). Not traceable —
    exporters pass a template derived from an input tensor instead."""
    T = int(durations.sum(dim=1).max().item()) if durations.numel() else 0
    return torch.zeros(durations.size(0), T, device=durations.device)


def frame_index(durations: torch.Tensor, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """durations [B, L] (int, >= 0, each row summing to at most T) ->
    (index [B, T] long, valid [B, T] bool).

    Frame t belongs to token idx[t]. Frames past an item's total length are not valid
    (their index points at a trailing token). A zero-length token is skipped.
    """
    d = durations.long()
    ends = torch.cumsum(d, dim=1)
    starts = ends - d                                                   # each <= T
    slots = torch.zeros_like(torch.cat([frames, frames[:, :1]], dim=1), dtype=torch.long)
    counts = slots.scatter_add(1, starts, torch.ones_like(starts))      # [B, T+1]
    idx = (torch.cumsum(counts, dim=1)[:, :-1] - 1).clamp(min=0)
    t = torch.cumsum(torch.ones_like(frames, dtype=torch.long), dim=1) - 1
    valid = t < ends[:, -1:]
    return idx, valid


def length_regulate(feat: torch.Tensor, durations: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Token features [B, L, D] -> frame features [B, T, D] (zeros past each length)."""
    idx, valid = frame_index(durations, frames)
    out = torch.gather(feat, 1, idx.unsqueeze(-1).expand(-1, -1, feat.size(-1)))
    return out * valid.unsqueeze(-1).to(out.dtype)


def position_in_token(durations: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """0 -> 1 ramp inside each token [B, T] (float), 0 past each length."""
    d = durations.long()
    idx, valid = frame_index(d, frames)
    starts = torch.cumsum(d, dim=1) - d
    s = torch.gather(starts, 1, idx).float()
    n = torch.gather(d, 1, idx).float()
    t = (torch.cumsum(torch.ones_like(frames, dtype=torch.long), dim=1) - 1).float()
    pos = (t - s) / torch.clamp(n - 1.0, min=1.0)
    return pos * valid.float()
