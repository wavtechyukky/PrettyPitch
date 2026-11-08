"""Train the PrettyPitch 3ch F0 diffusion model (plain / vibrato / v-uv).

    python -m train --config configs/ja_example.yaml \
        --data_dirs data/oniku data/natsume data/ritsu data/ritsu_normal data/ritsu_soft \
        --run_name pp_3singer --device cuda

Config = YAML (sections audio/model/pitch/decomp/train/gan/data). Re-running the same
command auto-resumes from the latest checkpoint in log/<run_name>/. `--init_from CKPT
--finetune` loads generator weights only (fresh D, step 0) for architecture-frozen
fine-tuning. Old hop-512 checkpoints cannot warm-start (the pitch input representation
changed).
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from prettypitch.models import (F0Diffusion3ch, F0Reflow3ch, MultiScaleDisc,
                                d_loss_fn, g_adv_fm_fn)
from prettypitch.losses import (STFTMagnitudeLoss, stft_configs_for,
                                _get_delta_kernels, _apply_delta, _delta_valid,
                                W_DELTA, W_DELTA_DELTA)
from prettypitch.config import merge_per_db_from_recipes
from prettypitch.pitch import hz_to_midi
from preprocess.vocab import Vocab
from dataset import PrettyPitchF0Dataset, f0_collate_fn, FrameBasedBatchSampler

# Phase-locked vibrato L1 covers every non-nucleus frame plus the FIRST vib_l1_tail_sec
# of the nucleus that follows a non-nucleus run — the scoop / onset transient, whose phase
# is fixed by the data. Vibrato proper starts ~0.1 s after the consonant, so 0.2 s is the
# ceiling; the old 0.3 s extended past every non-nucleus frame regardless of phoneme
# boundaries and ran through short vowels into the next consonant and the next vowel.
# The tail now stops at the end of its own phoneme and never crosses into the next one.
VIB_L1_TAIL_SEC = 0.2

PLAIN_STD = 0.07     # channel scales (kept from the proven recipe; stored in the ckpt)
VIB_STD = 0.04


# ── loss ─────────────────────────────────────────────────────────────────────

def compute_loss(model, batch, device, frame_rate, plain_std, vib_std,
                 stft_loss_fn=None, stft_loss_weight=0.0,
                 w_delta=W_DELTA, w_delta_delta=W_DELTA_DELTA,
                 vib_l1_tail_sec=VIB_L1_TAIL_SEC):
    """Returns (recon_total, loss_dict, aux). aux feeds the GAN step (ignored without GAN)."""
    b = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    cond = model.cond_from_inputs(b["ph_ids"], b["ph_pitch"], b["ph_durs"],
                                  spk_id=b["spk_id"], style_id=b["style_id"])

    x0 = torch.stack([b["plain"] / plain_std,
                      b["vibrato"] / vib_std], dim=1)       # [B,2,T] (plain, vibrato)

    T = min(x0.shape[2], cond.shape[2])
    x0, cond = x0[:, :, :T], cond[:, :, :T]
    valid = ~b["frame_mask"][:, :T]
    is_nuc = b["is_nucleus"][:, :T] > 0.5
    uv = b["uv"][:, :T]
    if not valid.any():
        z = x0.sum() * 0.0
        return z, {"x0": 0., "delta": 0., "delta_delta": 0.,
                   "vib": 0., "stft": 0., "vuv": 0.}, None

    Bs = x0.shape[0]
    is_reflow = hasattr(model, "predict_velocity")
    if is_reflow:
        # rectified flow: straight noise→data path, predict velocity v = x1 − x0(noise)
        z = torch.randn_like(x0)
        t = torch.rand(Bs, device=device)
        x_t = z + t[:, None, None] * (x0 - z)
        v_pred = model.predict_velocity(x_t, cond, t)
        x0_pred = x_t + (1.0 - t)[:, None, None] * v_pred  # 1-step clean estimate
        vm = valid.unsqueeze(1).expand(-1, x0.shape[1], -1)
        loss_vel = (v_pred - (x0 - z)).abs()[vm].mean()
        ab = None
    else:
        t = torch.randint(0, model.n_steps, (Bs,), device=device)
        ab = model.alpha_bar[t]
        eps = torch.randn_like(x0)
        x_t = torch.sqrt(ab[:, None, None]) * x0 + torch.sqrt(1 - ab[:, None, None]) * eps
        x0_pred = model.predict_x0(x_t, cond, t)           # [B,2,T]
        loss_vel = x0.sum() * 0.0

    pl_p, vb_p = x0_pred[:, 0], x0_pred[:, 1]
    pl_g, vb_g = x0[:, 0], x0[:, 1]

    k_d, k_dd = _get_delta_kernels(device)
    valid_d = _delta_valid(valid)

    # plain: value L1 + Δ + ΔΔ
    loss_x0 = F.l1_loss(pl_p[valid], pl_g[valid])
    if valid_d.any():
        loss_delta = F.l1_loss(_apply_delta(pl_p, k_d)[valid_d], _apply_delta(pl_g, k_d)[valid_d])
        loss_dd = F.l1_loss(_apply_delta(pl_p, k_dd)[valid_d], _apply_delta(pl_g, k_dd)[valid_d])
    else:
        loss_delta = loss_x0 * 0.0
        loss_dd = loss_x0 * 0.0

    # vibrato: (non-nucleus + tail) L1 with phase, + phase-insensitive STFT on voiced.
    # tail = the first n_ext frames of a nucleus phoneme that starts right after a
    # non-nucleus frame. It is bounded by that phoneme's own end (ph_durs), so a vowel
    # shorter than the tail is covered whole and nothing spills into the phoneme after it;
    # a melisma vowel following another vowel gets no tail (no consonant before it).
    cons_core = valid & (~is_nuc)
    n_ext = int(round(vib_l1_tail_sec * frame_rate))
    if n_ext > 0:
        durs = b["ph_durs"]                                          # [B,L] frames, 0 = pad
        ends = torch.cumsum(durs, dim=1)                             # exclusive end of phoneme
        t_idx = torch.arange(T, device=device).unsqueeze(0).expand(Bs, T)
        seg = torch.searchsorted(ends, t_idx, right=True).clamp(max=durs.shape[1] - 1)
        seg_start = torch.gather(ends - durs, 1, seg)                # [B,T] start of own phoneme
        prev_nuc = torch.gather(is_nuc.float(), 1, (seg_start - 1).clamp(min=0)) > 0.5
        after_cons = (~prev_nuc) | (seg_start == 0)
        tail = is_nuc & after_cons & ((t_idx - seg_start) < n_ext)
        consonant = (cons_core | tail) & valid
    else:
        consonant = cons_core
    loss_vib = (F.l1_loss(vb_p[consonant], vb_g[consonant])
                if consonant.any() else loss_x0 * 0.0)
    loss_stft = loss_x0.new_zeros(())
    if stft_loss_fn is not None and stft_loss_weight > 0.0:
        voiced = (uv > 0.5) & valid
        if voiced.any():
            loss_stft = stft_loss_fn(vb_p * vib_std, vb_g * vib_std, mask=voiced)

    # v/uv: separate deterministic head over the 256-dim condition (BCE, not diffused)
    uv_logit = model.uv_head(cond).squeeze(1)                    # [B,T]
    loss_uv = F.binary_cross_entropy_with_logits(uv_logit[valid], uv[valid])

    total = (loss_x0 + w_delta * loss_delta + w_delta_delta * loss_dd
             + loss_vib + stft_loss_weight * loss_stft + loss_uv + loss_vel)

    loss_dict = {"x0": loss_x0.item(), "delta": loss_delta.item(),
                 "delta_delta": loss_dd.item(), "vib": loss_vib.item(),
                 "stft": loss_stft.item(), "vuv": loss_uv.item(),
                 "vel": loss_vel.item()}
    # GAN sees plain/vib + the head's v/uv (±1, aligned to the 0/1 target)
    x0_gan = torch.cat([x0, (uv * 2 - 1).unsqueeze(1)], dim=1)                     # [B,3,T]
    x0_pred_gan = torch.cat([x0_pred, (2 * torch.sigmoid(uv_logit) - 1).unsqueeze(1)], dim=1)
    aux = {"x0_pred": x0_pred_gan, "x0": x0_gan, "valid": valid, "snr": ab}
    return total, loss_dict, aux


# ── sampling / eval / plots ──────────────────────────────────────────────────

@torch.no_grad()
def _sample3ch(model, batch, device, plain_std=PLAIN_STD, vib_std=VIB_STD):
    """Sample 3ch and reconstruct Hz. Returns plain_hz, full_hz, voiced [B, T]."""
    b = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    out3 = model.sample(b["ph_ids"], b["ph_pitch"], b["ph_durs"],
                        spk_id=b["spk_id"], style_id=b["style_id"])
    base = b["baseline"]
    T = min(out3.shape[2], base.shape[1])
    out3, base = out3[:, :, :T], base[:, :T]
    plain = out3[:, 0] * plain_std
    vib = out3[:, 1] * vib_std
    vuv = out3[:, 2]                                  # uv probability in [0,1]
    plain_hz = torch.pow(2.0, base + plain)
    full_hz = torch.pow(2.0, base + plain + vib)
    return plain_hz, full_hz, (vuv > 0.5)


@torch.no_grad()
def run_eval(model, eval_loader, device, tb_writer, step, log_fn,
             plain_std=PLAIN_STD, vib_std=VIB_STD):
    """cent RMSE on voiced frames + v/uv frame accuracy."""
    model.eval()
    cent_sq_sum = cent_n = 0
    vuv_ok = vuv_n = 0
    for vbatch in eval_loader:
        _, full_hz, pred_v = _sample3ch(model, vbatch, device, plain_std, vib_std)
        T = full_hz.shape[1]
        gt_hz = torch.as_tensor(vbatch["gt_f0_hz_full"][:, :T], device=device)
        uv = torch.as_tensor(vbatch["uv"][:, :T], device=device)
        valid = ~torch.as_tensor(vbatch["frame_mask"][:, :T], device=device)
        voiced = (uv > 0.5) & valid
        if voiced.any():
            cents = 1200.0 * torch.log2(
                (full_hz[voiced] / gt_hz[voiced].clamp(min=1e-6)).clamp(min=1e-6))
            cent_sq_sum += (cents ** 2).sum().item()
            cent_n += int(voiced.sum().item())
        vuv_ok += int(((pred_v == (uv > 0.5)) & valid).sum().item())
        vuv_n += int(valid.sum().item())
    cent_rmse = (cent_sq_sum / max(cent_n, 1)) ** 0.5
    vuv_acc = vuv_ok / max(vuv_n, 1)
    tb_writer.add_scalar("eval/cent_rmse", cent_rmse, step)
    tb_writer.add_scalar("eval/vuv_acc", vuv_acc, step)
    log_fn(f"  [EVAL] step={step}  cent_rmse={cent_rmse:.2f}c  vuv_acc={vuv_acc:.4f}")
    model.train()


def _hz_to_midi(hz):
    """hz_to_midi with NaN where there is no pitch (hz <= 1)."""
    hz = np.asarray(hz, dtype=np.float64)
    out = np.full(hz.shape, np.nan)
    m = hz > 1.0
    out[m] = hz_to_midi(hz[m])
    return out


def _midi_tick(m: int) -> str:
    return f"C{m // 12 - 1}"


def _plot_f0_vuv(gt_hz, pred_hz, base_hz, plain_hz, gt_vuv, pred_vuv, title, frame_rate,
                 gt_plain_hz=None):
    """F0 panel (MIDI) + GT/Pred v/uv strips."""
    T = len(gt_hz)
    t = np.arange(T) / frame_rate
    gt_m, pred_m = _hz_to_midi(gt_hz), _hz_to_midi(pred_hz)
    base_m, plain_m = _hz_to_midi(base_hz), _hz_to_midi(plain_hz)
    gt_plain_m = _hz_to_midi(gt_plain_hz) if gt_plain_hz is not None else None

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(3, 1, height_ratios=[6, 1, 1], hspace=0.1)

    ax = fig.add_subplot(gs[0])
    ax.plot(t, base_m, color="#88aacc", lw=1.0, ls="--", alpha=0.7, label="Baseline")
    ax.plot(t, gt_m, color="#22aa44", lw=1.4, alpha=0.55, label="GT F0")
    ax.plot(t, pred_m, color="#dd4444", lw=1.4, alpha=0.55, label="Pred F0")
    if gt_plain_m is not None:
        ax.plot(t, gt_plain_m, color="#ff8800", lw=2.0, label="GT plain")
    ax.plot(t, plain_m, color="#d000d0", lw=2.0, label="Pred plain")
    vals = [v[~np.isnan(v)] for v in (gt_m, pred_m, base_m) if np.any(~np.isnan(v))]
    if vals:
        a = np.concatenate(vals)
        y_lo, y_hi = int(np.floor(a.min())) - 1, int(np.ceil(a.max())) + 1
    else:
        y_lo, y_hi = 48, 84
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks([m for m in range(y_lo, y_hi + 1) if m % 12 == 0])
    ax.set_yticklabels([_midi_tick(m) for m in range(y_lo, y_hi + 1) if m % 12 == 0])
    ax.yaxis.grid(True, color="#cccccc", lw=0.8)
    ax.set_ylabel("MIDI note")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    plt.setp(ax.get_xticklabels(), visible=False)

    for gs_i, (vuv, c, lbl) in enumerate([(gt_vuv, "#22aa44", "GT v/uv"),
                                          (pred_vuv, "#dd4444", "Pred v/uv")], start=1):
        axv = fig.add_subplot(gs[gs_i], sharex=ax)
        axv.fill_between(t, 0, vuv, step="mid", color=c, alpha=0.6, linewidth=0)
        axv.set_ylim(-0.1, 1.1)
        axv.set_yticks([])
        axv.set_ylabel(lbl, fontsize=7, rotation=0, ha="right", va="center")
        if gs_i == 2:
            axv.set_xlabel("Time (s)")
        else:
            plt.setp(axv.get_xticklabels(), visible=False)

    fig.tight_layout()
    return fig


def _plot_vibrato(gt_vib_c, pred_vib_c, voiced, title, frame_rate):
    """Teacher vs generated vibrato (log2 deviation → cents) + diff panel.

    Full-range curves — the model generates vibrato on EVERY frame, so nothing is
    masked out. Unvoiced regions are shaded light gray instead; the diff RMS stays
    voiced-only (the teacher is ~0 in unvoiced regions)."""
    T = len(gt_vib_c)
    t = np.arange(T) / frame_rate
    v = voiced.astype(bool)
    diff = pred_vib_c - gt_vib_c
    rms = np.sqrt(np.mean(diff[v] ** 2)) if v.any() else 0.0

    unv = (~v).astype(int)
    edges = np.diff(np.concatenate(([0], unv, [0])))
    spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.12)

    ax = fig.add_subplot(gs[0])
    for a, b in spans:
        ax.axvspan(a / frame_rate, b / frame_rate, color="#000000", alpha=0.05, lw=0)
    ax.plot(t, gt_vib_c, color="#ff8800", lw=1.4, label="GT vibrato")
    ax.plot(t, pred_vib_c, color="#d000d0", lw=1.4, alpha=0.85, label="Pred vibrato")
    ax.axhline(0, color="#bbbbbb", lw=0.6)
    ax.set_ylabel("vibrato [cents]")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    plt.setp(ax.get_xticklabels(), visible=False)

    axd = fig.add_subplot(gs[1], sharex=ax)
    for a, b in spans:
        axd.axvspan(a / frame_rate, b / frame_rate, color="#000000", alpha=0.05, lw=0)
    axd.plot(t, diff, color="#dd2222", lw=1.0)
    axd.axhline(0, color="#bbbbbb", lw=0.6)
    axd.set_ylabel("Pred − GT [cents]")
    axd.set_xlabel("Time (s)")
    axd.text(0.01, 0.9, f"diff RMS = {rms:.0f} cents (voiced)", transform=axd.transAxes,
             fontsize=8, va="top")

    fig.tight_layout()
    return fig


def select_plot_items(ds, n_per_db: int, seed: int = 0):
    """DB（話者×スタイル単位）ごとに n 件ずつ、決定論的に選んでプロット項目にする。

    Returns: [(label, collated_batch), ...]  label = "<db>/<phrase名>"（TBで階層表示）。
    """
    groups: dict = {}
    for i, db in enumerate(ds.db_ids):
        groups.setdefault(db, []).append(i)
    out = []
    for gi, db in enumerate(sorted(groups)):
        idxs = groups[db]
        rng = random.Random(seed * 1000 + gi)
        for i in sorted(rng.sample(idxs, min(n_per_db, len(idxs)))):
            out.append((f"{db}/{ds.files[i][1]}", f0_collate_fn([ds[i]])))
    return out


@torch.no_grad()
def run_plots(model, plot_items, device, tb_writer, step, frame_rate,
              tb_prefix="f0_eval", plain_std=PLAIN_STD, vib_std=VIB_STD):
    """plot_items: select_plot_items の [(label, batch), ...]。"""
    model.eval()
    for name, item in plot_items:
        plain_hz_t, full_hz_t, voiced_t = _sample3ch(model, item, device, plain_std, vib_std)
        T = full_hz_t.shape[1]
        plain_hz = plain_hz_t[0].cpu().numpy()
        full_hz = full_hz_t[0].cpu().numpy()
        base_logf0 = item["baseline"][0, :T].cpu().numpy()
        base_hz = np.power(2.0, base_logf0)
        gt_hz = item["gt_f0_hz_full"][0, :T].cpu().numpy()
        gt_vuv = item["uv"][0, :T].cpu().numpy()
        pred_vuv = voiced_t[0].cpu().numpy().astype(np.float32)
        gt_plain_hz = np.power(2.0, base_logf0 + item["plain"][0, :T].cpu().numpy())

        fig = _plot_f0_vuv(gt_hz, full_hz, base_hz, plain_hz, gt_vuv, pred_vuv,
                           f"[{tb_prefix}] {name}", frame_rate, gt_plain_hz=gt_plain_hz)
        tb_writer.add_figure(f"{tb_prefix}/{name}", fig, step)
        plt.close(fig)

        gt_vib_c = item["vibrato"][0, :T].cpu().numpy() * 1200.0
        pred_vib_c = (np.log2(np.maximum(full_hz, 1.0))
                      - np.log2(np.maximum(plain_hz, 1.0))) * 1200.0
        figv = _plot_vibrato(gt_vib_c, pred_vib_c, gt_vuv,
                             f"[{tb_prefix}] {name} — vibrato", frame_rate)
        tb_writer.add_figure(f"{tb_prefix}_vib/{name}", figv, step)
        plt.close(figv)
    model.train()


# ── config helpers ───────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    for k in ("audio", "model", "pitch", "decomp", "train", "gan", "data"):
        cfg.setdefault(k, {})
    return cfg


def merge_id_maps(cfg: dict, data_dirs) -> tuple[dict, dict]:
    """話者/スタイル ID は config の data.spk_map / data.style_map **だけ**が決める。

    レシピは「その DB をどう読むか」だけを書く場所で、同一性は書かない。列挙もれは 0 に
    落ちて黙って話者を潰すので、警告して気づけるようにする。
    """
    dcfg = cfg.get("data", {})
    spk_map = dict(dcfg.get("spk_map") or {})
    style_map = dict(dcfg.get("style_map") or {})
    missing = [os.path.basename(os.path.normpath(d)) for d in data_dirs
               if os.path.basename(os.path.normpath(d)) not in spk_map]
    if missing:
        print(f"[warn] data.spk_map に無い DB: {missing} — 話者 0 として学習します。"
              f"別話者なら configs の data.spk_map / style_map に足してください")
    print(f"[spk_map] {spk_map}\n[style_map] {style_map}")
    return spk_map, style_map


def ckpt_config(cfg, ds, vocab, nucleus_list, n_speakers, n_styles) -> dict:
    """Flattened, self-describing model config stored in the checkpoint —
    infer.py rebuilds the model and all derived quantities purely from it."""
    m, p, dcp = cfg["model"], cfg["pitch"], cfg["decomp"]
    arch = str(m.get("arch", "diffusion"))
    assert arch in ("diffusion", "reflow"), arch
    sr, hop = int(cfg["audio"].get("sr", 44100)), int(cfg["audio"].get("hop", 256))
    if abs(float(ds.frame_rate) - sr / hop) > 1e-6:
        raise ValueError(f"config audio.sr/hop = {sr}/{hop} ({sr / hop:.6f} fps) but the shards were "
                         f"preprocessed at {float(ds.frame_rate):.6f} fps — fix the config or re-run preprocess")
    # model.encoder: notes（伸ばしの記号 hold_symbol を前の母音に足し込み、音素の列と音符の列で読む）。
    # model.hold_symbol は、伸ばしの記号で書いたラベルで学習したことの印。エンコーダに依らず ckpt に残す
    # （ホストはこれを見て、伸ばしの母音を記号で渡すかどうかを決める）。
    # 既定（phonemes・記号なし）では ckpt の中身は今までと同じ（キーも足さない）
    encoder = {}
    if str(m.get("encoder", "phonemes")) != "phonemes":
        encoder = {"encoder": str(m["encoder"]), "hold_symbol": m.get("hold_symbol")}
    elif m.get("hold_symbol") is not None:
        encoder = {"hold_symbol": str(m["hold_symbol"])}
    return {
        "arch": "f0_3ch_reflow" if arch == "reflow" else "f0_3ch_diffusion",
        "n_steps": int(m.get("n_steps", 10)),
        "infer_steps": int(m.get("infer_steps", 1)),
        "lang": str(cfg.get("lang", "ja")),
        "n_phonemes": ds.n_phonemes,
        "phonemes": vocab.phonemes,
        "nucleus": list(nucleus_list),
        "sample_rate": sr,
        "hop": hop,
        "frame_rate": ds.frame_rate,
        "n_speakers": n_speakers,
        "spk_dim": int(m.get("spk_dim", 32)),
        "n_styles": n_styles,
        "conditioner": str(m.get("conditioner", "convnext")),
        "convnext_dilations": list(m.get("convnext_dilations", [1, 5, 12, 24])),
        "pitch_center": float(m.get("pitch_center", 60.0)),
        "pitch_scale": float(m.get("pitch_scale", 12.0)),
        "plain_std": float(dcp.get("plain_std", PLAIN_STD)),
        "vib_std": float(dcp.get("vib_std", VIB_STD)),
        "baseline_sigma_sec": float(dcp.get("baseline_sigma_sec", 0.02)),
        "pitch_cfg": dict(p),
        "decomp_cfg": dict(dcp),
        **encoder,
    }


def encoder_kwargs(ckpt_cfg: dict) -> dict:
    """model.encoder: notes の引数（核・pau・伸ばしの記号の番号）を ckpt の音素表から作る。"""
    enc = str(ckpt_cfg.get("encoder", "phonemes"))
    if enc == "phonemes":
        return {}
    ix = {p: i for i, p in enumerate(ckpt_cfg["phonemes"])}
    hold = ckpt_cfg.get("hold_symbol")
    if hold is not None and hold not in ix:
        raise ValueError(f"hold_symbol {hold!r} is not in the phoneme table (--phonemes?)")
    return dict(encoder=enc,
                nucleus_ids=[ix[p] for p in ckpt_cfg["nucleus"] if p in ix and p != hold],
                pau_id=ix.get("pau", 0),
                hold_id=None if hold is None else ix[hold])


def build_model(ckpt_cfg: dict):
    common = dict(
        n_phonemes=int(ckpt_cfg["n_phonemes"]),
        n_speakers=int(ckpt_cfg["n_speakers"]),
        spk_dim=int(ckpt_cfg["spk_dim"]),
        n_styles=int(ckpt_cfg["n_styles"]),
        conditioner=str(ckpt_cfg.get("conditioner", "convnext")),
        convnext_dilations=ckpt_cfg.get("convnext_dilations"),
        **encoder_kwargs(ckpt_cfg),
    )
    if str(ckpt_cfg.get("arch", "f0_3ch_diffusion")) == "f0_3ch_reflow":
        return F0Reflow3ch(num_steps=int(ckpt_cfg.get("infer_steps", 1)), **common)
    return F0Diffusion3ch(n_steps=int(ckpt_cfg["n_steps"]), **common)


# ── training ─────────────────────────────────────────────────────────────────

def train(args):
    # 学習中の図は保存するだけなので非対話バックエンドにする。ここで切り替えるのは、
    # svs/render.py がこのモジュールを import するため（import 時に切り替えると
    # ノートブックのインライン描画が消える）。
    matplotlib.use("Agg")
    cfg = load_config(args.config)
    # dotted-key config overrides for A/B runs, e.g.
    #   --set model.conditioner=convnext --set model.n_steps=4
    for kv in (args.set or []):
        key, _, raw = kv.partition("=")
        node = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            node = node[k]
        node[parts[-1]] = yaml.safe_load(raw)
        print(f"config override: {key} = {node[parts[-1]]!r}")
    tr, gan = cfg["train"], cfg["gan"]
    device = torch.device(args.device)
    print(f"Device: {device}")

    # 言語は config の lang: が宣言する（既定 ja）。--phonemes / --nucleus は
    # その項目だけを上書きする。音素表・核音素はどちらも言語パック経由で解決するので、
    # 学習・前処理・推論で同じ 1 か所を見ることになる。model.hold_symbol（伸ばしの記号）は
    # 音素表と核の末尾に足す（前処理のレシピの hold_symbol と同じ値にする）。
    from duration_model import phoneset as _ps_mod
    lang = str(cfg.get("lang", _ps_mod.DEFAULT_LANG))
    hold = cfg["model"].get("hold_symbol")
    ps = _ps_mod.set_default(_ps_mod.load(lang, args.phonemes, args.nucleus,
                                          None if hold is None else str(hold)))
    vocab = Vocab(list(ps.phonemes))
    nucleus = frozenset(ps.nuclei)
    print(f"言語パック: {ps.lang}  音素{ps.n_phonemes} / 核{len(nucleus)}")
    spk_map, style_map = merge_id_maps(cfg, args.data_dirs)
    merge_per_db_from_recipes(cfg, args.data_dirs)
    n_speakers = (max(spk_map.values()) + 1) if spk_map else 1
    n_styles = int(cfg["model"].get("n_styles",
                                    (max(style_map.values()) + 1) if style_map else 0))

    pitch_cfg = dict(cfg["pitch"])
    pitch_cfg.setdefault("center", cfg["model"].get("pitch_center", 60.0))
    pitch_cfg.setdefault("scale", cfg["model"].get("pitch_scale", 12.0))
    decomp_cfg = dict(cfg["decomp"])
    plain_std = float(decomp_cfg.get("plain_std", PLAIN_STD))
    vib_std = float(decomp_cfg.get("vib_std", VIB_STD))
    _eval_dbs = cfg["data"].get("eval_dbs")
    if _eval_dbs is not None and not _eval_dbs:
        raise ValueError("data.eval_dbs が空です（eval split が空になります）")
    ds_kw = dict(eval_songs=int(cfg["data"].get("eval_songs", 2)),
                 eval_dbs=_eval_dbs,
                 min_sec=float(cfg["data"].get("min_sec", 1.7)),
                 spk_map=spk_map, style_map=style_map,
                 nucleus_names=sorted(nucleus),
                 pitch_cfg=pitch_cfg, decomp_cfg=decomp_cfg,
                 reject_f0_below=cfg["data"].get("reject_f0_below"),
                 reject_f0_per_db=cfg["data"].get("reject_f0_per_db"))
    train_ds = PrettyPitchF0Dataset(args.data_dirs, split="train", pau_aug=True, **ds_kw)
    eval_ds = PrettyPitchF0Dataset(args.data_dirs, split="eval", pau_aug=False, **ds_kw)
    frame_rate = train_ds.frame_rate
    assert train_ds.n_phonemes == vocab.n_phonemes, \
        (f"shard n_phonemes={train_ds.n_phonemes} != vocab {vocab.n_phonemes} "
         f"(--phonemes? レシピの hold_symbol と model.hold_symbol はそろっている?)")

    ck_cfg = ckpt_config(cfg, train_ds, vocab, sorted(nucleus), n_speakers, n_styles)
    model = build_model(ck_cfg).to(device)
    if ck_cfg.get("encoder"):
        print(f"encoder: {ck_cfg['encoder']}  hold_symbol={ck_cfg.get('hold_symbol')!r}"
              f"（伸ばしの記号は前の母音に足し込み、音素の列と音符の列で読む）")
    elif ck_cfg.get("hold_symbol") is not None:
        print(f"hold_symbol={ck_cfg['hold_symbol']!r}（伸ばしの記号を 1 つの音素として読む）")
    use_gan = bool(gan.get("enabled", True))
    D = MultiScaleDisc(n_scales=3).to(device) if use_gan else None
    print("G %.3fM%s  n_speakers=%d n_styles=%d  frame_rate=%.4f  FIR=%dtap" % (
        sum(p.numel() for p in model.parameters()) / 1e6,
        ("  D %.3fM" % (sum(p.numel() for p in D.parameters()) / 1e6)) if D else "",
        n_speakers, n_styles, frame_rate, len(train_ds._fir)))

    stft_w = float(tr.get("stft_loss_weight", 0.1))
    stft_loss_fn = (STFTMagnitudeLoss(stft_configs_for(frame_rate)).to(device)
                    if stft_w > 0 else None)
    if stft_loss_fn:
        print(f"  STFT loss: weight={stft_w}  configs={stft_loss_fn.configs}")

    lr = float(tr.get("lr", 2e-4))
    betas = tuple(tr.get("betas", (0.8, 0.99)))
    optG = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=0.0)
    schedG = torch.optim.lr_scheduler.StepLR(optG, step_size=int(tr.get("lr_step", 5000)),
                                             gamma=float(tr.get("lr_gamma", 0.5)))
    optD = schedD = None
    if D is not None:
        optD = torch.optim.AdamW(D.parameters(), lr=float(gan.get("lr_d", lr)),
                                 betas=betas, weight_decay=0.0)
        schedD = torch.optim.lr_scheduler.StepLR(optD, step_size=int(tr.get("lr_step", 5000)),
                                                 gamma=float(tr.get("lr_gamma", 0.5)))

    # speaker/db-balanced weighted sampling (optional)
    balance_by = tr.get("balance_by")
    weights = None
    if balance_by:
        import collections
        keys = {"speaker": train_ds.speaker_ids, "db": train_ds.db_ids}[balance_by]
        cnt = collections.Counter(keys)
        weights = [1.0 / cnt[k] for k in keys]
        print(f"[balance] by {balance_by}: {dict(cnt)}")

    max_batch_frames = int(tr.get("max_batch_frames", 160000))
    max_batch_size = int(tr.get("max_batch_size", 32))
    train_sampler = FrameBasedBatchSampler(train_ds.frame_counts, max_batch_frames,
                                           max_batch_size, shuffle=True, weights=weights)
    eval_sampler = FrameBasedBatchSampler(eval_ds.frame_counts, max_batch_frames,
                                          max_batch_size, shuffle=False)
    num_workers = int(tr.get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=f0_collate_fn,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    eval_loader = DataLoader(eval_ds, batch_sampler=eval_sampler, collate_fn=f0_collate_fn,
                             num_workers=0)

    out_dir = Path(args.out_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(str(out_dir))

    def log(msg):
        tqdm.write(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # resume / finetune
    global_step = 0
    existing = sorted(out_dir.glob("ckpt_*.pt"))
    if existing:
        ck = torch.load(existing[-1], map_location=device)
        model.load_state_dict(ck["model"])
        if D is not None and ck.get("D") is not None:
            try:
                D.load_state_dict(ck["D"])
                optD.load_state_dict(ck["optD"])
                schedD.load_state_dict(ck["schedD"])
            except Exception as e:  # noqa: BLE001
                print(f"[resume] disc mismatch -> fresh D ({e})")
        optG.load_state_dict(ck["optG"])
        schedG.load_state_dict(ck["schedG"])
        global_step = ck["step"]
        print(f"Resumed: {existing[-1].name}  (step {global_step})")
    elif args.init_from and args.finetune:
        ck = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"Finetune init from {args.init_from} (G only, fresh D, step 0)")

    def save_ckpt(step):
        path = out_dir / f"ckpt_{step:06d}.pt"
        torch.save({"step": step, "model": model.state_dict(),
                    "D": D.state_dict() if D is not None else None,
                    "optG": optG.state_dict(),
                    "optD": optD.state_dict() if optD is not None else None,
                    "schedG": schedG.state_dict(),
                    "schedD": schedD.state_dict() if schedD is not None else None,
                    "config": ck_cfg}, path)
        for old in sorted(out_dir.glob("ckpt_*.pt"))[:-3]:
            old.unlink()
        log(f"  Saved {path}")

    # プロット項目: eval/train とも DB（話者×スタイル）ごとに n 件ずつ全ユニットをカバー。
    # train 側は pau_aug 無しの別インスタンスから選ぶ（決定論的な同一フレーズを毎回描く）。
    plot_items = select_plot_items(eval_ds, int(tr.get("n_plot", 2)), seed=0)
    n_tplot = int(tr.get("n_train_plot", 2))
    train_plot_items = []
    if n_tplot > 0:
        plot_train_ds = PrettyPitchF0Dataset(args.data_dirs, split="train",
                                             pau_aug=False, **ds_kw)
        train_plot_items = select_plot_items(plot_train_ds, n_tplot, seed=1)
    print(f"[plots] eval {len(plot_items)} items / train {len(train_plot_items)} items "
          f"(per-DB coverage)")

    max_steps = int(args.max_steps or tr.get("max_steps", 20000))
    w_adv = float(gan.get("w_adv", 0.1))
    w_fm = float(gan.get("w_fm", 1.0))
    w_delta = float(tr.get("w_delta", W_DELTA))
    w_dd = float(tr.get("w_delta_delta", W_DELTA_DELTA))
    tail = float(tr.get("vib_l1_tail_sec", VIB_L1_TAIL_SEC))
    grad_clip = float(tr.get("grad_clip", 1.0))
    log_interval = int(tr.get("log_interval", 50))
    val_interval = int(tr.get("val_check_interval", 500))
    save_interval = int(tr.get("save_interval", 1000))

    model.train()
    if D is not None:
        D.train()
    pbar = tqdm(total=max_steps, initial=global_step, unit="step", dynamic_ncols=True)
    t0 = time.time()
    epoch = 0

    while global_step < max_steps:
        train_sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            if global_step >= max_steps:
                break

            recon, ld, aux = compute_loss(model, batch, device, frame_rate,
                                          plain_std, vib_std,
                                          stft_loss_fn=stft_loss_fn, stft_loss_weight=stft_w,
                                          w_delta=w_delta, w_delta_delta=w_dd,
                                          vib_l1_tail_sec=tail)
            if aux is None:
                continue

            if D is not None:
                loss_d, ro_feats = d_loss_fn(D, aux)
                optD.zero_grad(); loss_d.backward(); optD.step()
                loss_adv, loss_fm = g_adv_fm_fn(D, aux, ro_feats)
                lossG = recon + w_adv * loss_adv + w_fm * loss_fm
            else:
                loss_d = loss_adv = loss_fm = torch.zeros(())
                lossG = recon
            optG.zero_grad(); lossG.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optG.step(); schedG.step()
            if schedD is not None:
                schedD.step()

            global_step += 1
            pbar.update(1)
            pbar.set_postfix(g=f"{lossG.item():.3f}", d=f"{loss_d.item():.3f}",
                             vib=f'{ld["vib"]:.3f}')

            if global_step % log_interval == 0:
                lr_now = schedG.get_last_lr()[0]
                log(f"step={global_step:6d}  G={lossG.item():.4f} D={loss_d.item():.4f}  "
                    f'x0={ld["x0"]:.4f} vib={ld["vib"]:.4f} stft={ld["stft"]:.4f} '
                    f'vuv={ld["vuv"]:.4f}  adv={loss_adv.item():.4f} fm={loss_fm.item():.4f}  '
                    f"lr={lr_now:.2e}  elapsed={time.time() - t0:.0f}s")
                for k in ("x0", "delta", "delta_delta", "vib", "stft", "vuv"):
                    tb_writer.add_scalar(f"train/loss_{k}", ld[k], global_step)
                tb_writer.add_scalar("train/loss_G", lossG.item(), global_step)
                tb_writer.add_scalar("train/loss_D", loss_d.item(), global_step)
                tb_writer.add_scalar("train/loss_adv", loss_adv.item(), global_step)
                tb_writer.add_scalar("train/loss_fm", loss_fm.item(), global_step)
                tb_writer.add_scalar("train/lr", lr_now, global_step)

            if global_step % val_interval == 0:
                model.eval()
                vs, vn = {"x0": 0., "vib": 0., "stft": 0., "vuv": 0.}, 0
                with torch.no_grad():
                    for vbatch in eval_loader:
                        _, vld, _ = compute_loss(model, vbatch, device, frame_rate,
                                                 plain_std, vib_std,
                                                 stft_loss_fn=stft_loss_fn,
                                                 stft_loss_weight=stft_w,
                                                 w_delta=w_delta, w_delta_delta=w_dd,
                                                 vib_l1_tail_sec=tail)
                        for k in vs:
                            vs[k] += vld[k]
                        vn += 1
                if vn:
                    avg = {k: v / vn for k, v in vs.items()}
                    log(f'  [VAL] step={global_step}  x0={avg["x0"]:.4f} '
                        f'vib={avg["vib"]:.4f} stft={avg["stft"]:.4f} vuv={avg["vuv"]:.4f}')
                    for k, v in avg.items():
                        tb_writer.add_scalar(f"val/loss_{k}", v, global_step)
                model.train()

            if global_step % save_interval == 0:
                save_ckpt(global_step)
                run_eval(model, eval_loader, device, tb_writer, global_step, log,
                         plain_std, vib_std)
                run_plots(model, plot_items, device, tb_writer, global_step,
                          frame_rate=frame_rate, tb_prefix="f0_eval",
                          plain_std=plain_std, vib_std=vib_std)
                if train_plot_items:
                    run_plots(model, train_plot_items, device, tb_writer, global_step,
                              frame_rate=frame_rate, tb_prefix="f0_train",
                              plain_std=plain_std, vib_std=vib_std)

    pbar.close()
    save_ckpt(global_step)
    run_eval(model, eval_loader, device, tb_writer, global_step, log, plain_std, vib_std)
    run_plots(model, plot_items, device, tb_writer, global_step,
              frame_rate=frame_rate, tb_prefix="f0_eval",
              plain_std=plain_std, vib_std=vib_std)
    if train_plot_items:
        run_plots(model, train_plot_items, device, tb_writer, global_step,
                  frame_rate=frame_rate, tb_prefix="f0_train",
                  plain_std=plain_std, vib_std=vib_std)
    log("Training complete.")


def parse_args():
    p = argparse.ArgumentParser(description="Train the PrettyPitch F0 model.")
    p.add_argument("--config", required=True)
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--out_root", default="log")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_steps", type=int, default=None, help="override train.max_steps")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="dotted config override (repeatable), "
                        "e.g. --set model.conditioner=convnext --set model.n_steps=4")
    p.add_argument("--phonemes", default=None,
                   help="音素表の上書き（既定は config の lang: が指す言語パック）")
    p.add_argument("--nucleus", default=None,
                   help="核音素の上書き（既定は config の lang: が指す言語パック）")
    p.add_argument("--init_from", default=None, help="checkpoint to initialize G from")
    p.add_argument("--finetune", action="store_true",
                   help="with --init_from: G weights only, fresh D, step 0")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
