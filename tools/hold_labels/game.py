"""GAME（openvpi の音符抽出モデル、https://github.com/openvpi/GAME）を呼ぶ部分。

GAME の本体とモデルはこのリポジトリに入れない。取得の仕方は README.md。コードは MIT、モデル（GAME-1.0-*）は
CC BY-NC-SA 4.0。

使い方は GAME の align と同じ: 句の音声と、単語の境界を与える。単語は母音の頭で始まり、後ろの子音までを含む
（pau と句頭の子音は無声の単語で、続く無声の単語はまとめる）。単語の境界は動かさず、GAME は単語の中に
音符の境界を足すかだけを決める。D3PM の標本化は乱数を使うので、種を変えた n 行のバッチで 1 回に走らせる。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Sequence, Tuple

import numpy as np

NSTEPS, THRESHOLD, RADIUS, PRESENCE = 8, 0.2, 2, 0.2     # GAME の infer.py align の既定（t0 = 0、8 段）
PAD_SEC = 0.1                                            # 句の後ろに足す無音（メルの最後のフレームを確保する）


def load(game_root, model_dir, device: str = "cpu"):
    """GAME のモデルを読む。→ (model, lang_map, GAME の関数の入れ物)。
    game_root: GAME を clone したフォルダ。model_dir: config.yaml・lang_map.json・model.pt のあるフォルダ。"""
    import torch
    root = Path(game_root).resolve()
    if not (root / "inference" / "me_infer.py").exists():
        raise FileNotFoundError(f"{root} に GAME が見つかりません（tools/hold_labels/README.md の取得の手順を参照）")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lib.config.io import load_raw_config
    from lib.config.schema import InferenceConfig, ModelConfig
    from inference.me_infer import SegmentationEstimationInferenceModel
    from modules.d3pm import d3pm_time_schedule, remove_mutable_boundaries
    from modules.decoding import decode_soft_boundaries
    from modules.functional import boundaries_to_regions, format_boundaries

    mdir = Path(model_dir)
    cfg = load_raw_config(mdir / "config.yaml", inherit=False)
    mc = ModelConfig.model_validate(cfg["model"], scope=0)
    ic = InferenceConfig.model_validate(cfg["inference"], scope=0)
    mc.check(scope_mask=0)
    ic.check(scope_mask=0)
    model = SegmentationEstimationInferenceModel(model_config=mc, inference_config=ic)
    ck = torch.load(mdir / "model.pt", map_location="cpu")
    sd = ck.get("state_dict", {})
    sd.update(ck.get("ema_state_dict") or {})
    model.load_state_dict(sd, strict=True)
    fns = SimpleNamespace(d3pm_time_schedule=d3pm_time_schedule, remove_mutable_boundaries=remove_mutable_boundaries,
                          decode_soft_boundaries=decode_soft_boundaries, boundaries_to_regions=boundaries_to_regions,
                          format_boundaries=format_boundaries)
    return model.eval().to(device), json.load(open(mdir / "lang_map.json", encoding="utf-8")), fns


def words(phs: Sequence[str], durs: Sequence[float], is_nucleus: Callable[[str], bool]) -> Tuple[List[float], List[bool]]:
    """音素の列 → GAME に渡す単語（長さ（秒）の list, 有声か の list）。"""
    wd, wv = [], []
    for p, d in zip(phs, durs):
        voiced_start = is_nucleus(p)
        if voiced_start or (p == "pau" and (not wv or wv[-1])) or not wd:
            wd.append(float(d))
            wv.append(voiced_start)
        else:                       # 子音は前の単語に付ける（無声の単語のあとの子音・pau は無声の単語にまとめる）
            wd[-1] += float(d)
    return wd, wv


def added_boundaries(model, fns, wav: np.ndarray, word_dur: Sequence[float], lang: int, n_seeds: int) -> List[np.ndarray]:
    """GAME が単語の中に足した音符の境界の時刻（秒）を、種ごとに返す。エンコーダは 1 回、D3PM の標本化は
    種の数の行のバッチで（行ごとに乱数が違う。torch の種は 0 に固定するので、同じ入力なら同じ結果）。"""
    import torch
    with torch.no_grad():
        dev = next(model.parameters()).device
        torch.manual_seed(0)
        w = torch.from_numpy(np.asarray(wav, np.float32))[None].to(dev)
        kd = torch.tensor([list(word_dur)], dtype=torch.float32, device=dev)
        x_seg, x_est, mask = model.forward_encoder(w, kd.sum(1))
        S, T = n_seeds, x_seg.size(1)
        known = (fns.format_boundaries(kd, T, model.timestep) & mask).repeat(S, 1)
        x_seg, mask = x_seg.repeat(S, 1, 1), mask.repeat(S, 1)
        language = torch.full((S,), int(lang), dtype=torch.long, device=dev)
        b = known
        for i in range(NSTEPS):
            t = torch.full((S,), i / NSTEPS, dtype=torch.float32, device=dev)
            noisy = fns.remove_mutable_boundaries(b, known, p=fns.d3pm_time_schedule(t))
            logits, _ = model.model.forward_segmentation(
                x_seg, noise=fns.boundaries_to_regions(noisy, mask=mask), t=t, language=language, mask=mask)
            b = fns.decode_soft_boundaries(logits.sigmoid(), barriers=known, mask=mask,
                                           threshold=THRESHOLD, radius=RADIUS)
        b, known = b.cpu(), known.cpu()
        return [(b[s] & ~known[s]).nonzero()[:, 0].numpy() * model.timestep for s in range(S)]
