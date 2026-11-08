"""伸ばしのラベルの規則: GAME の結果から、どの母音のどこで分けるかを決め、lab とシャードに書く。

torch も GAME も使わない純粋な関数だけを置く（CI で試せるように）。GAME を呼ぶ部分は game.py、
コマンドは split.py。

母音を分ける条件（どれも満たしたものだけ分ける。既定値は Params）:
  1. 候補: 母音の有声の F0 を前後 2 つに分けたときの中央値の差の最大が min_diff セント以上で、
     母音が min_len 秒より長い（短い母音は、二重母音や移りの動きを音符と取り違えやすい）
  2. GAME の境界: 種を変えて何回か走らせ、cluster_sec 以内の境界を 1 つにまとめ、過半数の回に出たものだけ
  3. 端の余白: 母音の端と、前に採った境界から margin 秒以上離れている（入りのしゃくり・語尾の落ちを音符にしない）
  4. 段差: 境界の前後 STEP_WIN 秒（境界の両側 STEP_GAP 秒は除く）の有声の F0 の中央値の差が min_step セント以上
     （音高でなく音色の動きで分けたものを捨てる）
人の判断（overrides）がある母音は、GAME の代わりにそれを使う（長さの条件だけはかける）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

STEP_WIN, STEP_GAP = 0.10, 0.02     # 段差を測るとき、境界の前後を見る長さ・境界の両側で見ない長さ（秒）
MIN_SIDE_SEC = 0.04                 # 候補の音高の差を測るとき、2 つに分けた片側の最小の長さ（秒）


@dataclass(frozen=True)
class Params:
    margin: float = 0.08            # 端の余白（秒）
    min_diff: float = 30.0          # 候補の音高の差の下限（セント）
    min_len: float = 0.2            # 分ける母音の長さの下限（秒。これ以下は分けない）
    min_step: float = 80.0          # 境界の段差の下限（セント。0 で検査しない）
    cluster_sec: float = 0.03       # 種ごとの境界をまとめる距離（秒）


def best_split_diff(cents: Sequence[float], frame_rate: float, min_side_sec: float = MIN_SIDE_SEC) -> float:
    """有声フレームの F0（セント）を前後 2 つに分けたときの、中央値の差の最大（各側 min_side_sec 以上）。"""
    c = np.asarray(cents, float)
    L, mn = len(c), int(np.ceil(min_side_sec * frame_rate))
    top = 0.0
    for k in range(mn, L - mn + 1):
        top = max(top, abs(float(np.median(c[k:])) - float(np.median(c[:k]))))
    return top


def clusters(seeds: Sequence[Sequence[float]], cluster_sec: float = 0.03) -> List[Tuple[float, int]]:
    """種ごとの境界の時刻 → [(時刻の中央値, 何回の種に出たか)]（cluster_sec 以内を 1 つにまとめる）。"""
    pts = sorted((float(t), si) for si, ts in enumerate(seeds) for t in ts)
    out: List[list] = []
    for t, si in pts:
        if out and t - out[-1][-1][0] <= cluster_sec:
            out[-1].append((t, si))
        else:
            out.append([(t, si)])
    return [(float(np.median([t for t, _ in c])), len({si for _, si in c})) for c in out]


def keep_apart(ts: Sequence[float], l0: float, l1: float, margin: float) -> List[float]:
    """母音の端と、前に採った境界から margin 以上離れたものだけ残す。"""
    out, prev = [], l0
    for t in sorted(ts):
        if t - prev >= margin and l1 - t >= margin:
            out.append(t)
            prev = t
    return out


def step_at(cents: np.ndarray, voiced: np.ndarray, frame_rate: float, t: float, lo: float, hi: float) -> Optional[float]:
    """境界 t（句の中の秒）の前後の有声の F0 の中央値の差（セント、後 − 前）。lo・hi はこの境界が属する区間の端
    （母音の端か隣の境界）。片側の有声のフレームが 2 つ未満なら None。"""
    f = lambda s: max(0, int(round(s * frame_rate)))  # noqa: E731
    a0, a1 = f(max(lo, t - STEP_GAP - STEP_WIN)), f(t - STEP_GAP)
    b0, b1 = f(t + STEP_GAP), f(min(hi, t + STEP_GAP + STEP_WIN))
    pre = cents[a0:max(a0, a1)][voiced[a0:max(a0, a1)]]
    post = cents[b0:max(b0, b1)][voiced[b0:max(b0, b1)]]
    if len(pre) < 2 or len(post) < 2:
        return None
    return float(np.median(post) - np.median(pre))


def keep_steps(ts: Sequence[float], off: float, v0: float, v1: float, cents: np.ndarray, voiced: np.ndarray,
               frame_rate: float, min_step: float) -> List[float]:
    """境界（曲の中の秒）のうち、前後に min_step セント以上の段差があるものだけ残す。off は曲の時刻 − 句の時刻、
    v0・v1 は母音の句の中の端。前から順に、残した境界を次の境界の区間の端にする。"""
    out, prev = [], v0
    for k, t in enumerate(ts):
        tp = t - off
        nxt = ts[k + 1] - off if k + 1 < len(ts) else v1
        s = step_at(cents, voiced, frame_rate, tp, prev, nxt)
        if s is not None and abs(s) >= min_step:
            out.append(t)
            prev = tp
    return out


def choose_splits(v: dict, n_seeds: int, params: Params, f0: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                  frame_rate: Optional[float] = None, override: Optional[dict] = None) -> Tuple[List[float], str, int]:
    """1 つの母音（run の記録の vowels の要素）の境界（曲の中の秒）を決める。
    → (境界の list, 'game' か 'override', 段差の検査で捨てた境界の数)。f0 は句の (セント, 有声か) で、段差の検査に使う。
    override は人の判断 {"decision": "split"|"keep", "splits_sec": [句の中の秒, ...]}。"""
    off = v["l0"] - v["v0"]                      # 句の中の時刻 → 曲の中の時刻（母音の頭で合わせる）
    long_enough = v["l1"] - v["l0"] > params.min_len
    if override is not None:
        if override.get("decision") == "split" and long_enough:
            return sorted(t + off for t in override.get("splits_sec", [])), "override", 0
        return [], "override", 0
    if not (v["diff"] >= params.min_diff and long_enough):
        return [], "game", 0
    major = [t + off for t, n in clusters(v["seeds"], params.cluster_sec) if n * 2 > n_seeds]
    ts = keep_apart(major, v["l0"], v["l1"], params.margin)
    dropped = 0
    if ts and params.min_step > 0:
        if f0 is None or frame_rate is None:
            raise ValueError("段差の検査には句の F0 が要ります（min_step = 0 で検査しない）")
        kept = keep_steps(ts, off, v["v0"], v["v1"], f0[0], f0[1], frame_rate, params.min_step)
        dropped = len(ts) - len(kept)
        ts = kept
    return ts, "game", dropped


def _fmt_time(t: float, like: str, unit: str) -> str:
    if unit == "100ns":
        return str(int(round(t * 1e7)))
    dec = len(like.split(".", 1)[1]) if "." in like else 7
    return f"{t:.{dec}f}"


def split_lab_lines(lines: Sequence[str], splits_by_row: Dict[int, Sequence[float]], unit: str,
                    hold: str = "-") -> List[str]:
    """lab の行（文字列）に境界を入れる。行の番号は、3 つ以上の欄がある行だけを数えた番号（空行は落とし、欄の
    足りない行はそのまま残す）。分けた行は、最初の 1 つが元の記号、2 つ目からが hold になる（例 ay → ay -）。
    境界は曲の中の秒で、lab の単位（100ns か sec）で書く。"""
    out, i = [], 0
    for ln in lines:
        parts = ln.split(maxsplit=2)
        if len(parts) < 3:
            if ln.strip():
                out.append(ln.strip())
            continue
        a, b, label = parts
        label = label.strip()
        ts = sorted(splits_by_row.get(i, ()))
        if not ts:
            out.append(f"{a} {b} {label}")
        else:
            bounds = [a] + [_fmt_time(t, a, unit) for t in ts] + [b]
            for k, (x0, x1) in enumerate(zip(bounds[:-1], bounds[1:])):
                out.append(f"{x0} {x1} {label if k == 0 else hold}")
        i += 1
    return out


def insert_holds(ph_ids: np.ndarray, dur_sec: np.ndarray, frame_rate: float,
                 cuts: Dict[int, Sequence[int]], hold_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """句の音素列に伸ばしの記号を差し込む。cuts = {音素の番号: 句の頭からのフレームの境界の list}。
    分けた音素は最初の 1 つが元の音素、2 つ目からが hold_id になる。分けない音素の長さはそのまま写す
    （F0 と v/uv はフレームの列なので変わらない）。境界は音素の中に収め、両側に 1 フレーム以上残す。"""
    ids = np.asarray(ph_ids)
    dur = np.asarray(dur_sec, np.float32)
    bnd = np.concatenate([[0], np.round(np.cumsum(dur.astype(np.float64)) * frame_rate).astype(int)])
    new_ids, new_dur = [], []
    for j in range(len(ids)):
        a, b = int(bnd[j]), int(bnd[j + 1])
        cs = sorted({int(c) for c in cuts.get(j, ()) if a < int(c) < b})
        if not cs:
            new_ids.append(int(ids[j]))
            new_dur.append(dur[j])
            continue
        edges = [a] + cs + [b]
        for k, (x0, x1) in enumerate(zip(edges[:-1], edges[1:])):
            new_ids.append(int(ids[j]) if k == 0 else int(hold_id))
            new_dur.append(np.float32((x1 - x0) / frame_rate))
    return np.asarray(new_ids, ids.dtype), np.asarray(new_dur, np.float32)
