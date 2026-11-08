"""子音長モデルの特徴量抽出（学習と推論で共有）。

学習は F0 シャード由来の音素列・持続長・代表値から、推論は楽譜由来の音素列・
暫定持続長・ノート音高から、**同じ関数**でレコードを作る。ここを共有しないと
train/inference で特徴量がずれる。
"""
from __future__ import annotations

import numpy as np
import torch

from . import phoneset as _ps_mod

_BATCH_KEYS_LONG = ('cons_id', 'prev_ph_id', 'next_ph_id', 'prev2_ph_id', 'next2_ph_id',
                    'pitch_midi', 'prev_note_exists', 'cluster_len', 'spk_id', 'style_id')
_BATCH_KEYS_FLOAT = ('prev_note_dur', 'next_note_dur', 'pitch_diff')


def cluster_lengths(ph_strs: list[str], phoneset=None) -> np.ndarray:
    """音素ごとに「自分が属する子音連結の長さ」を返す（子音以外は 0）。

    連結 = 音節核の片側（onset 側 / coda 側）に連続して並ぶ子音の本数。**フレーズ全体の
    音素数とは無関係**で、音節ごと・核の片側ごとに決まる。例えば "supposed to" が
    `z d t uw` という音節になったとき、`z` `d` `t` はいずれも 3 を受け取る。

    前後 1 音素しか見ない MLP は、この値を原理的に表現できない（`next_ph` は「次も子音」と
    しか言えず、残り 1 個か 3 個かを区別できない）。

    分解は `_syllabify` に任せるので、学習と推論で同じ値になる。DiffSinger の持続長 ONNX が
    受け取る `word_div`（母音から次の母音までの音素数）と同じ量で、ホスト側で音素列から
    計算できる＝推論時に余分な情報を要求しない。
    """
    from svs.duration_postprocess import _syllabify      # 遅延 import（循環回避）

    ps = phoneset or _ps_mod.active()
    out = np.zeros(len(ph_strs), dtype=np.int64)
    i = 0
    while i < len(ph_strs):
        if ps.is_pau(ph_strs[i]):
            i += 1
            continue
        j = i
        while j < len(ph_strs) and not ps.is_pau(ph_strs[j]):
            j += 1
        pos = i
        for syl in _syllabify(ph_strs[i:j], ps):
            # 核の判定は `is_nucleus` を使う。`split_syllable` と同じ述語にするため。
            nuc = next((k for k, p in enumerate(syl) if ps.is_nucleus(p)), None)
            if nuc is None:
                out[pos:pos + len(syl)] = len(syl)          # 核を持たないグループ（子音だけのノート等）
            else:
                out[pos:pos + nuc] = nuc                                   # onset 側
                out[pos + nuc + 1:pos + len(syl)] = len(syl) - nuc - 1     # coda 側
            pos += len(syl)
        i = j
    return out


def syllable_span_durs(ph_strs: list[str], dur_sec: np.ndarray, phoneset=None) -> np.ndarray:
    """音素ごとの実測長 → 「音節の総時間を核に集め、子音は 0」にした配列。

    `prev_note_dur` / `next_note_dur` の元になる値を、**学習と推論で同じ意味**にするための
    変換。学習時は lab の実測長からここで音節スパンを作り、推論時は楽譜のノート長を
    そのまま核に置く（`svs/cons_dur._flatten`）。どちらも「この音節に与えられた総時間」。

    無音（pau 等）は音節に属さないのでその場に残す。
    """
    from svs.duration_postprocess import _syllabify      # 遅延 import（循環回避）

    ps = phoneset or _ps_mod.active()
    out = np.zeros(len(ph_strs), dtype=np.float64)
    i = 0
    while i < len(ph_strs):
        if ps.is_pau(ph_strs[i]):
            out[i] = float(dur_sec[i])
            i += 1
            continue
        j = i
        while j < len(ph_strs) and not ps.is_pau(ph_strs[j]):
            j += 1
        seg = ph_strs[i:j]
        pos = i
        for syl in _syllabify(seg, ps):
            span = float(np.sum(dur_sec[pos:pos + len(syl)]))
            nuc = next((k for k, p in enumerate(syl) if ps.is_nucleus(p)), None)
            out[pos + nuc if nuc is not None else pos] = span   # 核へ集約、他は 0 のまま
            pos += len(syl)
        i = j
    return out


def extract_phrase_records(ph_strs: list[str], dur_sec: np.ndarray, values: np.ndarray,
                    spk_id: int = 0, style_id: int = 0, phoneset=None,
                    span_dur: np.ndarray | None = None) -> list[dict]:
    """1 フレーズの音素列 + 代表値から子音レコードを作る。

    span_dur: prev/next_note_dur の算出に使う配列。未指定なら dur_sec から
        `syllable_span_durs` で作る（＝核に音節の総時間、子音は 0）。
        target_dur は常に dur_sec[i]（実測の子音長）から取る。
    """
    ps = phoneset or _ps_mod.active()
    if span_dur is None:
        span_dur = syllable_span_durs(ph_strs, dur_sec, ps)
    clu = cluster_lengths(ph_strs, ps)
    recs: list[dict] = []
    n = len(ph_strs)
    for i in range(n):
        ph = ph_strs[i]
        if ph not in ps.cons_set:
            continue
        prev_ph = ph_strs[i - 1] if i > 0 else 'pau'
        next_ph = ph_strs[i + 1] if i + 1 < n else 'pau'
        # ±2 の音素。クラスタの 3 番目が何かは ±1 では原理的に見えない
        # （`s t r` の `s` から `r` は届かない）。範囲外は pau で埋める。
        prev2_ph = ph_strs[i - 2] if i > 1 else 'pau'
        next2_ph = ph_strs[i + 2] if i + 2 < n else 'pau'

        # 直前ノート（核=母音的 まで遡る。無音で打ち切り）
        j = None
        for jj in range(i - 1, -1, -1):
            if ps.is_pau(ph_strs[jj]):
                break
            if ps.is_nucleus(ph_strs[jj]):
                j = jj
                break
        prev_note_exists = 1 if j is not None else 0
        if j is not None:
            # 前の音節の総時間。span_dur では核に集約され子音は 0 なので、
            # 範囲を足しても「前音節の尺」そのものになる（自分の正解は混ざらない）。
            prev_note_dur = float(span_dur[j:i + 1].sum())
            prev_val = float(values[j])
        else:
            prev_note_dur = 0.0
            prev_val = np.nan

        # 直後ノート（次の核まで）
        k = None
        for kk in range(i + 1, n):
            if ps.is_pau(ph_strs[kk]):
                break
            if ps.is_nucleus(ph_strs[kk]):
                k = kk
                break
        if k is None:
            # 後ろに核の無い子音（句末や休みの前）。既定では入るべきノートが無いので扱わない。
            # 言語パックが final_consonants: true のとき（終声で句が終わる韓国語など）は、前の
            # ノートの終声として扱う: 音高は前のノートの代表値、next_note_dur = 0（＝後ろに
            # ノートが無い。prev_note_dur = 0 が「前にノートが無い」を表すのと同じ約束）。
            if not (ps.final_consonants and j is not None):
                continue  # 末尾の子音（入るべきノートが無い）
            next_note_dur = 0.0
            pit = prev_val if np.isfinite(prev_val) else 60.0
            prev_val = pit
        else:
            # 次の音節の総時間。span_dur は核に集約済みなので trailing 子音を足す必要が無い。
            next_note_dur = float(span_dur[k])

            # ピッチ = 直後ノートの F0 代表値（MIDI）。NaN は近傍/中央にフォールバック。
            pit = float(values[k])
            if not np.isfinite(pit):
                pit = prev_val if np.isfinite(prev_val) else 60.0
            if not np.isfinite(prev_val):
                prev_val = pit

        recs.append({
            'idx':              i,          # フレーズ内の音素位置（推論で結果を戻すのに使う）
            'cons':             ph,
            'prev_ph':          prev_ph,
            'next_ph':          next_ph,
            'prev2_ph':         prev2_ph,
            'next2_ph':         next2_ph,
            'prev_note_dur':    prev_note_dur,
            'next_note_dur':    next_note_dur,
            'pitch_midi':       int(round(min(max(pit, 0.0), 127.0))),
            'pitch_diff':       float(pit - prev_val),
            'prev_note_exists': prev_note_exists,
            'cluster_len':      int(clu[i]),
            'target_dur':       float(dur_sec[i]),
            'spk_id':           spk_id,
            'style_id':         style_id,
        })
    return recs


def records_to_batch(records: list[dict], device='cpu', phoneset=None) -> dict[str, torch.Tensor]:
    """レコード列 → model.predict 用のバッチ（ConsonantDurDataset.__getitem__ と同じ並び）。"""
    ps = phoneset or _ps_mod.active()
    dev = torch.device(device)
    src = {
        'cons_id':          [ps.to_id(r['cons']) for r in records],
        'prev_ph_id':       [ps.to_id(r['prev_ph']) for r in records],
        'next_ph_id':       [ps.to_id(r['next_ph']) for r in records],
        'prev2_ph_id':      [ps.to_id(r.get('prev2_ph', 'pau')) for r in records],
        'next2_ph_id':      [ps.to_id(r.get('next2_ph', 'pau')) for r in records],
        'pitch_midi':       [r['pitch_midi'] for r in records],
        'prev_note_exists': [r['prev_note_exists'] for r in records],
        'cluster_len':      [r.get('cluster_len', 1) for r in records],
        'spk_id':           [r['spk_id'] for r in records],
        'style_id':         [r['style_id'] for r in records],
        'prev_note_dur':    [r['prev_note_dur'] for r in records],
        'next_note_dur':    [r['next_note_dur'] for r in records],
        'pitch_diff':       [r['pitch_diff'] for r in records],
    }
    b = {k: torch.tensor(src[k], dtype=torch.long, device=dev) for k in _BATCH_KEYS_LONG}
    b.update({k: torch.tensor(src[k], dtype=torch.float32, device=dev) for k in _BATCH_KEYS_FLOAT})
    return b
