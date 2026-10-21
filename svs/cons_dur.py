"""子音長モデル（duration_model）を SVS の Phrase に繋ぐブリッジ。

`postprocess_durations` は子音の「自然長」を onset_nat / coda_nat で受け取る。ここは
楽譜由来の Phrase からその自然長をモデルで予測して組み立てる。特徴量の抽出は学習と
**同じ関数**（duration_model.features.extract_phrase_records）を使うので train/inference
でずれない。

**特徴量の意味も両側で揃えてある**: `_flatten` は音節の総時間（＝ノート長）を核に置き、
子音には 0 を置く。学習側は実測ラベルから `features.syllable_span_durs` が同じ形を作る。

音素表は `model.phoneset`（ckpt が記録したもの）を既定に使う。モジュール既定に落ちると
英語 ckpt が日本語の音素表で動き、全子音が無言でフォールバック長になる。

使い方:
    from svs.duration_postprocess import split_phrase_syllables, postprocess_durations
    from svs.cons_dur import load_cons_dur, predict_cons_naturals

    model, prof = load_cons_dur('checkpoints/cons_dur_3singer.pt')
    ph = split_phrase_syllables(phrase)                 # 多モーラノートをグループへ展開
    on_nat, cd_nat = predict_cons_naturals(ph, model, spk_id=2, style_id=0)
    seq = postprocess_durations(ph, prof, on_nat, cd_nat, rest_before=1e9)
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from duration_model import phoneset as _ps_mod
from duration_model.features import extract_phrase_records, records_to_batch
from duration_model.model import load_cons_dur_model
from duration_model.profile import ConsDurProfile, from_dict, load_profile

from .duration_postprocess import FALLBACK_CONS_DUR, Phrase, split_syllable

PAU = 'pau'


def load_cons_dur(ckpt_path, profile_path=None, device: str = 'cpu'):
    """子音長モデルと floor プロファイルを読む。→ (model, profile)

    profile_path 未指定なら ckpt と同名の JSON（duration_model/profiles/<stem>.json）を探し、
    無ければ組込み既定（子音 50ms・母音 75ms）にフォールバックする。
    """
    import json
    import torch
    ck = torch.load(str(ckpt_path), map_location=device)
    model = load_cons_dur_model(ck, device=device)       # 持続長モデルは ConsonantDurModel の 1 種
    prof: ConsDurProfile
    if profile_path is None:
        prof = load_profile(str(ckpt_path))              # duration_model/profiles/<stem>.json か組込み既定
    else:
        prof = from_dict(json.loads(Path(profile_path).read_text(encoding='utf-8')))
    return model, prof


def model_phoneset(model):
    """子音長モデルが学習された音素表を返す（無ければ None）。

    `load_cons_dur_model` が ckpt の config から復元して `model.phoneset` に付けている。
    推論側はこれを使うこと。モジュール既定（`phoneset.active()`）に頼ると、英語の ckpt を
    日本語の既定で動かして**無言でフォールバックする**（子音が全部 FALLBACK_CONS_DUR になる）。
    """
    return getattr(model, 'phoneset', None)


def _flatten(phrase: Phrase, phoneset=None):
    """Phrase → (音素列, 暫定持続長, ノート音高値, 各子音の戻し先).

    戻し先 = flat index → (note_idx, 'onset'|'coda'|'core', position)。

    'core' は核を持たないノート（子音だけのノート）の子音。従来はノート尺を等分するだけで
    モデルを呼んでいなかったが、自然長を予測して比例配分できるよう戻し先を持たせる。
    span は学習側 `features.syllable_span_durs` と同じく**先頭の音素**に置く
    （あちらも核が無いグループでは先頭に集約している）。
    """
    ps = phoneset or _ps_mod.active()
    phs: List[str] = [PAU]
    durs: List[float] = [float(phrase.leading_pau_sec)]
    vals: List[float] = [np.nan]
    where: dict = {}

    for j, note in enumerate(phrase.notes):
        onset, nucleus, coda = split_syllable(note.phonemes, ps)
        # 「音節の総時間を核に置き、子音は 0」。学習側（features.syllable_span_durs）と
        # 同じ意味になるので、prev/next_note_dur が両側で一致する。
        # ここで子音長を見積もる必要が無い（見積もると学習と違う値をモデルへ渡すことになる）。
        span = float(note.dur_sec)
        per = span / len(nucleus) if nucleus else span

        for k, c in enumerate(onset):
            where[len(phs)] = (j, 'onset', k)
            phs.append(c); durs.append(0.0); vals.append(np.nan)
        core_only = bool(nucleus) and not any(
            ps.is_nucleus(v) or ps.is_pau(v) for v in nucleus)
        if core_only:
            for k, c in enumerate(nucleus):
                where[len(phs)] = (j, 'core', k)
                phs.append(c); durs.append(span if k == 0 else 0.0); vals.append(np.nan)
        else:
            for v in nucleus:
                phs.append(v); durs.append(per)
                vals.append(np.nan if v == PAU else float(note.pitch_midi))
        if not nucleus:                       # 核を持たないノート（cl のみ等）は尺だけ確保
            phs.append(PAU); durs.append(span); vals.append(np.nan)
        for k, c in enumerate(coda):
            where[len(phs)] = (j, 'coda', k)
            phs.append(c); durs.append(0.0); vals.append(np.nan)
        if note.gap_after_sec > 0:
            phs.append(PAU); durs.append(float(note.gap_after_sec)); vals.append(np.nan)

    phs.append(PAU); durs.append(float(phrase.trailing_pau_sec)); vals.append(np.nan)
    return phs, np.asarray(durs, np.float64), np.asarray(vals, np.float64), where


def predict_cons_naturals(phrase: Phrase, model, *, spk_id: int = 0, style_id: int = 0,
                          device: str = 'cpu', phoneset=None, with_core: bool = False,
                          style_mix=None):
    """Phrase の各ノートについて (onset_nat, coda_nat) を予測する。

    style_mix=[n_styles] の混合比を渡すと、スタイル埋め込みをその比で混ぜる（連続スタイル。
    one-hot なら style_id と同じ結果）。

    with_core=True なら (onset_nat, coda_nat, core_nat) の 3 要素を返す。core_nat は
    核を持たないノート（子音だけのノート）の自然長で、`postprocess_durations(core_nat=...)`
    に渡すと尺を等分でなく自然長の比で配分できる。既定は False なので既存の呼び出しは不変。

    phrase は `split_phrase_syllables` を通した「1 ノート = `_syllabify` の 1 グループ」の
    形を想定（通さなくても動くが、多モーラノートの内部子音は予測されずフォールバックになる）。
    予測対象にならない子音（フレーズ末尾で入るノートが無い等）は FALLBACK_CONS_DUR。
    ただし言語パックが final_consonants: true（韓国語など）なら、句末・休みの前の終声も予測する。

    phoneset 未指定なら **ckpt が記録した音素表**（`model.phoneset`）を使う。モジュール
    既定に落ちると、英語 ckpt を日本語の既定で動かして全子音がフォールバックする。
    """
    ps = phoneset or model_phoneset(model)
    onset_nat: List[List[float]] = []
    coda_nat: List[List[float]] = []
    core_nat: List[List[float]] = []
    for note in phrase.notes:
        onset, nuc, coda = split_syllable(note.phonemes, ps)
        core_only = bool(nuc) and not any(ps.is_nucleus(v) or ps.is_pau(v) for v in nuc)
        onset_nat.append([FALLBACK_CONS_DUR] * len(onset))
        coda_nat.append([FALLBACK_CONS_DUR] * len(coda))
        core_nat.append([FALLBACK_CONS_DUR] * len(nuc) if core_only else [])
    out = (onset_nat, coda_nat, core_nat) if with_core else (onset_nat, coda_nat)
    if not phrase.notes:
        return out

    slots = {'onset': onset_nat, 'coda': coda_nat, 'core': core_nat}
    phs, durs, vals, where = _flatten(phrase, ps)
    recs = extract_phrase_records(phs, durs, vals, int(spk_id), int(style_id), phoneset=ps)
    recs = [r for r in recs if r['idx'] in where]
    if not recs:
        return out

    batch = records_to_batch(recs, device=device, phoneset=ps)
    if style_mix is not None:
        batch['style_mix'] = torch.as_tensor(np.asarray(style_mix, np.float32),
                                             device=device).expand(len(recs), -1)
    pred = model.predict(batch).cpu().numpy()
    for r, d in zip(recs, pred):
        j, slot, k = where[r['idx']]
        slots[slot][j][k] = float(d)
    return out
