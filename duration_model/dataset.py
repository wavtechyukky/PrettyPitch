"""ConsonantDurDataset — score-free 版。

slab/ust/xml は使わない。PrettyPitch と同じ preprocess の F0 シャード
（phoneme_ids / dur_sec / f0_interp / uv）を PrettyPitchF0Dataset 経由で読み、
dataloader が **ビブラート除去後の F0 代表値**（devib 2パス）でピッチを決める。

各サンプル（子音 1 件）:
    cons_id / prev_ph_id / next_ph_id  : 対象子音・直前・直後の音素（統一 id・区別なし）
    prev_note_dur / next_note_dur      : 直前/直後ノートの長さ [sec]（音素アライメント由来）
    pitch_midi / pitch_diff            : 直後ノートの F0 代表値（MIDI）とその差
    prev_note_exists                   : 0=フレーズ先頭, 1=通常
    target_dur                         : 子音の持続長 [sec]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .features import extract_phrase_records as _extract_phrase
from . import phoneset as _ps_mod

# PrettyPitch の F0 データローダを再利用（同じシャード・同じ代表値ロジック）
_PP_DIR = Path(__file__).resolve().parent.parent
if str(_PP_DIR) not in sys.path:
    sys.path.insert(0, str(_PP_DIR))
from dataset import PrettyPitchF0Dataset   # noqa: E402  (PrettyPitch/dataset.py)


class ConsonantDurDataset(Dataset):
    """F0 シャードから子音長レコードを抽出（score-free）。"""

    def __init__(self, dirs, split: str = 'train', eval_songs: int = 2,
                 min_sec: float = 0.5, seed: int = 42, nucleus_names=None,
                 pitch_cfg: dict | None = None, decomp_cfg: dict | None = None,
                 spk_map: dict | None = None, style_map: dict | None = None,
                 phoneset=None, eval_dbs=None):
        self.ps = phoneset or _ps_mod.active()
        self._f0 = PrettyPitchF0Dataset(
            dirs, split=split, eval_songs=eval_songs, eval_dbs=eval_dbs, min_sec=min_sec, seed=seed,
            pau_aug=False, nucleus_names=nucleus_names,
            pitch_cfg=pitch_cfg, decomp_cfg=decomp_cfg)
        id2ph = {v: k for k, v in self._f0.meta['phoneme_vocab'].items()}

        self.records: list[dict] = []
        for i in range(len(self._f0)):
            ph_ids, dur_sec, values = self._f0.phrase_pitch(i)
            ph_strs = [id2ph.get(int(p), 'pau') for p in ph_ids]
            dbname = os.path.basename(os.path.normpath(self._f0.files[i][2]))
            spk_id = int(spk_map.get(dbname, 0)) if spk_map else 0
            style_id = int(style_map.get(dbname, 0)) if style_map else 0
            self.records.extend(_extract_phrase(ph_strs, dur_sec, values, spk_id, style_id))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.records[idx]
        return {
            'cons_id':          torch.tensor(self.ps.to_id(r['cons']),    dtype=torch.long),
            'prev_ph_id':       torch.tensor(self.ps.to_id(r['prev_ph']), dtype=torch.long),
            'next_ph_id':       torch.tensor(self.ps.to_id(r['next_ph']), dtype=torch.long),
            'prev2_ph_id':      torch.tensor(self.ps.to_id(r.get('prev2_ph', 'pau')), dtype=torch.long),
            'next2_ph_id':      torch.tensor(self.ps.to_id(r.get('next2_ph', 'pau')), dtype=torch.long),
            'prev_note_dur':    torch.tensor(r['prev_note_dur'],      dtype=torch.float32),
            'next_note_dur':    torch.tensor(r['next_note_dur'],      dtype=torch.float32),
            'pitch_midi':       torch.tensor(r['pitch_midi'],         dtype=torch.long),
            'pitch_diff':       torch.tensor(r['pitch_diff'],         dtype=torch.float32),
            'prev_note_exists': torch.tensor(r['prev_note_exists'],   dtype=torch.long),
            'cluster_len':      torch.tensor(r.get('cluster_len', 1),  dtype=torch.long),
            'target_dur':       torch.tensor(r['target_dur'],         dtype=torch.float32),
            'spk_id':           torch.tensor(r['spk_id'],             dtype=torch.long),
            'style_id':         torch.tensor(r['style_id'],           dtype=torch.long),
        }

    @property
    def consonant_labels(self) -> list[str]:
        return [r['cons'] for r in self.records]
