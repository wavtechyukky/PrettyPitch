"""音素インベントリ（言語パック駆動・自己完結）。

1 言語 = `dict/<lang>.yaml` 1 枚。音素表・核音素・モーラ表はそこから **参照** され、
リストでは表せない設定（無声化母音の正規化写像、無音トークン、楽譜記号など）が
同じ YAML に載る。新しい言語を足すのにコードを書く必要は無い。

    ja = load('ja')          # dict/ja.yaml
    en = load('en')          # dict/en.yaml
    ja.is_nucleus('a')       # True
    en.is_nucleus('OW1')     # True（case_fold + strip_stress_digits）

**言語の選び方（優先順）**

1. 明示引数 — 音素に触る公開関数は `phoneset=` を受ける。これが最優先。
2. チェックポイント — 学習時に音素表と核音素表が ckpt へ書かれ、読み込み時に
   PhoneSet が再構築される。推論で言語を取り違えられない。
3. モジュール既定 — `active()`。ckpt を持たない CLI 入口のために `set_default()`。
   `from . import phoneset` と属性アクセスにするか、`phoneset=` 引数で受け渡すこと
   （import 時の値を束縛すると、言語を切り替えても届かなくなる）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

import yaml

_DICT_DIR = Path(__file__).resolve().parent.parent / 'dict'
DEFAULT_LANG = 'ja'


def _read_list(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            out.append(line)
    return out


@dataclass(frozen=True)
class PhoneSet:
    """1 言語分の音素インベントリと派生集合。イミュータブル。"""

    lang: str
    phonemes: tuple[str, ...]
    nuclei: frozenset[str]
    vowels: frozenset[str]
    vowel_normalize: Mapping[str, str]
    silence: frozenset[str]
    consonants: tuple[str, ...]
    # 前処理で生ラベル記号を pau へ書き換える写像（silence とは役割が違う）
    sil_map: Mapping[str, str] = field(default_factory=dict)
    case_fold: Optional[str] = None            # None | 'lower' | 'upper'
    strip_stress_digits: bool = False
    # 後ろに核の無い子音（句末・休みの前の終声）も子音長モデルの対象にするか。
    # 終声で句が終わる言語（韓国語など）で true。duration_model/features.py を参照
    final_consonants: bool = False
    score: Mapping = field(default_factory=dict)
    phonemes_path: Optional[Path] = None
    nucleus_path: Optional[Path] = None
    mora_path: Optional[Path] = None

    # ── 派生（dataclass なので __post_init__ で object.__setattr__）──────────
    def __post_init__(self):
        object.__setattr__(self, 'ph2id', {ph: i for i, ph in enumerate(self.phonemes)})
        object.__setattr__(self, 'id2ph', {i: ph for i, ph in enumerate(self.phonemes)})
        object.__setattr__(self, 'n_phonemes', len(self.phonemes))
        object.__setattr__(self, 'n_consonants', len(self.consonants))
        object.__setattr__(self, 'cons_set', frozenset(self.consonants))
        object.__setattr__(self, 'pau', self.phonemes[0])
        object.__setattr__(self, 'pau_id', 0)
        object.__setattr__(self, 'known_phonemes',
                           frozenset(self.phonemes) | self.silence)

    # ── 判定 ────────────────────────────────────────────────────────────────
    def fold(self, ph: str) -> str:
        """表記ゆれの吸収（大小文字・ストレス数字）。正規化の前段。"""
        s = str(ph)
        if self.strip_stress_digits:
            s = ''.join(c for c in s if not c.isdigit())
        if self.case_fold == 'lower':
            s = s.lower()
        elif self.case_fold == 'upper':
            s = s.upper()
        return s

    def normalize_ph(self, ph: str) -> str:
        """統一 id 空間へ正規化: 無音→pau, 無声化母音/グロッタル→有声母音。"""
        if ph in self.silence:
            return self.pau
        f = self.fold(ph)
        if f in self.silence:
            return self.pau
        return self.vowel_normalize.get(ph, self.vowel_normalize.get(f, f))

    def to_id(self, ph: str) -> int:
        return self.ph2id[self.normalize_ph(ph)]

    def is_pau(self, ph: str) -> bool:
        return self.normalize_ph(ph) == self.pau

    def is_nucleus(self, ph: str) -> bool:
        """核（ピッチ/尺を担う）か。言語に依らずこの 1 本の規則だけ。"""
        if self.is_pau(ph):
            return False
        return self.normalize_ph(ph) in self.nuclei

    def is_consonant(self, ph: str) -> bool:
        return ph in self.cons_set

    def digest(self) -> dict:
        """派生集合の中身を比較可能な形で返す（回帰テスト用）。"""
        return {
            'lang': self.lang,
            'phonemes': list(self.phonemes),
            'nuclei': sorted(self.nuclei),
            'vowels': sorted(self.vowels),
            'vowel_normalize': dict(sorted(self.vowel_normalize.items())),
            'silence': sorted(self.silence),
            'sil_map': dict(sorted(self.sil_map.items())),
            'consonants': list(self.consonants),
            'n_phonemes': self.n_phonemes,
            'n_consonants': self.n_consonants,
            'case_fold': self.case_fold,
            'strip_stress_digits': self.strip_stress_digits,
            'final_consonants': self.final_consonants,
        }


def _pack_path(lang: str) -> Path:
    p = Path(lang)
    if p.suffix in ('.yaml', '.yml') and p.exists():
        return p
    cand = _DICT_DIR / f'{lang}.yaml'
    if not cand.exists():
        avail = sorted(x.stem for x in _DICT_DIR.glob('*.yaml'))
        raise FileNotFoundError(
            f'言語パックが見つかりません: {cand}（利用可能: {avail}）')
    return cand


@lru_cache(maxsize=None)
def load(lang: str = DEFAULT_LANG,
         phonemes: Optional[str] = None,
         nucleus: Optional[str] = None) -> PhoneSet:
    """`dict/<lang>.yaml` から PhoneSet を構築する。

    phonemes / nucleus を渡すと、その項目だけパックの指定を上書きする
    （例: --lang ja --phonemes dict/<独自>.phonemes で、音素表だけ差し替える）。
    yaml の nucleus: はリスト直書きでもファイル名でもよい。引数の nucleus は
    ファイル名のみ（この関数は lru_cache 済みで、リストは hash できない）。
    """
    pack_path = _pack_path(lang)
    pack = yaml.safe_load(pack_path.read_text(encoding='utf-8')) or {}
    base = pack_path.parent

    def _resolve(v, override):
        if override:
            p = Path(override)
            # まず渡されたとおりに解釈する（--phonemes dict/<独自>.phonemes のような
            # cwd 相対の既存の使い方を壊さないため）。見つからなければ dict/ 相対として
            # 解釈し直す。ckpt には dict/ 相対で焼くので、配布先でもこちらで解決できる。
            return p if p.is_absolute() or p.exists() else (base / p)
        return (base / v) if v else None

    ph_path = _resolve(pack.get('phonemes'), phonemes)
    mora_path = _resolve(pack.get('mora'), None)
    # nucleus は yaml に **リストで直接** 書くのが既定（11〜15 語しかなく、差し替える
    # 動機もほぼ無いため、言語あたり1ファイル減らす）。外部データセット向けに
    # yaml 側でのファイル名指定も引き続き受ける。--nucleus での上書きはファイル名。
    nu_spec = nucleus if nucleus is not None else pack.get('nucleus')
    nu_path = None if isinstance(nu_spec, (list, tuple)) else _resolve(nu_spec, nucleus)
    if ph_path is None:
        raise ValueError(f'{pack_path}: phonemes が未指定です')

    phs = _read_list(ph_path)
    if not phs or phs[0] != 'pau':
        # 実際の仕様（id 0 = pad/無音）。assert だと python -O で消えるので例外にする。
        raise ValueError(f'{ph_path}: 音素表の id 0 は pau である必要があります')
    if len(set(phs)) != len(phs):
        dup = sorted({p for p in phs if phs.count(p) > 1})
        raise ValueError(f'{ph_path}: 音素が重複しています: {dup}')

    if isinstance(nu_spec, (list, tuple)):
        nuclei = frozenset(str(x) for x in nu_spec)
        nu_src = f'{pack_path} の nucleus:'
    else:
        nuclei = frozenset(_read_list(nu_path)) if nu_path else frozenset()
        nu_src = str(nu_path)
    missing = nuclei - set(phs)
    if missing:
        raise ValueError(f'{nu_src}: 音素表に無い核音素: {sorted(missing)}')

    vowel_norm = dict(pack.get('vowel_normalize') or {})
    vowels = frozenset(pack.get('vowels') or [])
    silence = frozenset(pack.get('silence') or ['pau'])

    non_consonant = (frozenset({'pau'}) | nuclei | frozenset(vowel_norm)
                     | frozenset(pack.get('non_consonant_extra') or []))
    consonants = tuple(ph for ph in phs if ph not in non_consonant)
    if not consonants:
        raise ValueError(f'{ph_path}: 音素表に子音が 1 つもありません（非子音集合を確認）')

    return PhoneSet(
        lang=pack_path.stem,
        phonemes=tuple(phs),
        nuclei=nuclei,
        vowels=vowels,
        vowel_normalize=vowel_norm,
        silence=silence,
        consonants=consonants,
        sil_map=dict(pack.get('sil_map') or {}),
        case_fold=pack.get('case_fold') or None,
        strip_stress_digits=bool(pack.get('strip_stress_digits', False)),
        final_consonants=bool(pack.get('final_consonants', False)),
        score=dict(pack.get('score') or {}),
        phonemes_path=ph_path,
        nucleus_path=nu_path,
        mora_path=mora_path,
    )


# ── モジュール既定 ──────────────────────────────────────────────────────────
_DEFAULT: PhoneSet = load(DEFAULT_LANG)


def active() -> PhoneSet:
    """現在の既定 PhoneSet。"""
    return _DEFAULT


def set_default(ps_or_lang) -> PhoneSet:
    """既定を差し替える（ckpt を持たない CLI 入口用）。モジュール定数も張り替える。"""
    global _DEFAULT
    _DEFAULT = ps_or_lang if isinstance(ps_or_lang, PhoneSet) else load(ps_or_lang)
    return _DEFAULT
