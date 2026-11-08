"""英語の G2P: 歌詞 → en_full の表記の音素（dict/en_full.phonemes）。

単語: 発音辞書 dict/en_cmudict.txt（tools/build_en_cmudict.py が CMUdict から作る。強勢の無い AH0 は ax）で引く。
大文字・小文字は区別せず、前後の句読点（, . ! ? " など）は捨て、’ は ' にする。ハイフンでつないだ語（well-known）は
語ごとに引く。1 つの歌詞に何語あってもよい（I love → ay l ah v）。辞書に無い語があれば読まない（None）。

音素: 次の書き方は単語ではなく音素として読む（単語と同じ綴りの音素 oh・ah なども、これで書き分ける）
  [hh eh l ow]        角かっこの中（空白区切り）。hello[hh eh l ow] のように前に単語を書いてもよい（表示用。読まない）
  hh eh l ow          空白で区切った 2 つ以上の記号が、すべて音素
  s・t・dx            子音の音素 1 つ（語としての読み「エス」より、終わりの子音だけの音符を優先する）
音素は大文字・強勢の数字付き（HH AH0 L OW1）でもよい（AH0 は ax、ほかは数字を落とす。辞書と同じ写し方）。

音符への配り方（続く記号の音符へ音節を前から順に配る。+ は次の音節、- と +~ は伸ばし、+3 は 3 番目の音素から）は
svs/score_io.py の _apply_g2p_en。
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

_DATA = Path(__file__).resolve().parents[1] / "dict" / "en_cmudict.txt"
_PUNCT = ".,!?;:\"“”()[]{}…«»¡¿"
_BRACKET = re.compile(r"^\s*([^\[\]]*)\[([^\[\]]*)\]\s*$")


@lru_cache(maxsize=1)
def load_dict() -> Dict[str, List[str]]:
    """単語（小文字）→ 音素の列。"""
    out: Dict[str, List[str]] = {}
    for line in _DATA.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        w, *phs = line.split()
        out[w] = phs
    return out


def _norm_word(tok: str) -> Optional[str]:
    """歌詞の 1 語 → 辞書の見出し（無ければ None）。前後の句読点を捨て、それでも無ければ前後の ' も捨てる（'cause は残す）。"""
    d = load_dict()
    t = tok.replace("’", "'").replace("‘", "'").lower().strip(_PUNCT)
    if t in d:
        return t
    t = t.strip("'")
    return t if t in d else None


def lyric_words(lyric: str) -> Optional[List[str]]:
    """歌詞 → 辞書の見出しの列。語が 1 つも無いか、辞書に無い語があれば None。"""
    words: List[str] = []
    for tok in lyric.split():
        for part in tok.split("-"):
            if not part.strip(_PUNCT + "'’‘"):
                continue                          # ハイフンや句読点だけの切れ端
            w = _norm_word(part)
            if w is None:
                return None
            words.append(w)
    return words or None


def words_to_phonemes(words: List[str]) -> List[str]:
    d = load_dict()
    return [p for w in words for p in d[w]]


def _fold(tok: str, phoneset) -> str:
    return "ax" if tok.upper() == "AH0" else phoneset.fold(tok)


def explicit_phonemes(lyric: str, phoneset) -> Optional[List[str]]:
    """歌詞が音素の書き方（上の 3 つ）なら音素の列、そうでなければ None。角かっこの中に音素表に無い記号があれば KeyError。"""
    known = phoneset.ph2id
    m = _BRACKET.match(lyric)
    if m:
        toks = [_fold(t, phoneset) for t in m.group(2).split()]
        bad = [t for t in toks if t not in known]
        if not toks or bad:
            raise KeyError(f"歌詞 {lyric!r} の [ ] の中に音素表に無い記号があります: {bad or '（空）'}")
        return toks
    toks = [_fold(t, phoneset) for t in lyric.split()]
    if len(toks) >= 2 and all(t in known for t in toks):
        return toks
    if len(toks) == 1 and toks[0] in phoneset.cons_set:
        return toks
    return None


def lyric_phonemes(lyric: str, phoneset) -> Optional[List[str]]:
    """歌詞 → 音素の列（音素の書き方なら音素、そうでなければ辞書）。どちらでも読めなければ None。"""
    ph = explicit_phonemes(lyric, phoneset)
    if ph is not None:
        return ph
    words = lyric_words(lyric)
    return words_to_phonemes(words) if words else None
