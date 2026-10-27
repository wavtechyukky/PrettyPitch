"""中国語（普通話）の G2P: 漢字・拼音 → ACV 表記の音素（dict/zh_acv.phonemes）か Opencpop 表記の音素（dict/zh_opencpop.phonemes）。

漢字 → 拼音: dict/zh_pinyin.json（tools/build_zh_pinyin.py が pypinyin のデータから作る表）。1 字ずつの既定の読みと、
多音字の読みが既定と違う熟語（长大 zhang da、音乐 yin yue など）の表を持つ。続けて歌う漢字の並びを、熟語の表の
左からの最長一致で読み、当たらない字は既定の読みにする。
拼音 → 音素: 声母 + 韻母の規則（ACV の表記、https://github.com/Archivoice/AV-diffsinger-Chinese-support）。
  介音は主母音とひとまとまり（xiao = x iA U、kuai = k ua ii、juan = j yE N）、ゼロ声母の y w は頭子音（ya = y a）、
  yue・yuan は rr（rr E、rr E N）、yi wu yu yin ying yun は頭子音なし。声調は読まない（数字・声調記号は捨てる）。

歌詞 1 つに漢字と拼音を混ぜてもよい（「我ai你」）。拼音は小文字で、空白・' で区切るか、続けて書けば音節の最長一致で切る。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_DATA = Path(__file__).resolve().parents[1] / "dict" / "zh_pinyin.json"

# 声母（長いものを先に）
_INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s")
# 声母のある音節の韻母 → 主母音と韻尾
_FINALS: Dict[str, List[str]] = {
    "a": ["a"], "o": ["O"], "e": ["e"], "ai": ["a", "ii"], "ei": ["E", "ii"], "ao": ["A", "U"], "ou": ["O", "U"],
    "an": ["a", "N"], "en": ["e", "N"], "ang": ["A", "ng"], "eng": ["e", "ng"], "ong": ["O", "ng"],
    "i": ["i"], "ia": ["ia"], "io": ["iO"], "ie": ["ie"], "iao": ["iA", "U"], "iu": ["iO", "U"], "ian": ["ie", "N"],
    "in": ["I", "N"], "iang": ["iA", "ng"], "ing": ["I", "ng"], "iong": ["iO", "ng"],
    "u": ["u"], "ua": ["ua"], "uo": ["uo"], "uai": ["ua", "ii"], "ui": ["uE", "ii"], "uan": ["ua", "N"],
    "un": ["V", "N"], "uang": ["uA", "ng"],
    "v": ["yu"], "ve": ["yE"], "van": ["yE", "N"], "vn": ["yI", "N"],
}
# 声母の無い音節（綴りの y w は頭子音の y w。yue・yuan は rr）
_ZERO: Dict[str, List[str]] = {
    "a": ["a"], "o": ["O"], "e": ["e"], "ai": ["a", "ii"], "ei": ["E", "ii"], "ao": ["A", "U"], "ou": ["O", "U"],
    "an": ["a", "N"], "en": ["e", "N"], "ang": ["A", "ng"], "eng": ["e", "ng"], "er": ["er"],
    "yi": ["i"], "ya": ["y", "a"], "yo": ["y", "O"], "ye": ["y", "E"], "yao": ["y", "A", "U"], "you": ["y", "O", "U"],
    "yan": ["y", "E", "N"], "yin": ["I", "N"], "yang": ["y", "A", "ng"], "ying": ["I", "ng"], "yong": ["y", "O", "ng"],
    "wu": ["u"], "wa": ["w", "a"], "wo": ["w", "O"], "wai": ["w", "a", "ii"], "wei": ["w", "E", "ii"],
    "wan": ["w", "a", "N"], "wen": ["w", "e", "N"], "wang": ["w", "A", "ng"], "weng": ["w", "e", "ng"],
    "yu": ["yu"], "yue": ["rr", "E"], "yuan": ["rr", "E", "N"], "yun": ["yI", "N"],
}
_RETROFLEX = {"zh", "ch", "sh", "r"}       # + i → ih
_DENTAL = {"z", "c", "s"}                  # + i → i0
_PALATAL = {"j", "q", "x"}                 # 綴りの u は ü


def pinyin_to_phonemes(syl: str) -> Optional[List[str]]:
    """拼音 1 音節（小文字・声調なし・ü は v）→ 音素の列。拼音として読めなければ None。"""
    if syl in _ZERO:
        return list(_ZERO[syl])
    ini = next((i for i in _INITIALS if syl.startswith(i)), None)
    if ini is None:
        return None
    fin = syl[len(ini):]
    if fin == "i" and ini in _RETROFLEX:
        return [ini, "ih"]
    if fin == "i" and ini in _DENTAL:
        return [ini, "i0"]
    if ini in _PALATAL and fin.startswith("u"):
        fin = "v" + fin[1:]
    if fin == "ue":                        # lue・nue（lüe・nüe）
        fin = "ve"
    if fin == "o" and ini in ("b", "p", "m"):
        return [ini, "uo"]                 # bo po mo は uo（fo は O）
    if fin not in _FINALS:
        return None
    return [ini] + _FINALS[fin]


# Opencpop の表記（dict/zh_opencpop.phonemes）: 声母 + 韻母（韻母は介音・韻尾を含めて 1 つ）。ゼロ声母は綴りの y w を
# 声母として残し（ya = y a、wu = w u、yue = y ve、yuan = y van）、a o e で始まる音節は韻母だけ（ai = ai）。
# zhi chi shi ri の i は ir、zi ci si の i は i0（LeapSinger の tools/convert_labels.py --split_i と同じ）。
_OPENCPOP_FINALS = frozenset("a o e ai ei ao ou an en ang eng ong er i u v ia ie iao iu ian in iang ing iong "
                             "ua uo uai ui uan un uang ve van vn".split())
_OPENCPOP_ZERO: Dict[str, List[str]] = {"yu": ["y", "v"], "yue": ["y", "ve"], "yuan": ["y", "van"], "yun": ["y", "vn"]}


def pinyin_to_opencpop(syl: str) -> Optional[List[str]]:
    """拼音 1 音節（小文字・声調なし・ü は v）→ Opencpop の表記の音素の列。拼音として読めなければ None。"""
    if syl in _OPENCPOP_ZERO:
        return list(_OPENCPOP_ZERO[syl])
    if syl in _OPENCPOP_FINALS and syl[0] in "aoe":
        return [syl]
    ini = next((i for i in _INITIALS + ("y", "w") if syl.startswith(i)), None)
    if ini is None:
        return None
    fin = syl[len(ini):]
    if fin == "i" and ini in _RETROFLEX:
        return [ini, "ir"]
    if fin == "i" and ini in _DENTAL:
        return [ini, "i0"]
    if ini in _PALATAL and fin.startswith("u"):
        fin = "v" + fin[1:]
    if fin == "ue":                        # lue・nue（lüe・nüe）
        fin = "ve"
    if fin not in _OPENCPOP_FINALS:
        return None
    return [ini, fin]


def is_hanzi(ch: str) -> bool:
    return len(ch) == 1 and (0x3400 <= ord(ch) <= 0x9FFF or 0xF900 <= ord(ch) <= 0xFAFF)


# 声調記号つきの字 → 外した字（ü は v）。ブラウザ版（g2p_zh.ts）と同じ表で、Unicode の正規化には頼らない。
# 拼音は小文字だけ: 大文字を含む歌詞は音素として読む（ACV の音素は大文字と小文字が別の音。xiAU = x iA U）
_TONE_MARKS = {c: base for base, cs in (("a", "āáǎà"), ("e", "ēéěè"), ("i", "īíǐì"), ("o", "ōóǒò"), ("u", "ūúǔù"),
                                         ("v", "ǖǘǚǜü"), ("n", "ńňǹ"), ("m", "ḿ")) for c in cs}
_PINYIN_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789:") | set(_TONE_MARKS)


def _normalize(tok: str) -> str:
    """声調記号を外し、ü（u: も）を v に。"""
    return "".join(_TONE_MARKS.get(c, c) for c in tok).replace("u:", "v")


def _split_pinyin(run: str) -> Optional[List[str]]:
    """続けて書いた拼音（声調の数字は区切りとして扱う）→ 音節の列。読めなければ None。
    切るときの候補は実在する読み（表の読みと、ゼロ声母の綴り・lüe を lue と書いたもの）の最長一致。"""
    sylls, maxsyl = _syllables()
    out: List[str] = []
    for part in "".join(" " if c in "0123456789" else c for c in run).split(" "):
        i = 0
        while i < len(part):
            m = next((part[i:i + L] for L in range(min(maxsyl, len(part) - i), 0, -1)
                      if part[i:i + L] in sylls), None)
            if m is None:
                return None
            out.append(m)
            i += len(m)
    return out


# 拼音の区切り・歌詞に付いてくる句読点（読まない）。半角の : は ü の書き方（nu:）なので入れない
_SEPARATORS = set(" \t'’-·,，.。!！?？、;；：~～…\"“”「」『』()（）")


def lyric_syllables(lyric: str) -> List[str]:
    """歌詞 → 音節の列（漢字は 1 字ずつそのまま、拼音は声調を外した綴り）。漢字・拼音として読めない文字があれば []。"""
    out: List[str] = []
    buf = ""

    def flush() -> bool:
        nonlocal buf
        if buf:
            s = _split_pinyin(_normalize(buf))
            if s is None:
                return False
            out.extend(s)
        buf = ""
        return True

    for ch in lyric:
        if is_hanzi(ch):
            if not flush():
                return []
            out.append(ch)
        elif ch in _SEPARATORS:
            if not flush():
                return []
        elif ch in _PINYIN_CHARS:
            buf += ch
        else:
            return []
    if not flush():
        return []
    return out


@lru_cache(maxsize=1)
def _tables() -> Tuple[Dict[str, str], Dict[str, List[str]], int]:
    d = json.loads(_DATA.read_text(encoding="utf-8"))
    chars = {c: rd for rd, cs in d["chars"].items() for c in cs}
    phrases = {k: v.split() for k, v in d["phrases"].items()}
    return chars, phrases, max((len(k) for k in phrases), default=1)


@lru_cache(maxsize=1)
def _syllables() -> Tuple[frozenset, int]:
    rds = set(json.loads(_DATA.read_text(encoding="utf-8"))["chars"]) | set(_ZERO)
    rds |= {r.replace("ve", "ue") for r in rds if r.endswith("ve")}
    s = frozenset(r for r in rds if pinyin_to_phonemes(r) is not None)
    return s, max(len(r) for r in s)


def hanzi_to_pinyin(text: str) -> List[str]:
    """漢字の並び → 1 字ずつの拼音（熟語の表の左からの最長一致、当たらない字は既定の読み）。"""
    chars, phrases, maxlen = _tables()
    out: List[str] = []
    i = 0
    while i < len(text):
        hit = next((text[i:i + L] for L in range(min(maxlen, len(text) - i), 1, -1) if text[i:i + L] in phrases), None)
        if hit is not None:
            out.extend(phrases[hit])
            i += len(hit)
            continue
        rd = chars.get(text[i])
        if rd is None:
            raise KeyError(f"漢字「{text[i]}」の読みが辞書にありません（拼音か音素で書いてください）")
        out.append(rd)
        i += 1
    return out


def syllables_to_phonemes(syllables: Sequence[str], scheme: str = "acv") -> List[List[str]]:
    """続けて歌う音節（漢字 1 字か拼音 1 音節ずつ）の並び → 音節ごとの音素の列。
    漢字は続いている所をまとめて読む（拼音を挟むと熟語の表の当たりはそこで切れる）。
    scheme: 'acv'（ACV の表記）か 'opencpop'（Opencpop の表記）。"""
    to_ph = pinyin_to_opencpop if scheme == "opencpop" else pinyin_to_phonemes
    pins: List[str] = []
    k = 0
    while k < len(syllables):
        if is_hanzi(syllables[k]):
            j = k
            while j < len(syllables) and is_hanzi(syllables[j]):
                j += 1
            pins.extend(hanzi_to_pinyin("".join(syllables[k:j])))
            k = j
        else:
            pins.append(syllables[k])
            k += 1
    out = []
    for s, p in zip(syllables, pins):
        ph = to_ph(p)
        if ph is None:
            raise KeyError(f"「{s}」の読み {p!r} を音素にできません（拼音か音素で書いてください）")
        out.append(ph)
    return out
