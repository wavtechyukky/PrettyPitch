"""韓国語の G2P: ハングルの音節の並び → Team CODA の音素（dict/ko.phonemes）。

韓国語は前後の音節で発音が変わるので、1 音節ずつ表で引くことはできない（日本語のモーラ表のようにはいかない）。
休みを挟まずに続けて歌う音節の並びをまとめて受け取り、音節ごとの音素の列を返す。

扱う規則（標準発音法の主なもの。語の切れ目や品詞を見る規則は扱わない）:
  - 終声の 7 代表音（K N T L M P NG）、二重パッチムの一方だけを読む
  - 連音: 終声 + ㅇ で始まる音節 → 終声が次の音節の頭へ移る（음악 → 으막）。二重パッチムは後ろだけが移る（읽어 → 일거）
  - 口蓋音化: ㄷ・ㅌ + 이 → 지・치（굳이 → 구지、같이 → 가치）
  - ㅎ: 終声 ㅎ + ㅇ は落ちる（좋아 → 조아）、ㅎ + ㄱㄷㅈ は激音（좋다 → 조타）、ㅎ + ㅅ → ㅆ、ㅎ + ㄴ → ㄴㄴ。
        終声 ㄱㄷㅂㅈ + ㅎ も激音（축하 → 추카）
  - 鼻音化: K T P + ㄴ ㅁ → NG N M（국물 → 궁물）、ㅁ ㅇ + ㄹ → ㄴ（종로 → 종노）、K P + ㄹ → NG M + ㄴ（국립 → 궁닙）
  - 流音化: ㄴ + ㄹ・ㄹ + ㄴ → ㄹㄹ（신라 → 실라、칼날 → 칼랄）
  - 濃音化: K T P + ㄱㄷㅂㅅㅈ → 濃音（학교 → 학꾜）。ㄺ ㄼ ㄾ の後も濃音（맑게 → 말께）
  - ㅈ ㅉ ㅊ の後の y は落とす（져 → 저）。의 は頭子音が無ければ eu i、あれば i（희망 → 히망）
表記の約束（学習のラベルと同じ）: ㄹㄹ は `L r`、母音の間の ㄹ は `r`、ㅐ・ㅔ は `e`、ㅚ・ㅙ・ㅞ は `w e`。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

_BASE, _N_JUNG, _N_JONG = 0xAC00, 21, 28
_CHO = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
_JUNG = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
_JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ",
         "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

# 頭子音として読むときの音素（ㅇ は頭子音なし）
_ONSET = {"ㄱ": "g", "ㄲ": "kk", "ㄴ": "n", "ㄷ": "d", "ㄸ": "tt", "ㄹ": "r", "ㅁ": "m", "ㅂ": "b", "ㅃ": "pp", "ㅅ": "s",
          "ㅆ": "ss", "ㅇ": None, "ㅈ": "j", "ㅉ": "jj", "ㅊ": "ch", "ㅋ": "k", "ㅌ": "t", "ㅍ": "p", "ㅎ": "h"}
# 母音（わたり音 + 母音）
_VOWEL = {"ㅏ": ["a"], "ㅐ": ["e"], "ㅑ": ["y", "a"], "ㅒ": ["y", "e"], "ㅓ": ["eo"], "ㅔ": ["e"], "ㅕ": ["y", "eo"],
          "ㅖ": ["y", "e"], "ㅗ": ["o"], "ㅘ": ["w", "a"], "ㅙ": ["w", "e"], "ㅚ": ["w", "e"], "ㅛ": ["y", "o"],
          "ㅜ": ["u"], "ㅝ": ["w", "eo"], "ㅞ": ["w", "e"], "ㅟ": ["w", "i"], "ㅠ": ["y", "u"], "ㅡ": ["eu"],
          "ㅢ": ["eu", "i"], "ㅣ": ["i"]}
# 終声の代表音
_CODA = {"ㄱ": "K", "ㄲ": "K", "ㄳ": "K", "ㄴ": "N", "ㄵ": "N", "ㄶ": "N", "ㄷ": "T", "ㄹ": "L", "ㄺ": "K", "ㄻ": "M",
         "ㄼ": "L", "ㄽ": "L", "ㄾ": "L", "ㄿ": "P", "ㅀ": "L", "ㅁ": "M", "ㅂ": "P", "ㅄ": "P", "ㅅ": "T", "ㅆ": "T",
         "ㅇ": "NG", "ㅈ": "T", "ㅊ": "T", "ㅋ": "K", "ㅌ": "T", "ㅍ": "P", "ㅎ": "T"}
# 二重パッチム → (残る方の代表音, 次の音節へ移る方の字母)
_DOUBLE = {"ㄳ": ("K", "ㅅ"), "ㄵ": ("N", "ㅈ"), "ㄶ": ("N", "ㅎ"), "ㄺ": ("L", "ㄱ"), "ㄻ": ("L", "ㅁ"), "ㄼ": ("L", "ㅂ"),
           "ㄽ": ("L", "ㅅ"), "ㄾ": ("L", "ㅌ"), "ㄿ": ("L", "ㅍ"), "ㅀ": ("L", "ㅎ"), "ㅄ": ("P", "ㅅ")}
_FORTIS = {"g": "kk", "d": "tt", "b": "pp", "s": "ss", "j": "jj"}
_ASPIRATE = {"g": "k", "d": "t", "b": "p", "j": "ch"}
_NASAL = {"K": "NG", "T": "N", "P": "M"}
_PALATAL_Y_DROP = {"j", "jj", "ch"}


def is_hangul_syllable(ch: str) -> bool:
    return len(ch) == 1 and 0xAC00 <= ord(ch) <= 0xD7A3


def _split(ch: str):
    k = ord(ch) - _BASE
    return _CHO[k // (_N_JUNG * _N_JONG)], _JUNG[(k // _N_JONG) % _N_JUNG], _JONG[k % _N_JONG]


def syllables_to_phonemes(syllables: Sequence[str]) -> List[List[str]]:
    """続けて歌う音節（ハングル 1 文字ずつ）の並び → 音節ごとの音素の列。"""
    sy = [_split(c) for c in syllables]
    n = len(sy)
    onset: List[Optional[str]] = [_ONSET[c] for c, _, _ in sy]
    coda: List[Optional[str]] = [None] * n
    for i, (cho, jung, jong) in enumerate(sy):
        nxt = sy[i + 1] if i + 1 < n else None
        if not jong:
            continue
        if nxt is None:                                        # 句末: 代表音
            coda[i] = _CODA[jong]
            continue
        ncho, njung, _ = nxt
        if ncho == "ㅇ":                                       # ── 連音 ──
            if jong == "ㅇ":
                coda[i] = "NG"
            elif jong == "ㅎ":                                  # 좋아 → 조아
                pass
            elif jong in _DOUBLE:
                keep, move = _DOUBLE[jong]
                if move == "ㅎ":                               # 않아 → 아나、싫어 → 시러
                    onset[i + 1] = "n" if keep == "N" else "r"
                else:
                    coda[i] = keep
                    mv = _ONSET[move]
                    if move == "ㅌ" and njung == "ㅣ":         # 핥이다 → 할치다
                        mv = "ch"
                    if keep in ("K", "P") and mv in _FORTIS:   # 넋이 → 넉씨、값이 → 갑씨
                        mv = _FORTIS[mv]
                    onset[i + 1] = mv
            else:
                mv = _ONSET[jong]
                if njung == "ㅣ" and jong in ("ㄷ", "ㅌ"):     # 口蓋音化: 굳이 → 구지、같이 → 가치
                    mv = "j" if jong == "ㄷ" else "ch"
                onset[i + 1] = mv
            continue
        nb = onset[i + 1]
        if ncho == "ㅎ" and jong in ("ㄱ", "ㄲ", "ㄷ", "ㅂ", "ㅈ", "ㅅ", "ㅊ", "ㅌ", "ㅍ", "ㅋ", "ㄺ", "ㄼ", "ㄵ"):
            # 激音化: 축하 → 추카、입학 → 이팍、맞히다 → 마치다。二重パッチムは前が残る（읽히다 → 일키다）
            rep = _CODA[jong] if jong not in _DOUBLE else _DOUBLE[jong][0]
            src = {"K": "k", "T": "t", "P": "p", "N": "ch", "L": {"ㄺ": "k", "ㄼ": "p"}.get(jong, "t")}[rep]
            if jong in ("ㅈ", "ㅊ"):
                src = "ch"
            if jong in _DOUBLE:
                coda[i] = _DOUBLE[jong][0]
            onset[i + 1] = src
            continue
        if jong in ("ㅎ", "ㄶ", "ㅀ"):                         # 終声の ㅎ + 子音
            keep = {"ㅎ": None, "ㄶ": "N", "ㅀ": "L"}[jong]
            if nb in _ASPIRATE:                                 # 좋다 → 조타、많고 → 만코
                coda[i], onset[i + 1] = keep, _ASPIRATE[nb]
            elif nb == "s":                                     # 좋소 → 조쏘
                coda[i], onset[i + 1] = keep, "ss"
            elif nb == "n":                                     # 놓는 → 논는、않는 → 안는、뚫는 → 뚤른
                if keep == "L":
                    coda[i], onset[i + 1] = "L", "r"
                else:
                    coda[i] = "N"
            else:
                coda[i] = keep if keep else "T"
            continue
        # ── 代表音にしてから、次の頭子音と同化 ──
        c = _CODA[jong]
        if jong in ("ㄺ",) and nb == "g":                       # 맑게 → 말께
            c = "L"
        if c in _NASAL and nb in ("n", "m"):                    # 鼻音化: 국물 → 궁물
            c = _NASAL[c]
        elif nb == "r":
            if c in ("M", "NG"):                                # 종로 → 종노
                onset[i + 1] = "n"
            elif c in ("K", "P"):                               # 국립 → 궁닙
                c, onset[i + 1] = _NASAL[c], "n"
            elif c == "T":
                c, onset[i + 1] = "N", "n"
            elif c == "N":                                      # 신라 → 실라
                c = "L"
            # c == "L": 흘러 → L r（そのまま）
        elif c == "L" and nb == "n":                            # 칼날 → 칼랄
            onset[i + 1] = "r"
        elif c in ("K", "T", "P") and nb in _FORTIS:            # 濃音化: 학교 → 학꾜
            onset[i + 1] = _FORTIS[nb]
        elif c == "L" and jong in ("ㄺ", "ㄼ", "ㄾ") and nb in _FORTIS:   # 넓게 → 널께、핥다 → 할따
            onset[i + 1] = _FORTIS[nb]
        coda[i] = c
    out: List[List[str]] = []
    for i, (cho, jung, jong) in enumerate(sy):
        v = list(_VOWEL[jung])
        if jung == "ㅢ" and onset[i] is not None:               # 희망 → 히망
            v = ["i"]
        if v[0] == "y" and onset[i] in _PALATAL_Y_DROP:         # 져 → 저
            v = v[1:]
        out.append(([onset[i]] if onset[i] else []) + v + ([coda[i]] if coda[i] else []))
    return out


def lyric_syllables(lyric: str) -> List[str]:
    """歌詞の文字列からハングルの音節だけを取り出す（空白・記号は捨てる）。"""
    return [c for c in lyric if is_hangul_syllable(c)]
