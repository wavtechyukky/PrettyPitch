"""中国語の G2P（svs/g2p_zh.py）が使う「漢字 → 拼音」の表 dict/zh_pinyin.json を、pypinyin のデータから作る。

    pip install pypinyin
    python -m tools.build_zh_pinyin

中身（声調は持たない。ü は v）:
  chars    読み → その読みを既定とする字を並べた文字列（pypinyin が 1 字だけのときに返す読み）
  phrases  熟語 → 字ごとの読み（空白区切り）。多音字を含み、読みが既定の読みと違うものだけ（长大 zhang da など）
読みが ACV の音素にできない字（呣 m・嗯 ng など）と、そうした読みを含む熟語は入れない。
"""
from __future__ import annotations

import json
from pathlib import Path

from pypinyin import Style, lazy_pinyin, pinyin
from pypinyin.constants import PHRASES_DICT, PINYIN_DICT
import pypinyin

from svs.g2p_zh import is_hanzi, pinyin_to_phonemes

OUT = Path(__file__).resolve().parents[1] / "dict" / "zh_pinyin.json"


def main() -> None:
    chars: dict[str, str] = {}
    poly: set[str] = set()
    skipped: dict[str, str] = {}
    for cp in sorted(PINYIN_DICT):
        c = chr(cp)
        if not is_hanzi(c):
            continue
        rd = lazy_pinyin(c, style=Style.NORMAL)[0]
        if pinyin_to_phonemes(rd) is None:
            skipped[c] = rd
            continue
        chars[c] = rd
        if len(set(pinyin(c, style=Style.NORMAL, heteronym=True)[0])) > 1:
            poly.add(c)
    phrases: dict[str, str] = {}
    for ph in sorted(PHRASES_DICT):
        if not all(c in chars for c in ph) or not any(c in poly for c in ph):
            continue
        rds = lazy_pinyin(ph, style=Style.NORMAL)
        if len(rds) != len(ph) or any(pinyin_to_phonemes(r) is None for r in rds):
            continue
        if rds != [chars[c] for c in ph]:
            phrases[ph] = " ".join(rds)
    by_reading: dict[str, str] = {}
    for c, rd in chars.items():
        by_reading[rd] = by_reading.get(rd, "") + c
    data = {"source": f"pypinyin {pypinyin.__version__} (MIT)",
            "chars": dict(sorted(by_reading.items())), "phrases": phrases}
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{OUT}: {len(chars)} 字（読み {len(by_reading)} 種）、熟語 {len(phrases)}、"
          f"{OUT.stat().st_size / 1e3:.0f} KB。音素にできず外した字 {len(skipped)}: "
          + " ".join(f"{c}={r}" for c, r in list(skipped.items())[:20]))


if __name__ == "__main__":
    main()
