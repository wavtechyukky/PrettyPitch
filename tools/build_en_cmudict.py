"""英語の G2P（svs/g2p_en.py）が使う発音辞書 dict/en_cmudict.txt を、CMUdict（cmusphinx/cmudict の cmudict.dict）から作る。

    curl -LO https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
    python -m tools.build_en_cmudict cmudict.dict

中身: 1 行に「単語 音素 音素 …」（単語は小文字、音素は dict/en_full.phonemes の表記）。単語ごとに最初の発音だけ。
写し方: ARPAbet を小文字にして強勢の数字を落とす。ただし強勢の無い AH0 は ax（シュワー）にする（AIdol のラベルと
同じ書き分け）。en_full の音素表に無い音素を含む語は入れない。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dict" / "en_cmudict.txt"


def to_en_full(ph: str) -> str:
    """CMUdict の音素 1 つ（AH0・EY1 など）→ en_full の表記。"""
    return "ax" if ph == "AH0" else "".join(c for c in ph if not c.isdigit()).lower()


def main() -> None:
    src = Path(sys.argv[1])
    known = set()
    for line in (ROOT / "dict" / "en_full.phonemes").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            known.add(line)
    words: dict[str, list[str]] = {}
    dropped = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()      # 行末の注記（# place, danish など）
        if not line:
            continue
        w, *phs = line.split()
        if "(" in w:                               # 別の発音（word(2)）は使わない
            continue
        phs = [to_en_full(p) for p in phs]
        if not phs or any(p not in known for p in phs):
            dropped.append(w)
            continue
        words[w] = phs
    head = (f"# CMUdict（https://github.com/cmusphinx/cmudict の cmudict.dict）を tools/build_en_cmudict.py で"
            f" en_full の表記にしたもの。{len(words)} 語。\n"
            "# Copyright (C) 1993-2015 Carnegie Mellon University. BSD 系のライセンス（LICENSES/cmudict-BSD.txt）。\n")
    OUT.write_text(head + "".join(f"{w} {' '.join(p)}\n" for w, p in words.items()), encoding="utf-8")
    print(f"{OUT}: {len(words)} 語、{OUT.stat().st_size / 1e6:.1f} MB。外した語 {len(dropped)}: {dropped[:20]}")


if __name__ == "__main__":
    main()
