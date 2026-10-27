"""中国語の言語パック（dict/zh_acv.yaml、ArchiVoice の ACV 表記）: 読めること、核・無音・子音の分け方、
音節の分け方（頭子音・介音を含んだ主母音・韻尾）、LeapSinger と同じ音素表であること、ACV のレシピの drop_phones。"""
from pathlib import Path

import yaml

from duration_model import phoneset
from svs.duration_postprocess import split_syllable

ROOT = Path(__file__).resolve().parent.parent


def test_zh_acv_pack_loads_with_main_vowel_nuclei():
    zh = phoneset.load("zh_acv")
    assert zh.lang == "zh_acv"
    assert zh.phonemes[0] == "pau" and zh.n_phonemes == 55
    assert len(zh.nuclei) == 24 and zh.nuclei == zh.vowels                  # 主母音 23 + ハミング M
    assert {"iA", "uA", "yE", "uE", "iO", "M"} <= zh.nuclei
    # 大文字と小文字は別の音素。畳まない
    assert zh.fold("A") == "A" and zh.to_id("a") != zh.to_id("A") and zh.to_id("m") != zh.to_id("M")


def test_zh_acv_silence_and_consonants():
    zh = phoneset.load("zh_acv")
    for s in ("SP", "AP", "mlem", "exh", "axh", "hx", "pau"):
        assert zh.is_pau(s), s
    assert not zh.is_pau("vf") and "vf" in zh.consonants                     # ボーカルフライは音素のまま
    # 声母・ゼロ声母の頭・韻尾・閉鎖は子音（長さを予測する）
    for c in ("b", "zh", "y", "w", "rr", "ii", "U", "N", "ng", "cl", "vf"):
        assert zh.is_consonant(c) and not zh.is_nucleus(c), c
    assert zh.final_consonants                                               # 句末の韻尾も予測の対象


def test_zh_acv_syllables_split_into_onset_main_vowel_and_coda():
    zh = phoneset.load("zh_acv")
    assert split_syllable(["x", "iA", "U"], zh) == (["x"], ["iA"], ["U"])     # xiao
    assert split_syllable(["zh", "uA", "ng"], zh) == (["zh"], ["uA"], ["ng"])  # zhuang
    assert split_syllable(["rr", "E", "N"], zh) == (["rr"], ["E"], ["N"])      # yuan
    assert split_syllable(["a", "ii"], zh) == ([], ["a"], ["ii"])             # ai
    assert split_syllable(["vf", "er"], zh) == (["vf"], ["er"], [])           # 頭にボーカルフライ


def test_phoneme_list_matches_leapsinger_when_it_sits_next_to_this_repository():
    other = ROOT.parent / "LeapSinger_v2" / "dict" / "zh_acv.phonemes"
    if not other.exists():
        other = ROOT.parent / "LeapSinger" / "dict" / "zh_acv.phonemes"
    if not other.exists():
        return                                                               # 隣に無ければ確かめない
    read = phoneset._read_list
    assert read(ROOT / "dict" / "zh_acv.phonemes") == read(other)


def test_acv_recipes_and_config_agree():
    cfg = yaml.safe_load((ROOT / "configs" / "zh_acv.yaml").read_text(encoding="utf-8"))
    assert cfg["lang"] == "zh_acv"
    for db, spk in cfg["data"]["spk_map"].items():
        r = yaml.safe_load((ROOT / "configs" / "recipes" / f"{db}.yaml").read_text(encoding="utf-8"))
        assert r["name"] == db and r["lang"] == "zh_acv" and r["drop_phones"] == ["en/"], db
    assert cfg["data"]["spk_map"] == {"acv001_cn": 0, "acv002_cn": 1}
    assert set(cfg["data"]["eval_dbs"]) <= set(cfg["data"]["spk_map"])


def test_drop_phones_turns_the_rows_into_pau_and_reports_their_spans(tmp_path):
    import numpy as np
    from preprocess.run import _drop_fix, _drop_spans
    lab = tmp_path / "mix.lab"                                               # 中国語の曲の中の英単語
    lab.write_text("0 1000000 SP\n1000000 2000000 n\n2000000 4000000 i\n4000000 5000000 en/l\n"
                   "5000000 7000000 en/ay\n7000000 9000000 SP\n", encoding="utf-8")
    assert np.allclose(_drop_spans(str(lab), "100ns", None, ("en/",)), [[0.4, 0.5], [0.5, 0.7]])
    assert _drop_spans(str(lab), "100ns", None, ()) == []
    rows = _drop_fix(None, ("en/",))([(0.0, 0.1, "n"), (0.1, 0.2, "en/l"), (0.2, 0.3, "i")])
    assert [p for _s, _e, p in rows] == ["n", "pau", "i"]
    assert _drop_fix(None, ()) is None
