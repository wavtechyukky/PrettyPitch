"""中国語の言語パック（dict/zh_opencpop.yaml、Opencpop の声母 + 韻母）: 読めること、核・無音・子音の分け方、
音節の分け方、LeapSinger と同じ音素表であること、Opencpop のレシピと設定、前処理の layout: from_lab。"""
from pathlib import Path

import yaml

from duration_model import phoneset
from svs.duration_postprocess import split_syllable

ROOT = Path(__file__).resolve().parent.parent


def test_zh_opencpop_pack_loads_with_final_nuclei():
    zh = phoneset.load("zh_opencpop")
    assert zh.lang == "zh_opencpop"
    assert zh.phonemes[0] == "pau" and zh.n_phonemes == 61
    assert len(zh.nuclei) == 37 and zh.nuclei == zh.vowels                  # 韻母 35 + ir i0
    assert {"iao", "uang", "van", "vn", "ir", "i0"} <= zh.nuclei
    assert not zh.final_consonants


def test_zh_opencpop_silence_and_consonants():
    zh = phoneset.load("zh_opencpop")
    for s in ("SP", "AP", "pau"):
        assert zh.is_pau(s), s
    for c in ("b", "zh", "x", "y", "w", "r"):
        assert zh.is_consonant(c) and not zh.is_nucleus(c), c


def test_zh_opencpop_syllables_split_into_initial_and_final():
    zh = phoneset.load("zh_opencpop")
    assert split_syllable(["x", "iao"], zh) == (["x"], ["iao"], [])
    assert split_syllable(["y", "van"], zh) == (["y"], ["van"], [])
    assert split_syllable(["ang"], zh) == ([], ["ang"], [])


def test_phoneme_list_matches_leapsinger_when_it_sits_next_to_this_repository():
    other = ROOT.parent / "LeapSinger_v2" / "dict" / "zh_opencpop.phonemes"
    if not other.exists():
        return                                                               # 隣に無ければ確かめない
    read = phoneset._read_list
    assert read(ROOT / "dict" / "zh_opencpop.phonemes") == read(other)


def test_opencpop_recipes_and_config_agree():
    cfg = yaml.safe_load((ROOT / "configs" / "zh_opencpop.yaml").read_text(encoding="utf-8"))
    assert cfg["lang"] == "zh_opencpop"
    for db in cfg["data"]["spk_map"]:
        r = yaml.safe_load((ROOT / "configs" / "recipes" / f"{db}.yaml").read_text(encoding="utf-8"))
        assert r["name"] == db and r["lang"] == "zh_opencpop" and r["layout"] == "from_lab", db
    assert set(cfg["data"]["spk_map"].values()) == {0}
    assert cfg["data"]["eval_dbs"] == ["opencpop_test"]


def test_from_lab_layout_lists_the_songs_that_have_a_label(tmp_path):
    from preprocess.run import _enumerate_songs
    (tmp_path / "wavs").mkdir()
    (tmp_path / "labels" / "train").mkdir(parents=True)
    for s in ("a1", "a2", "b1"):
        (tmp_path / "wavs" / f"{s}.wav").write_bytes(b"")
    for s in ("a1", "a2"):
        (tmp_path / "labels" / "train" / f"{s}.lab").write_text("", encoding="utf-8")
    r = {"layout": "from_lab", "wav": "wavs/{song}.wav", "lab": "../labels/train/{song}.lab"}
    assert _enumerate_songs(r, str(tmp_path / "wavs" / "..")) == []          # db_root/../labels は無い
    assert _enumerate_songs(r, str(tmp_path / "wavs")) == ["a1", "a2"]
