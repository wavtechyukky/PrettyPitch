"""韓国語の言語パック（dict/ko.yaml）: 読めること、核・無音・子音の分け方、音節の分け方（頭子音のまとまりと終声）。"""
from duration_model import phoneset
from svs.duration_postprocess import split_syllable


def test_ko_pack_loads_with_seven_vowel_nuclei():
    ko = phoneset.load("ko")
    assert ko.lang == "ko"
    assert ko.phonemes[0] == "pau" and ko.n_phonemes == 37
    assert ko.nuclei == {"a", "eo", "o", "u", "eu", "i", "e"} == ko.vowels
    # 初声（小文字）と終声（大文字）は別の音素。畳まない
    assert {"n", "N", "k", "K", "t", "T", "p", "P", "m", "M"} <= set(ko.phonemes)
    assert ko.fold("N") == "N" and ko.to_id("n") != ko.to_id("N")


def test_ko_silence_and_consonants():
    ko = phoneset.load("ko")
    for s in ("SP", "AP", "exh", "trash", "sil", "pau"):
        assert ko.is_pau(s), s
    # vf（ボーカルフライ）は声のある区間なので無音にしない（日本語のパックとは違う）
    assert not ko.is_pau("vf") and "vf" in ko.consonants
    assert phoneset.load("ja").is_pau("vf")
    # わたり音・終声・閉鎖は子音（長さを予測する）
    for c in ("y", "w", "N", "L", "M", "NG", "K", "T", "P", "cl", "kk", "ch"):
        assert ko.is_consonant(c) and not ko.is_nucleus(c), c


def test_ko_syllables_split_into_onset_cluster_nucleus_and_coda():
    ko = phoneset.load("ko")
    assert split_syllable(["g", "w", "a", "N"], ko) == (["g", "w"], ["a"], ["N"])      # 관
    assert split_syllable(["s", "eu", "L"], ko) == (["s"], ["eu"], ["L"])               # 슬
    assert split_syllable(["eo", "NG"], ko) == ([], ["eo"], ["NG"])                     # 엉
    assert split_syllable(["y", "eo"], ko) == (["y"], ["eo"], [])                       # 여
