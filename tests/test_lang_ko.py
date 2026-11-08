"""韓国語の言語パック（dict/ko.yaml）: 読めること、核・無音・子音の分け方、音節の分け方（頭子音のまとまりと終声）、
句末の終声を子音長モデルの対象にすること（final_consonants）。"""
import numpy as np
import torch

from duration_model import phoneset
from duration_model.features import extract_phrase_records
from duration_model.model import ConsonantDurModel
from svs.cons_dur import predict_cons_naturals
from svs.duration_postprocess import FALLBACK_CONS_DUR, Note, Phrase, split_syllable


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


def test_final_consonants_become_records_only_where_the_pack_says_so():
    ko, ja = phoneset.load("ko"), phoneset.load("ja")
    assert ko.final_consonants and not ja.final_consonants
    # 사랑 + 休み: 句末の NG は前のノートの終声として記録される（音高は前のノート、next_note_dur = 0）
    phs = ["pau", "s", "a", "r", "a", "NG", "pau"]
    durs = np.array([0.2, 0.08, 0.3, 0.05, 0.5, 0.15, 0.2])
    vals = np.array([np.nan, np.nan, 62.0, np.nan, 64.0, np.nan, np.nan])
    recs = {r["cons"]: r for r in extract_phrase_records(phs, durs, vals, phoneset=ko)}
    assert set(recs) == {"s", "r", "NG"}
    ng = recs["NG"]
    assert ng["pitch_midi"] == 64 and ng["pitch_diff"] == 0.0 and ng["next_note_dur"] == 0.0
    assert ng["prev_note_exists"] == 1 and ng["target_dur"] == 0.15
    assert recs["r"]["pitch_midi"] == 64 and recs["r"]["next_note_dur"] > 0                  # 句の中は今までどおり
    # 日本語（final_consonants: false）は今までどおり、句末の子音を記録しない
    ja_recs = extract_phrase_records(["pau", "k", "a", "cl", "pau"], np.array([0.2, 0.08, 0.3, 0.1, 0.2]),
                                     np.array([np.nan, np.nan, 60.0, np.nan, np.nan]), phoneset=ja)
    assert [r["cons"] for r in ja_recs] == ["k"]


def test_inference_predicts_the_phrase_final_coda_instead_of_the_fallback():
    ko = phoneset.load("ko")
    torch.manual_seed(0)
    model = ConsonantDurModel(n_phonemes=ko.n_phonemes, use_cluster_len=True).eval()
    model.phoneset = ko
    phrase = Phrase(notes=[Note(["s", "a"], 62, 0.4), Note(["r", "a", "NG"], 64, 0.8)])
    onset, coda = predict_cons_naturals(phrase, model, phoneset=ko)
    assert len(coda[1]) == 1 and coda[1][0] != FALLBACK_CONS_DUR                          # 句末の NG も予測された
    assert onset[0][0] != FALLBACK_CONS_DUR and onset[1][0] != FALLBACK_CONS_DUR
