"""Held vowels in svs: a vowel continued over a hold note ("skies" "-").

1. The phoneme sequence marks the continued vowel (``PhonemeSequence.held``) only when the
   note was a hold mark in the score and its vowel is the one already sounding; a vowel sung
   again ("my" "eyes", "か" "あ") is not held.
2. An F0 model trained with a hold symbol gets that symbol for held slots; other models get
   the repeated vowel exactly as before.
3. ``merge_held_vowels`` folds held slots into the previous phoneme for the acoustic model and
   keeps the total frame count.
"""
import numpy as np

from duration_model import phoneset
from svs.duration_postprocess import postprocess_durations, split_phrase_syllables
from svs.render import _f0_score_from_seq, merge_held_vowels
from svs.score_io import ScoreNote, score_to_phrases


def _seq(lang, *notes):
    ps = phoneset.load(lang)
    sns = [ScoreNote(dur_beats=1.0, midi=m, lyric=lyr) for lyr, m in notes]
    phrase = split_phrase_syllables(score_to_phrases(sns, bpm=100, phoneset=ps)[0], ps)
    return postprocess_durations(phrase, phoneset=ps)


def _held(seq):
    return [p for p, h in zip(seq.phonemes, seq.held) if h]


def test_hold_mark_continues_the_vowel():
    seq = _seq("en_full", ("skies", 60), ("-", 62))
    assert _held(seq) == ["ay"]
    i = int(np.flatnonzero(seq.held)[0])
    assert seq.phonemes[i - 1] == "ay" and seq.note_midi[i] == 62


def test_vowel_sung_again_is_not_held():
    assert not _seq("en_full", ("my", 60), ("eyes", 62)).held.any()
    assert not _seq("ja", ("か", 60), ("あ", 62)).held.any()


def test_japanese_hold_marks_are_held_too():
    assert _held(_seq("ja", ("か", 60), ("-", 62))) == ["a"]
    assert _held(_seq("ja", ("か", 60), ("ー", 62), ("-", 64))) == ["a", "a"]


def test_f0_input_uses_the_symbol_only_for_models_trained_with_it():
    seq = _seq("en_full", ("skies", 60), ("-", 62))
    old = [s["ph"] for s in _f0_score_from_seq(seq)["phonemes"]]
    new = _f0_score_from_seq(seq, hold_symbol="-")["phonemes"]
    assert old == list(seq.phonemes)                              # unchanged: the vowel repeats
    assert [s["ph"] for s in new] == ["-" if h else p for p, h in zip(seq.phonemes, seq.held)]
    i = int(np.flatnonzero(seq.held)[0])
    assert new[i]["midi"] == 62                                   # the held note keeps its own pitch


def test_merge_held_vowels_keeps_the_total_length():
    ids, durs = np.array([0, 1, 2, 2, 3]), np.array([5, 3, 4, 6, 2], np.int32)
    m_ids, m_durs = merge_held_vowels(ids, durs, np.array([False, False, False, True, False]))
    assert m_ids.tolist() == [0, 1, 2, 3] and m_durs.tolist() == [5, 3, 10, 2]
    assert int(m_durs.sum()) == int(durs.sum()) and m_durs.dtype == durs.dtype
    same = merge_held_vowels(ids, durs, None)
    assert same[0] is ids and same[1] is durs
