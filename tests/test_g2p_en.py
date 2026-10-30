"""英語の G2P（svs/g2p_en.py）と、楽譜処理への組み込み（svs/score_io.py の _apply_g2p_en）。"""
import pytest

from duration_model import phoneset
from svs.g2p_en import explicit_phonemes, load_dict, lyric_phonemes, lyric_words
from svs.score_io import ScoreNote, score_to_phrases


def _en():
    return phoneset.load("en_full")


def _notes(*lyrics):
    return [ScoreNote(dur_beats=1.0, midi=60, lyric=lyr) for lyr in lyrics]


def _per_note(*lyrics):
    return [" ".join(n.phonemes) for n in score_to_phrases(_notes(*lyrics), bpm=120, phoneset=_en())[0].notes]


def test_dictionary_uses_only_en_full_phonemes_and_writes_unstressed_ah_as_ax():
    en, d = _en(), load_dict()
    assert len(d) > 100_000
    assert all(p in en.ph2id for phs in d.values() for p in phs)
    assert d["the"] == ["dh", "ax"] and d["hello"] == ["hh", "ax", "l", "ow"] and d["love"] == ["l", "ah", "v"]


@pytest.mark.parametrize("lyric, expected", [
    ("hello", "hh ax l ow"), ("Hello,", "hh ax l ow"), ("don’t", "d ow n t"), ("'cause", "k ax z"),
    ("I love", "ay l ah v"), ("well-known", "w eh l n ow n"), ("oh", "ow"),
])
def test_words(lyric, expected):
    assert " ".join(lyric_phonemes(lyric, _en())) == expected


def test_unknown_words_are_not_read():
    assert lyric_words("zqxvwk") is None
    assert lyric_phonemes("zqxvwk", _en()) is None


@pytest.mark.parametrize("lyric, expected", [
    ("[ow]", "ow"), ("hello[hh eh l ow]", "hh eh l ow"), ("hh eh l ow", "hh eh l ow"), ("s", "s"), ("dx", "dx"),
    ("HH AH0 L OW1", "hh ax l ow"),
])
def test_phonemes_written_directly(lyric, expected):
    assert " ".join(explicit_phonemes(lyric, _en())) == expected


def test_a_word_that_is_also_a_vowel_phoneme_is_read_as_the_word():
    assert explicit_phonemes("oh", _en()) is None              # 単語の oh（ow）。音素の oh は [oh]
    assert explicit_phonemes("[oh]", _en()) == ["oh"]


def test_bad_phonemes_in_brackets_raise():
    with pytest.raises(KeyError):
        explicit_phonemes("[hh xx]", _en())


def test_syllables_go_to_the_following_plus_notes():
    assert _per_note("again", "+") == ["ax", "g eh n"]
    assert _per_note("circumstances", "+", "+", "+") == ["s er", "k ax", "m s t ae", "n s ax z"]
    assert _per_note("[hh eh l ow]", "+") == ["hh eh", "l ow"]


def test_more_syllables_than_notes_stay_on_the_last_note():
    assert _per_note("circumstances", "+") == ["s er", "k ax m s t ae n s ax z"]
    assert _per_note("hello") == ["hh ax l ow"]


def test_more_notes_than_syllables_hold_the_vowel_and_move_the_coda_to_the_end():
    assert _per_note("night", "+", "+") == ["n ay", "ay", "ay t"]
    assert _per_note("again", "+", "+") == ["ax", "g eh", "eh n"]


def test_consonant_only_note_and_unknown_word():
    assert _per_note("night", "s") == ["n ay t", "s"]
    with pytest.raises(KeyError):
        _per_note("zqxvwk")
