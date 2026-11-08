"""英語の G2P（svs/g2p_en.py）と、楽譜処理への組み込み（svs/score_io.py の _apply_g2p_en）。"""
import pytest

from duration_model import phoneset
from svs.g2p_en import explicit_phonemes, load_dict, lyric_phonemes, lyric_words
from svs.score_io import ScoreNote, score_to_phrases


def _en():
    return phoneset.load("en_full")


def _notes(*lyrics):
    return [ScoreNote(dur_beats=1.0, midi=60, lyric=lyr) for lyr in lyrics]


def _per_note(*lyrics, style="sequential"):
    phrases = score_to_phrases(_notes(*lyrics), bpm=120, phoneset=_en(), lyric_style=style)
    return [" ".join(n.phonemes) for n in phrases[0].notes]


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


@pytest.mark.parametrize("style", ["sequential", "count"])
def test_hold_marks_keep_the_vowel_without_taking_a_syllable(style):
    for hold in ("-", "+~", "+*", "ー", None):
        assert _per_note("night", hold, style=style) == ["n ay", "ay t"]
    assert _per_note("wonderful", "+~", "+", "+", style=style) == ["w ah", "ah", "n d er", "f ax l"]
    assert _per_note("again", "+", "-", style=style) == ["ax", "g eh", "eh n"]


@pytest.mark.parametrize("style", ["sequential", "count"])
def test_plus_number_starts_at_that_phoneme_of_the_word(style):
    assert _per_note("wonderful", "+4", "+", style=style) == ["w ah n", "d er", "f ax l"]
    assert _per_note("wonderful", "+3", "+", style=style) == ["w ah", "n d er", "f ax l"]
    assert _per_note("wonderful", "+1", "+", style=style) == ["w ah", "n d er", "f ax l"]   # 前の音符より前は + と同じ


def test_sequential_style_puts_leftover_syllables_on_the_last_note_and_reads_plus_plus_as_plus():
    assert _per_note("accompany", "++", "+") == ["ax", "k ah", "m p ax n iy"]
    assert _per_note("accompany", "+", "-") == ["ax", "k ah m p ax n iy", "iy"]


def test_count_style_takes_as_many_syllables_as_pluses_and_drops_the_rest():
    assert _per_note("accompany", "++", "+", style="count") == ["ax", "k ah m p ax", "n iy"]
    assert _per_note("accompany", "++", style="count") == ["ax", "k ah m p ax"]
    assert _per_note("accompany", "+", "-", style="count") == ["ax", "k ah", "ah"]
    assert _per_note("accompany", "-", style="count") == ["ax k ah m p ax n iy", "iy"]   # + が無ければ単語を全部
    assert _per_note("night", "+", "+", style="count") == ["n ay", "ay", "ay t"]


def test_marks_after_a_rest_hold_the_previous_vowel():
    notes = _notes("night", "+~")
    notes.insert(1, ScoreNote(dur_beats=0.25, midi=0, lyric=None, is_rest=True))
    assert [" ".join(n.phonemes) for n in score_to_phrases(notes, bpm=120, phoneset=_en())[0].notes] == ["n ay t", "ay"]


def test_unknown_lyric_style_raises():
    with pytest.raises(ValueError):
        _per_note("night", style="other")
