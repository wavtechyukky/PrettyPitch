"""中国語の G2P（svs/g2p_zh.py）と、楽譜処理への組み込み（svs/score_io.py の _apply_g2p）。"""
import pytest

from duration_model import phoneset
from svs.g2p_zh import lyric_syllables, pinyin_to_phonemes, syllables_to_phonemes
from svs.score_io import ScoreNote, score_to_phrases


@pytest.mark.parametrize("syl, expected", [
    ("xiao", "x iA U"), ("kuai", "k ua ii"), ("juan", "j yE N"), ("xiong", "x iO ng"),
    ("ya", "y a"), ("yue", "rr E"), ("yuan", "rr E N"), ("yun", "yI N"), ("yi", "i"), ("wu", "u"),
    ("ye", "y E"), ("yan", "y E N"), ("yin", "I N"), ("ying", "I ng"), ("wei", "w E ii"),
    ("zhi", "zh ih"), ("ri", "r ih"), ("si", "s i0"), ("bo", "b uo"), ("fo", "f O"),
    ("jiu", "j iO U"), ("dui", "d uE ii"), ("lun", "l V N"), ("lv", "l yu"), ("lve", "l yE"), ("nue", "n yE"),
    ("qu", "q yu"), ("er", "er"), ("ai", "a ii"), ("ou", "O U"),
])
def test_pinyin_to_acv(syl, expected):
    assert " ".join(pinyin_to_phonemes(syl)) == expected


def test_pinyin_rules_only_use_the_acv_phonemes():
    zh = phoneset.load("zh_acv")
    from svs.g2p_zh import _syllables
    sylls, _ = _syllables()
    assert len(sylls) > 400
    for s in sylls:
        assert all(p in zh.phonemes for p in pinyin_to_phonemes(s)), s


def test_lyric_syllables_reads_hanzi_and_pinyin_with_or_without_tones():
    assert lyric_syllables("我爱你") == ["我", "爱", "你"]
    assert lyric_syllables("nǐ hǎo") == ["ni", "hao"]
    assert lyric_syllables("ni3hao3") == ["ni", "hao"]
    assert lyric_syllables("nihao") == ["ni", "hao"]
    assert lyric_syllables("xi'an") == ["xi", "an"]
    assert lyric_syllables("lǜ") == ["lv"] and lyric_syllables("nu:") == ["nv"]
    assert lyric_syllables("我ai你") == ["我", "ai", "你"]
    assert lyric_syllables("你好，") == ["你", "好"]
    assert lyric_syllables("xiAU") == []           # 大文字は音素（拼音ではない）
    assert lyric_syllables("あ") == [] and lyric_syllables("qqq") == []


def g2p(text: str) -> str:
    return " | ".join(" ".join(p) for p in syllables_to_phonemes(lyric_syllables(text)))


def test_hanzi_readings_follow_the_phrase_table():
    assert g2p("长大") == "zh A ng | d a"
    assert g2p("长城") == "ch A ng | ch e ng"
    assert g2p("音乐") == "I N | rr E"
    assert g2p("快乐") == "k ua ii | l e"
    assert g2p("银行") == "I N | h A ng"
    assert g2p("我们") == "w O | m e N"
    assert g2p("唱歌") == "ch A ng | g e"


def test_unknown_reading_raises():
    with pytest.raises(KeyError):
        syllables_to_phonemes(["嗯"])


def _notes(*lyrics):
    return [ScoreNote(dur_beats=1.0, midi=60, lyric=lyr) for lyr in lyrics]


def test_score_reads_hanzi_across_notes_and_moves_the_coda_after_a_melisma():
    zh = phoneset.load("zh_acv")
    ph = score_to_phrases(_notes("音", "乐", "看", "-", "-"), bpm=120, phoneset=zh)[0]
    assert [n.phonemes for n in ph.notes] == [["I", "N"], ["rr", "E"], ["k", "a"], ["a"], ["a", "N"]]


def test_score_accepts_pinyin_and_phonemes_directly():
    zh = phoneset.load("zh_acv")
    ph = score_to_phrases(_notes("xiao3", "x iA U", "ni"), bpm=120, phoneset=zh)[0]
    assert [n.phonemes for n in ph.notes] == [["x", "iA", "U"], ["x", "iA", "U"], ["n", "i"]]
