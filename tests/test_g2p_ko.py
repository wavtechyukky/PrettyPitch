"""韓国語の G2P（svs/g2p_ko.py）と、楽譜処理への組み込み（svs/score_io.py の _apply_g2p）。"""
import pytest

from duration_model import phoneset
from svs.g2p_ko import lyric_syllables, syllables_to_phonemes
from svs.score_io import ScoreNote, score_to_phrases


def g2p(word: str) -> str:
    return " | ".join(" ".join(p) for p in syllables_to_phonemes(list(word)))


@pytest.mark.parametrize("word, expected", [
    ("사랑해", "s a | r a NG | h e"),               # ㅇ + ㅎ はそのまま、母音の間の ㄹ は r
    ("음악", "eu | m a K"),                         # 連音
    ("있어", "i | ss eo"),                          # 濃音の終声の連音
    ("좋아", "j o | a"),                            # ㅎ + ㅇ は落ちる
    ("좋다", "j o | t a"),                          # ㅎ + ㄷ → ㅌ
    ("많다", "m a N | t a"),                        # ㄶ + ㄷ → ㄴ ㅌ
    ("싫어", "s i | r eo"),                         # ㅀ + ㅇ → ㄹ が移る
    ("놓는", "n o N | n eu N"),                     # ㅎ + ㄴ → ㄴ ㄴ
    ("읽어", "i L | g eo"),                         # 二重パッチムは後ろだけ移る
    ("값이", "g a P | ss i"),                       # ㅄ + 이 → ㅂ ㅆ
    ("앉아", "a N | j a"),
    ("닭", "d a K"),                                # 句末の二重パッチム → 代表音
    ("맑게", "m a L | kk e"),                       # ㄺ + ㄱ → ㄹ ㄲ
    ("굳이", "g u | j i"),                          # 口蓋音化
    ("같이", "g a | ch i"),
    ("꽃이", "kk o | ch i"),                        # ㅊ の連音
    ("축하", "ch u | k a"),                         # ㄱ + ㅎ → ㅋ
    ("국물", "g u NG | m u L"),                     # 鼻音化
    ("밥만", "b a M | m a N"),
    ("종로", "j o NG | n o"),                       # ㅇ + ㄹ → ㄴ
    ("국립", "g u NG | n i P"),                     # ㄱ + ㄹ → ㅇ ㄴ
    ("신라", "s i L | r a"),                        # 流音化
    ("칼날", "k a L | r a L"),
    ("흘러", "h eu L | r eo"),                      # ㄹㄹ = L r
    ("학교", "h a K | kk y o"),                     # 濃音化
    ("없어", "eo P | ss eo"),
    ("희망", "h i | m a NG"),                       # 子音 + ㅢ → i
    ("의자", "eu i | j a"),                         # 頭子音の無い ㅢ → eu i
    ("저", "j eo"),
    ("져", "j eo"),                                 # ㅈ の後の y は落ちる
    ("왜", "w e"),
    ("외", "w e"),
    ("귀", "g w i"),
])
def test_g2p_words(word, expected):
    assert g2p(word) == expected


def test_every_phoneme_is_in_the_korean_list():
    ko = set(phoneset.load("ko").phonemes)
    allsyl = [chr(c) for c in range(0xAC00, 0xD7A4, 7)]           # 1,596 音節（全体から間引き）
    for i in range(0, len(allsyl) - 1, 2):
        for p in syllables_to_phonemes(allsyl[i:i + 2]):
            assert set(p) <= ko, (allsyl[i:i + 2], p)


def test_lyric_syllables_ignores_spaces_and_marks():
    assert lyric_syllables("아리 랑~!") == ["아", "리", "랑"]


def test_score_to_phrases_applies_g2p_across_notes_and_moves_the_coda_to_the_melisma_end():
    ko = phoneset.load("ko")
    sn = [ScoreNote(1, 60, "음"), ScoreNote(1, 62, "악"), ScoreNote(1, 0, None, is_rest=True),
          ScoreNote(1, 64, "밤"), ScoreNote(1, 65, "-"), ScoreNote(1, 67, "-"),
          ScoreNote(0.5, 0, None, is_rest=True), ScoreNote(1, 60, "사랑")]
    phrases = score_to_phrases(sn, 120, phoneset=ko)
    notes = [n.phonemes for ph in phrases for n in ph.notes]
    assert notes == [["eu"], ["m", "a", "K"],                       # 連音は休符をまたがない
                     ["b", "a"], ["a"], ["a", "M"],                 # 밤 の終声はメリスマの最後へ
                     ["s", "a", "r", "a", "NG"]]                    # 1 ノートに 2 音節
    # 日本語は今までどおり（g2p なし）
    ja = phoneset.load("ja")
    assert ja.g2p is None
    assert [n.phonemes for ph in score_to_phrases([ScoreNote(1, 60, "か")], 120, phoneset=ja)
            for n in ph.notes] == [["k", "a"]]
