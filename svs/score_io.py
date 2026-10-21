"""楽譜の読み込み（UST / MusicXML）→ 共通 Note 形式 → フレーズ列。

UST（UTAU）と MusicXML を同じ中間表現 `ScoreNote` に正規化し、モーラ↔音素表で
音素を割り当て、静音区間でフレーズに分割して `duration_postprocess.Phrase` を返す。
フレーズ列はそのまま `postprocess_durations` に渡せる。

言語非依存の方針: モーラ↔音素表（dict/ja.mora）・音素表（dict/ja.phonemes）・核音素集合
（dict/<lang>.yaml の nucleus:）の 3 つを差し替えれば他言語でも動く。

関数引数で変えられる値: split_silence_sec（フレーズ分割する静音長）・
    pau_lead_sec / pau_trail_sec（両端 pau）。
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from duration_model import phoneset as _ps_mod
from .duration_postprocess import Note, Phrase, _is_nucleus_ph

_DICT_DIR = Path(__file__).resolve().parent.parent / 'dict'
# 既定は日本語。他言語は dict/<lang>.mora を作り load_score(mora_table=...) か
# `python -m svs.render --mora_table ...` で差し替える。
DEFAULT_MORA_TABLE = _DICT_DIR / 'ja.mora'

# UTAU の 1 拍（4 分音符）= 480 tick。
_UST_TICKS_PER_BEAT = 480
# 休符記号・メリスマ記号・既定メリスマ母音・UST の文字コードは言語パックが決める
# （dict/<lang>.yaml の score: ブロック）。ここに直書きしない。


def _score_cfg(phoneset=None) -> dict:
    return dict((phoneset or _ps_mod.active()).score or {})


def _rest_lyrics(ps=None) -> frozenset:
    return frozenset(_score_cfg(ps).get('rest_lyrics') or ())


def _melisma_lyrics(ps=None) -> frozenset:
    return frozenset(_score_cfg(ps).get('melisma_lyrics') or ())


def _melisma_vowel(ps=None) -> str:
    return _score_cfg(ps).get('melisma_vowel') or 'a'


def _ust_encoding(ps=None) -> str:
    return _score_cfg(ps).get('ust_encoding') or 'utf-8'


_XML_STEP = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


# ── モーラ↔音素表 ─────────────────────────────────────────────────────────────

def load_mora_table(path: Optional[Path] = None, phoneset=None) -> Dict[str, List[str]]:
    """`<mora>  <ph...>` 形式の表を読む。既定は言語パックの mora:（日本語なら dict/ja.mora）。"""
    if path is None:
        path = (phoneset or _ps_mod.active()).mora_path or DEFAULT_MORA_TABLE
    table: Dict[str, List[str]] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            table[parts[0]] = parts[1:]
    return table


def _as_mora_table(x, phoneset=None) -> Dict[str, List[str]]:
    """dict / パス文字列 / Path / None を読み込み済みモーラ表に正規化する。"""
    if isinstance(x, dict):
        return x
    if x is not None:
        return load_mora_table(Path(x))
    ps = phoneset or _ps_mod.active()
    if ps.mora_path is None:
        # 英語のように汎用のモーラ表を持たない言語。歌詞→音素は呼び出し側が
        # ScoreNote.phonemes で明示するか、--mora_table で語彙限定の表を渡す。
        return {}
    return load_mora_table(ps.mora_path, ps)


# ── 中間表現 ─────────────────────────────────────────────────────────────────

@dataclass
class ScoreNote:
    """楽譜 1 ノート（休符含む）。UST/MusicXML 共通の中間形式。"""
    dur_beats: float               # 長さ [4 分音符 = 1]
    midi: int                      # MIDI ピッチ（休符は 0）
    lyric: Optional[str]           # 歌詞トークン（休符/メリスマは None もしくは休符記号）
    is_rest: bool = False
    phonemes: Optional[List[str]] = None   # 明示指定があれば優先


# ── UST パーサ ───────────────────────────────────────────────────────────────

def parse_ust(path: Path, encoding: str | None = None,
              phoneset=None) -> Tuple[List[ScoreNote], float]:
    """UST → (ScoreNote 列, bpm)。文字コードは言語パックの score.ust_encoding。"""
    encoding = encoding or _ust_encoding(phoneset)
    rest_set, mel_set = _rest_lyrics(phoneset), _melisma_lyrics(phoneset)
    txt = Path(path).read_text(encoding=encoding, errors='replace')
    bpm = 120.0
    m = re.search(r'^Tempo=([\d.]+)', txt, re.M)
    if m:
        bpm = float(m.group(1))

    notes: List[ScoreNote] = []
    for _num, body in re.findall(r'\[#(\d+|PREV|NEXT)\](.*?)(?=\[#|\Z)', txt, re.S):
        fields = dict(re.findall(r'^([A-Za-z0-9]+)=(.*)$', body, re.M))
        if 'Length' not in fields:
            continue
        length = float(fields['Length'])
        lyric = fields.get('Lyric', '').strip()
        note_num = int(fields.get('NoteNum', '0') or 0)
        is_rest = lyric in rest_set and lyric not in mel_set
        notes.append(ScoreNote(
            dur_beats=length / _UST_TICKS_PER_BEAT,
            midi=note_num,
            lyric=lyric,
            is_rest=is_rest))
    return notes, bpm


# ── MusicXML パーサ ─────────────────────────────────────────────────────────

def _xml_local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]   # 名前空間を除く


def _xml_pitch_to_midi(pitch: ET.Element) -> int:
    step = (pitch.findtext('step') or 'C').strip().upper()
    octave = int(pitch.findtext('octave') or 4)
    alter = int(float(pitch.findtext('alter') or 0))
    return (octave + 1) * 12 + _XML_STEP.get(step, 0) + alter


def _mxl_root(path: Path) -> ET.Element:
    """圧縮 MusicXML（.mxl）から本体の XML を取り出す。

    .mxl は zip で、`META-INF/container.xml` の最初の `<rootfile full-path=...>` が本体。
    MuseScore の既定の保存形式がこれなので、開けないと「ソフトで書いて保存したファイルが
    読めない」ということになる。
    """
    import zipfile
    with zipfile.ZipFile(str(path)) as z:
        name = None
        try:
            cont = ET.fromstring(z.read('META-INF/container.xml'))
            for e in cont.iter():
                if _xml_local(e.tag) == 'rootfile' and e.get('full-path'):
                    name = e.get('full-path')
                    break
        except KeyError:
            pass
        if name is None:                       # container が無い/壊れている場合の保険
            cand = [n for n in z.namelist()
                    if n.lower().endswith(('.xml', '.musicxml')) and not n.startswith('META-INF/')]
            if not cand:
                raise ValueError(f'{path.name}: .mxl の中に楽譜 XML が見つかりません')
            name = cand[0]
        return ET.fromstring(z.read(name))


def _xml_parts(root: ET.Element) -> List[Tuple[str, str, ET.Element]]:
    """(part id, part-name, <part> 要素) の一覧。順序は文書どおり。"""
    names = {}
    for sp in root.iter():
        if _xml_local(sp.tag) == 'score-part' and sp.get('id'):
            nm = ''
            for c in sp:
                if _xml_local(c.tag) == 'part-name' and c.text:
                    nm = c.text.strip()
            names[sp.get('id')] = nm
    out = []
    for p in root.iter():
        if _xml_local(p.tag) == 'part':
            pid = p.get('id') or ''
            out.append((pid, names.get(pid, ''), p))
    return out


def _has_lyrics(part: ET.Element) -> bool:
    return any(_xml_local(e.tag) == 'lyric' for e in part.iter())


def parse_musicxml(path: Path, part: Optional[str] = None,
                   warn: bool = True) -> Tuple[List[ScoreNote], float]:
    """MusicXML（単旋律・声部 1）→ (ScoreNote 列, bpm)。

    `.mxl`（圧縮）も読む。複数パートがあるときは歌詞のある最初のパートを選ぶ
    （`part` に id か part-name を渡せば明示指定）。同じパート内の複数声部・繰り返し・
    途中のテンポ変更は**未対応**で、見つかったら警告を出す。
    """
    path = Path(path)
    if path.suffix.lower() == '.mxl':
        root = _mxl_root(path)
    else:
        root = ET.parse(str(path)).getroot()

    def find(el: ET.Element, name: str) -> Optional[ET.Element]:
        for c in el:
            if _xml_local(c.tag) == name:
                return c
        return None

    def note(msg: str):
        if warn:
            print(f'  [warn] {path.name}: {msg}')

    # ── パートを 1 つ選ぶ。全 <note> を拾うと伴奏まで歌ってしまう ──────────────
    parts = _xml_parts(root)
    if not parts:
        raise ValueError(f'{path.name}: <part> がありません')
    if part is not None:
        sel = [p for p in parts if part in (p[0], p[1])]
        if not sel:
            raise ValueError(f'{path.name}: part {part!r} が見つかりません（'
                             + ', '.join(f'{i}={n!r}' for i, n, _ in parts) + '）')
        chosen = sel[0]
    else:
        with_lyric = [p for p in parts if _has_lyrics(p[2])]
        chosen = (with_lyric or parts)[0]
    if len(parts) > 1:
        note(f'{len(parts)} パートあります。歌うのは {chosen[0]}'
             + (f'（{chosen[1]}）' if chosen[1] else '')
             + ' だけです。別のパートなら part= で指定してください')
    part_el = chosen[2]

    # ── 未対応の記法を報告する（黙って違う音を出さない）────────────────────────
    voices = {e.text.strip() for e in part_el.iter()
              if _xml_local(e.tag) == 'voice' and e.text}
    if len(voices) > 1:
        note(f'声部が {len(voices)} つあります（未対応）。全部つなげて読むので譜面と違う音になります')
    if any(_xml_local(e.tag) in ('backup', 'forward') for e in part_el.iter()):
        note('<backup>/<forward> があります（未対応）。時刻がずれます')
    if any(_xml_local(e.tag) == 'repeat' for e in part_el.iter()):
        note('繰り返し記号があります（未対応）。展開せず、書かれた順に 1 回だけ歌います')

    divs = [int(e.text) for e in part_el.iter()
            if _xml_local(e.tag) == 'divisions' and e.text]
    if len(set(divs)) > 1:
        note(f'divisions が途中で変わります {sorted(set(divs))}（未対応）。最初の値を使います')
    divisions = divs[0] if divs else 1

    tempos = [float(e.get('tempo')) for e in part_el.iter()
              if _xml_local(e.tag) == 'sound' and e.get('tempo')]
    if not tempos:                              # <sound tempo> が無ければ metronome から
        for e in part_el.iter():
            if _xml_local(e.tag) == 'metronome':
                pm = find(e, 'per-minute')
                if pm is not None and pm.text:
                    try:
                        tempos.append(float(pm.text.strip()))
                    except ValueError:
                        pass
    if len(set(tempos)) > 1:
        note(f'テンポが途中で変わります {sorted(set(tempos))}（未対応）。最初の値を使います')
    if not tempos:
        note('テンポの指定がありません。120 とみなします')
    bpm = tempos[0] if tempos else 120.0

    # ── ノート ───────────────────────────────────────────────────────────────
    notes: List[ScoreNote] = []
    tied_in = False          # 直前のノートから tie で繋がっているか
    for n in part_el.iter():
        if _xml_local(n.tag) != 'note':
            continue
        # 和音構成音・装飾音（duration 無し）はスキップ（単旋律前提）。
        if find(n, 'chord') is not None or find(n, 'grace') is not None:
            continue
        dur_el = find(n, 'duration')
        if dur_el is None:
            continue
        dur_beats = float(dur_el.text) / divisions
        is_rest = find(n, 'rest') is not None
        if is_rest:
            notes.append(ScoreNote(dur_beats=dur_beats, midi=0, lyric=None, is_rest=True))
            tied_in = False
            continue
        pitch = find(n, 'pitch')
        midi = _xml_pitch_to_midi(pitch) if pitch is not None else 0
        # タイは 1 つのノートに畳む。畳まないと伸ばしの途中で音素が切り直され、
        # 余分なノート境界がピッチ補正にも入る。
        ties = {e.get('type') for e in n if _xml_local(e.tag) == 'tie'}
        if tied_in and notes and not notes[-1].is_rest and notes[-1].midi == midi:
            notes[-1] = replace(notes[-1], dur_beats=notes[-1].dur_beats + dur_beats)
            tied_in = 'start' in ties
            continue
        lyric_el = n.find('.//{*}text')
        lyric = lyric_el.text.strip() if (lyric_el is not None and lyric_el.text) else None
        notes.append(ScoreNote(dur_beats=dur_beats, midi=midi, lyric=lyric, is_rest=False))
        tied_in = 'start' in ties
    return notes, bpm


# ── 音素解決・フレーズ分割 ──────────────────────────────────────────────────

def _last_nucleus(phonemes: List[str]) -> Optional[str]:
    for p in reversed(phonemes):
        if _is_nucleus_ph(p):
            return p
    return None


def _tokenize_lyric(lyric: str, mora_table: Dict[str, List[str]],
                    maxlen: int, prev_vowel: str = 'a', phoneset=None) -> List[str]:
    """1 ノートの歌詞をモーラ表で最長一致分割し、音素列を連結する。

    1 ノートに複数モーラが載る場合（例「いっ」= い＋っ → i cl）に対応。
    歌詞の途中に現れる伸ばし棒（ー ｰ - +）は直前の核音素を繰り返す
    （例「のーと」→ n o o t o）。モーラ表がその文字を定義していればそちらを優先する。
    """
    phs: List[str] = []
    i = 0
    n = len(lyric)
    while i < n:
        matched = None
        for L in range(min(maxlen, n - i), 0, -1):
            sub = lyric[i:i + L]
            if sub in mora_table:
                matched = (sub, L)
                break
        if matched is None and lyric[i] in _melisma_lyrics(phoneset):
            phs.append(_last_nucleus(phs) or prev_vowel)     # 伸ばし棒 = 直前の核を伸ばす
            i += 1
            continue
        if matched is None:
            raise KeyError(
                f'歌詞 {lyric!r} の {lyric[i]!r} 付近がモーラ表にありません'
                f'（phonemes を明示するか表に追加してください）')
        phs.extend(mora_table[matched[0]])
        i += matched[1]
    return phs


def _as_phoneme_list(lyric: str, phoneset=None) -> Optional[List[str]]:
    """歌詞が音素列そのものなら、その列を返す。そうでなければ None。

    空白で割って**全トークンが音素表にある**ときだけ音素列とみなす。英語のように
    汎用のモーラ表を持たない言語で、歌詞欄に `k l ow s` と直接書けるようにするため。
    曲ごとに単語表を作らずに 1 曲を鳴らせる。

    日本語と衝突しない。モーラは仮名・音素はローマ字なので原理的に重ならず、実際に
    重なるのは `br` と `pau` の 2 つだけ（どちらも音素でもモーラでも同じ無音なので
    どちらに解釈しても結果が変わらない）。
    """
    toks = lyric.split()
    if not toks:
        return None
    ps = phoneset or _ps_mod.active()
    known = set(ps.phonemes)
    return toks if all(t in known for t in toks) else None


def _resolve_phonemes(sn: ScoreNote, mora_table: Dict[str, List[str]],
                      maxlen: int, prev_vowel: str, phoneset=None) -> List[str]:
    if sn.phonemes is not None:
        return list(sn.phonemes)
    lyr = sn.lyric
    if lyr is None or lyr in _melisma_lyrics(phoneset):   # メリスマ = 前音節の母音を引き延ばす
        return [prev_vowel]
    direct = _as_phoneme_list(lyr, phoneset)              # 歌詞欄に音素を直接書いた場合
    if direct is not None:
        return direct
    return _tokenize_lyric(lyr, mora_table, maxlen, prev_vowel, phoneset)


def _apply_g2p(score_notes: List[ScoreNote], phoneset=None) -> List[ScoreNote]:
    """言語パックが g2p: を持つ言語（韓国語）で、歌詞の音素を先に決めておく。

    韓国語は前後の音節で発音が変わる（連音・鼻音化・濃音化など）ので、1 ノートずつは引けない。
    休符を挟まずに続くノートの並び（休符・音素を直接書いたノート・ハングルの無い歌詞で区切る）ごとに
    G2P をかけ、ノートごとの音素を ScoreNote.phonemes に入れる。1 ノートに複数の音節があってもよい。
    メリスマ（歌詞なし・伸ばし記号）のノートは、直前の音節の母音を伸ばし、その音節の終声は
    メリスマの最後のノートへ移す（「밤 -」→ b a | a M。終声は音を伸ばし終えてから閉じる）。
    """
    ps = phoneset or _ps_mod.active()
    if ps.g2p != 'ko':
        return score_notes
    from .g2p_ko import lyric_syllables, syllables_to_phonemes
    mel = _melisma_lyrics(ps)
    out = list(score_notes)

    def kind(sn: ScoreNote) -> str:
        if sn.is_rest:
            return 'break'
        if sn.phonemes is not None:
            return 'break'
        if sn.lyric is None or sn.lyric in mel or sn.lyric == '':
            return 'mel'
        if _as_phoneme_list(sn.lyric, ps) is not None or not lyric_syllables(sn.lyric):
            return 'break'
        return 'syl'

    def run_g2p(idx: List[int]) -> None:
        syls, owner = [], []
        for i in idx:
            if kind(out[i]) == 'syl':
                s = lyric_syllables(out[i].lyric)
                syls += s
                owner += [i] * len(s)
        if not syls:
            return
        per = syllables_to_phonemes(syls)
        phs = {i: [] for i in idx if kind(out[i]) == 'syl'}
        for i, p in zip(owner, per):
            phs[i] += p
        last_syl = None                          # メリスマが伸ばす直前の音節のノート
        k = 0
        while k < len(idx):
            i = idx[k]
            if kind(out[i]) == 'syl':
                last_syl = i
                k += 1
                continue
            j = k                                # メリスマが続く範囲 [k, j)
            while j < len(idx) and kind(out[idx[j]]) == 'mel':
                j += 1
            if last_syl is not None:
                src = phs[last_syl]
                nuc = max((m for m, p in enumerate(src) if ps.is_nucleus(p)), default=None)
                if nuc is not None:
                    vowel, coda = src[nuc], src[nuc + 1:]
                    phs[last_syl] = src[:nuc + 1]
                    for m in range(k, j):
                        phs[idx[m]] = [vowel] + (coda if m == j - 1 else [])
            k = j
        for i, p in phs.items():
            out[i] = replace(out[i], phonemes=p)

    run: List[int] = []
    for i, sn in enumerate(out):
        if kind(sn) == 'break':
            run_g2p(run)
            run = []
        else:
            run.append(i)
    run_g2p(run)
    return out


def score_to_phrases(
    score_notes: List[ScoreNote],
    bpm: float,
    mora_table=None,                       # dict | パス | None（既定 dict/ja.mora）
    split_silence_sec: float = 0.5,
    pau_lead_sec: float = 0.25,
    pau_trail_sec: float = 0.25,
    phoneset=None,
) -> List[Phrase]:
    """ScoreNote 列 → Phrase 列。

    * 休符が split_silence_sec 以上 → フレーズ境界。未満 → 直前ノートの gap_after_sec。
    * 歌詞→音素はモーラ表。歌詞なし/メリスマ記号 → 直前母音の引き延ばし。
    """
    ps = phoneset or _ps_mod.active()
    mora_table = _as_mora_table(mora_table, ps)
    maxlen = max((len(k) for k in mora_table), default=1)
    score_notes = _apply_g2p(score_notes, ps)     # 韓国語など、規則の G2P を持つ言語だけ
    b2s = 60.0 / bpm

    phrases: List[Phrase] = []
    cur: List[Note] = []
    prev_vowel = _melisma_vowel(ps)
    cursor = 0.0               # 楽譜上の絶対時刻 [s]（休符含め全ノートを積算）
    phrase_start = 0.0         # 現フレーズ先頭ノートの絶対開始時刻

    def flush() -> None:
        nonlocal cur
        if cur:
            cur[-1].gap_after_sec = 0.0          # フレーズ末尾は trailing pau が担う
            phrases.append(Phrase(notes=cur,
                                  leading_pau_sec=pau_lead_sec,
                                  trailing_pau_sec=pau_trail_sec,
                                  start_sec=phrase_start))
            cur = []

    for sn in score_notes:
        dur_s = sn.dur_beats * b2s
        if sn.is_rest:
            if dur_s >= split_silence_sec:
                flush()                          # 長い休符 = フレーズを切る
            elif cur:
                cur[-1].gap_after_sec += dur_s   # 短い休符 = 直前ノートの gap（先頭休符は無視）
            cursor += dur_s
            continue
        if not cur:                              # フレーズ先頭ノート = 絶対開始時刻を記録
            phrase_start = cursor
        phs = _resolve_phonemes(sn, mora_table, maxlen, prev_vowel, ps)
        cur.append(Note(phonemes=phs, pitch_midi=sn.midi,
                        dur_sec=dur_s, gap_after_sec=0.0))
        nuc = _last_nucleus(phs)
        if nuc is not None:
            prev_vowel = nuc
        cursor += dur_s

    flush()
    return phrases


# ── 便利関数（拡張子で自動判別）────────────────────────────────────────────

def load_score(path: Path, mora_table=None, phoneset=None, lang=None,
               part=None, **kwargs) -> List[Phrase]:
    """Read a score (.ust, .musicxml or .mxl, by extension) into a list of phrases.

    Args:
        path: the score file. ``.mxl`` is the compressed MusicXML that notation software
            such as MuseScore saves by default; it is unpacked here.
        mora_table: lyric → phoneme table (a dict, a path, or None for the language
            pack's ``mora:`` file).
        phoneset: a ``PhoneSet`` to use directly (mutually exclusive with ``lang``).
        lang: language pack name ('ja', 'en', or a path to a dict/*.yaml).
        part: which MusicXML part to sing, by part id or part-name. The default picks the
            first part that has lyrics, so a score with a piano accompaniment works.
        **kwargs: passed to ``score_to_phrases`` — ``split_silence_sec`` (a rest at least
            this long starts a new phrase, default 0.5) and ``pau_lead_sec`` /
            ``pau_trail_sec`` (silence added at both ends of a phrase, default 0.25).

    Returns:
        ``List[Phrase]`` — each with its notes (phonemes, pitch, length, gap) and the
        absolute start time of its first note.
    """
    ps = phoneset or (_ps_mod.load(lang) if lang else _ps_mod.active())
    path = Path(path)
    ext = path.suffix.lower()
    if ext == '.ust':
        notes, bpm = parse_ust(path, phoneset=ps)
    elif ext in ('.musicxml', '.xml', '.mxl'):
        notes, bpm = parse_musicxml(path, part=part)
    else:
        raise ValueError(f'未対応の楽譜形式: {ext}（.ust / .musicxml / .mxl のみ）')
    return score_to_phrases(notes, bpm, mora_table=mora_table, phoneset=ps, **kwargs)
