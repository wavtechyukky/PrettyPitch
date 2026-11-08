"""持続長の後処理（score-free・多音素対応）。

子音長モデルが各子音に付けた「自然長」を、ノート長・ノート間ギャップ・floor（最低保証）
に照らして最終的な音素列＋各音素の秒数へ落とし込む。**1 ノートに複数の子音**
（onset クラスタ・coda）が来ても同じ規則で扱う。

── 時間配置の規則（例: close/to/you = k l / ow s t / uw y / uw）──
  各ノートは音素列を [onset 子音…][母音…][coda 子音…] に分ける。
  * onset クラスタ … そのノートの開始時刻より前へ「追い出され」、前ノート末（先頭ノートは
    leading pau）へ後ろ向きに食い込む。
  * coda 子音     … そのノートの区間内・母音の後ろに残る。
  * 次ノートの onset … 現ノート末へ overlap で食い込む（gap をまず消費し、足りない分だけ母音を削る）。
  よって「ノート j の区間に現れる音素」= 母音_j + coda_j + onset_(j+1) となる。

── floor / 圧縮（リージョン方式）──
  各子音は「自分が食い込む先のノート区間」で 1 度だけ floor/圧縮される。
    * onset_(j+1) と coda_j は "ノート j の区間" を奪い合う → ノート j で処理。
    * onset_0（先頭ノートの onset）は leading pau を奪う → 先頭交渉で処理。
  これで各子音の重複処理が起きない。

規則: **なるべく floor を保証**（子音は自然長、ただし母音に v_min を残す
ところまで）し、**母音 floor + 子音 floor がノート長に入らないときだけ重み比で按分**する。
母音 floor は profile.v_min(spk, style)（既定 75ms・話者別可）、子音 floor は profile.c_min
（子音別・平均−1σ 統計・話者別可）、重みは profile.weight（音素ごとの取り分・
音素別/話者別可・既定 1.0 ＝ 全音素が等分）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from duration_model import phoneset as _ps_mod
from duration_model.profile import (ConsDurProfile, default_profile,
                                    _DEFAULT_PHONEME_WEIGHT)

# 核（ピッチ/尺を担う）音素の判定は言語パック（dict/<lang>.yaml の nucleus:）が決める。
# ここに音素を直書きしない。核でないもの（未知音素を含む）は子音＝onset/coda 扱いになる
# （英語子音 l/s/t 等を取りこぼさないため）。
# cl（促音の閉鎖）は日本語パックで **核に含めていない**。F0 側は ckpt の nucleus を見るので、
# ここで核扱いすると持続長側と F0 側で定義が食い違う。


def _ps(phoneset=None):
    return phoneset or _ps_mod.active()


def _is_pau(ph: str, ps=None) -> bool:
    """無音トークンか（言語パックの silence で決まる）。"""
    return _ps(ps).is_pau(ph)


def _is_nucleus_ph(ph: str, ps=None) -> bool:
    """ph が核（母音的・ピッチや尺を担う）か。onset/coda 分割の境界に使う。"""
    return _ps(ps).is_nucleus(ph)


def is_nucleus_ph(ph: str, phoneset=None) -> bool:
    """公開名。1 引数呼び出し（既定の言語）も従来どおり使える。"""
    return _ps(phoneset).is_nucleus(ph)


def _is_edge_consonant(ph: str, ps=None) -> bool:
    """onset/coda へ押し出す実子音か（pau・核は除く）。"""
    p = _ps(ps)
    return not p.is_pau(ph) and not p.is_nucleus(ph)

# 子音長モデルが無い／その子音の予測が欠けているときに使う 1 子音あたりの長さ [s]。
FALLBACK_CONS_DUR = 0.08
# 母音長の下限（圧縮しても 0 にはしない安全値）。
_NUC_EPS = 0.005
# ノート間 gap の残りがこれ未満なら pau を置かない [s]。1〜2 フレーム（hop256/44.1k で
# 5.80ms/frame）しか無い無音は音素として成立せず、学習データの内部 pau の最小値
# （日本語 11.6ms / 英語 5.8ms）も下回る。消した分の時間は直前の核へ返すので尺は保存される。
PAU_DROP_SEC = 0.012
# PrettyPitch の音響 vocab（dict/ja.phonemes）は id 0 = 'pau'（小文字）。
PAU = 'pau'


# ── 入力データ構造 ────────────────────────────────────────────────────────────

@dataclass
class Note:
    """1 ノート。phonemes は onset…母音…coda の並び（正規化前でも可）。"""
    phonemes: List[str]
    pitch_midi: int
    dur_sec: float
    gap_after_sec: float = 0.0
    # 楽譜で伸ばし・音節の続きの記号（歌詞なし・-・ー・+ で始まる歌詞）だった音符。
    # 前の音符の母音をそのまま続けるかどうか（PhonemeSequence.held）を決めるのに使う
    melisma: bool = False


@dataclass
class Phrase:
    """合成単位。先頭・末尾 pau 込み。"""
    notes: List[Note]
    leading_pau_sec: float = 0.1
    trailing_pau_sec: float = 0.1
    start_sec: float = 0.0     # フレーズ先頭ノートの楽譜上の絶対開始時刻（タイムライン配置用）


@dataclass
class PhonemeSequence:
    """後処理の出力。音響モデルへ渡すフラットな音素列。"""
    phonemes: List[str]              # 例: ['PAU','k','l','ow','s','t','uw', ...]
    dur_sec: np.ndarray              # 各音素の秒数 [L]
    note_midi: np.ndarray            # 各音素の MIDI ピッチ [L]（pau=0）
    is_vowel: np.ndarray             # 母音（＝ピッチを担う）か [L] bool
    note_event_starts: np.ndarray    # ノートイベント先頭スロットで True [L] bool
    leading_pau_sec: float           # 先頭交渉後の leading pau [s]
    # 伸ばしの音符で、前の音符の母音がそのまま続くスロットで True [L] bool（「skies」「-」の 2 つ目の ay）。
    # 伸ばしの記号で学習した F0 モデルにはここを記号で渡し、--merge_held_vowels では音響モデルへ前の母音とまとめて渡す
    held: Optional[np.ndarray] = None


# ── 内部作業構造（グループ分解＋自然長／確定長）───────────────────────────────

@dataclass
class _Syl:
    onset: List[str]
    nucleus: List[str]
    coda: List[str]
    pitch_midi: int
    dur_sec: float
    gap_after_sec: float
    onset_nat: List[float] = field(default_factory=list)   # 自然長（子音長モデル由来）
    coda_nat: List[float] = field(default_factory=list)
    # 核を持たないノート（子音だけのノート）の自然長。無ければ等分にフォールバックする。
    core_nat: List[float] = field(default_factory=list)
    onset_dur: List[float] = field(default_factory=list)   # 圧縮・floor 後の確定長
    coda_dur: List[float] = field(default_factory=list)
    melisma: bool = False                                  # Note.melisma（ノートの最初のグループだけ）


# ── グループ分解（核を最大 1 つ持つ単位へ）─────────────────────────────────────

def split_syllable(phonemes: Sequence[str], phoneset=None) -> Tuple[List[str], List[str], List[str]]:
    """音素列を (onset, nucleus, coda) に分ける。

    onset = 先頭の連続子音、coda = 末尾の連続子音、nucleus = その間（母音を含む核）。

    核が 1 つも無いとき（子音のみ・cl のみ等）は ([], phs, []) を返し、音素を region 内に
    留める。onset に押し出すと前ノートへ食い込んだうえ region が pau で埋められ、実音素が
    出力から消えてしまうため。
    """
    # 無音は 'pau' に、無声化母音等は phoneset の正規化に合わせる（vocab を安定させる）。
    p_ = _ps(phoneset)
    phs = [p_.pau if p_.is_pau(x) else p_.normalize_ph(x) for x in phonemes]
    if phs and not any(p_.is_nucleus(x) or p_.is_pau(x) for x in phs):
        return [], phs, []
    n = len(phs)
    i = 0
    while i < n and _is_edge_consonant(phs[i], p_):
        i += 1
    j = n
    while j > i and _is_edge_consonant(phs[j - 1], p_):
        j -= 1
    return phs[:i], phs[i:j], phs[j:]


def _syllabify(phonemes: Sequence[str], phoneset=None) -> List[List[str]]:
    """音素列をグループに分ける（核を最大 1 つ＋その前後の子音 = 1 グループ）。

    1 ノートに複数モーラが入っている場合（例「かか」= k a k a）に、ノートをグループ数で等分して
    「1 モーラ 1 ノートが並んでいる場合」と同じ扱いにするための前処理。
    核が 0〜1 個なら分割しない（子音のみ・cl のみ・pau のみ・空も安全に 1 グループ）。
    核を持たないグループもあるので、返るものが全て音節とは限らない。
    """
    phs = list(phonemes)
    p_ = _ps(phoneset)
    core = [i for i, p in enumerate(phs) if p_.is_nucleus(p) or p_.is_pau(p)]
    if len(core) <= 1:
        return [phs]
    starts = [0]
    for a, b in zip(core, core[1:]):
        j = b                       # b の直前に続く子音列は b 側の onset として次グループへ
        while j > a + 1 and not (p_.is_nucleus(phs[j - 1]) or p_.is_pau(phs[j - 1])):
            j -= 1
        starts.append(j)
    return [phs[x:y] for x, y in zip(starts, starts[1:] + [len(phs)])]


def split_phrase_syllables(phrase: Phrase, phoneset=None) -> Phrase:
    """多モーラノートを `_syllabify` のグループごとのノートへ展開した Phrase を返す。

    dur はノートをグループ数で等分、gap_after_sec は最終グループだけが持つ
    （= postprocess_durations が内部で行う分割と同一）。分割済みノートは _syllabify が
    1 グループを返すので**冪等**で、これを postprocess_durations へ渡しても結果は変わらない。
    子音長モデルへ「グループごと」の自然長を渡したいときに使う。
    """
    notes: List[Note] = []
    for note in phrase.notes:
        groups = _syllabify(note.phonemes, phoneset)
        m = len(groups)
        for gi, g in enumerate(groups):
            notes.append(Note(phonemes=list(g), pitch_midi=note.pitch_midi,
                              dur_sec=note.dur_sec / m,
                              gap_after_sec=note.gap_after_sec if gi == m - 1 else 0.0,
                              melisma=note.melisma and gi == 0))
    return Phrase(notes=notes, leading_pau_sec=phrase.leading_pau_sec,
                  trailing_pau_sec=phrase.trailing_pau_sec, start_sec=phrase.start_sec)


# ── floor / 圧縮ヘルパ ──────────────────────────────────────────────────────

def _floor_each(cons: Sequence[str], nat: Sequence[float],
                profile: ConsDurProfile, lead: bool = False,
                spk_id=None, style_id=None) -> List[float]:
    """各子音を自然長と c_min の大きい方に。lead=True は先頭 floor(lead_c_min)。

    spk_id / style_id を渡すと話者別 floor（profile.per_speaker）を優先する。
    その話者にその音素の統計が無ければ全話者共通値へ落ちる（profile._floor 参照）。
    """
    out: List[float] = []
    for c, x in zip(cons, nat):
        floor = (profile.lead_c_min(c, spk_id, style_id) if lead
                 else profile.c_min(c, spk_id, style_id))
        out.append(max(float(x), floor))
    return out


def _scale_to_total(nat: Sequence[float], total: float) -> List[float]:
    """自然長の比を保ったまま合計を total に合わせる。"""
    s = float(sum(nat))
    if s <= 0.0:
        # 全て 0 → 均等割り
        k = len(nat)
        return [total / k] * k if k else []
    return [x / s * total for x in nat]


def _lead_pau_cons(cons_nat: float, space: float,
                   pau_target: float) -> Tuple[float, float]:
    """先頭 onset クラスタ×直前余白 space から (leading_pau, onset 合計長) を決める。

    **自然な子音長を最優先**する。子音は常に min(自然長, space) を受け取り、pau は
    その余りを pau_target まで。削られるのは余白そのものが足りないときだけ。

      space >= cons_nat + pau_target : pau = pau_target,      子音 = 自然長
      space >= cons_nat              : pau = space - cons_nat, 子音 = 自然長
      space <  cons_nat              : pau = 0,                子音 = space

    space について連続かつ単調で、合計が space を超えない。pau に下限を置いて子音を削る
    形にしないのは、「先頭だから長くすべき」と決めた先頭 floor を pau のために削ることになり
    自己矛盾するため。
    """
    cons_nat = max(0.0, float(cons_nat))
    space = max(0.0, float(space))
    if space <= 0.0:
        return 0.0, 0.0
    cons = min(cons_nat, space)
    return min(float(pau_target), space - cons), cons


def _compress_region(dur: float, gap: float,
                     coda_nat: List[float], onset_next_nat: List[float],
                     profile: ConsDurProfile,
                     coda_floor: Optional[List[float]] = None,
                     onset_next_floor: Optional[List[float]] = None,
                     coda_weight: Optional[List[float]] = None,
                     onset_next_weight: Optional[List[float]] = None,
                     nucleus_weight: float = None,
                     v_min: float = None,
                     ) -> Tuple[List[float], List[float]]:
    """ノート区間 dur を奪い合う coda_j と onset_(j+1) を確定長へ。

      (1) 母音 floor + 子音 floor の合計 ≤ dur  … floor を保証する。
          子音は自然長のまま、ただし合計が dur − v_min を超えるならそこまで縮める
          （母音に v_min を残す）。縮めても各子音は自分の floor を割らない。
      (2) それが入らない（v_min + Σc_min > dur）… 重み比で按分する。
          区間に居る音素（子音と母音）の重みの合計で区間を割った 1 単位を、それぞれの重み
          ぶん取る。全音素が既定 1.0 なら等分になる。

    **(2) で floor の比を配分に使ってはいけない**（c_min は分布の下限で、配分の重みでは
    ない。摩擦音は下限が高く破裂音は低いので破裂音だけが構造的に潰れる）。

    次の onset が gap に丸ごと収まる（overlap = 0）ときは、その onset はこの区間を
    奪っていないので配分の対象外で、floor 済み自然長のまま残る。

    coda_floor / onset_next_floor には floor そのもの、v_min には母音 floor
    （profile.v_min(spk, style)）を渡すこと。戻り値 = (coda_dur, onset_next_dur)。
    """
    O_next = float(sum(onset_next_nat))
    overlap = max(0.0, O_next - gap)            # 次 onset が dur を食う量
    coda_total = float(sum(coda_nat))
    demand = coda_total + overlap               # dur 内で子音が占める時間（自然長）

    if demand <= 0.0:
        return list(coda_nat), list(onset_next_nat)

    if profile.enabled is False:
        # floor 無効：母音を食い尽くさない頭打ちのみ（子音は自然長）。
        room = max(0.0, dur - _NUC_EPS)
        if demand <= room:
            return list(coda_nat), list(onset_next_nat)
        return _shrink(coda_nat, onset_next_nat, gap, overlap, room / demand)

    vm = float(profile.vowel_floor if v_min is None else v_min)
    coda_b = list(coda_floor) if coda_floor is not None else list(coda_nat)
    onset_b = list(onset_next_floor) if onset_next_floor is not None else list(onset_next_nat)
    overlap_b = max(0.0, float(sum(onset_b)) - gap)
    floor_total = float(sum(coda_b)) + overlap_b   # dur 内の子音 floor の合計

    if floor_total + vm <= dur:
        # (1) floor を保証できる: 子音は自然長、ただし母音に v_min を残す所まで
        cap = dur - vm
        if demand <= cap:
            return list(coda_nat), list(onset_next_nat)
        # 自然長を一律に縮め、floor を割る子音はそこで止める（止めた分は他へ）
        coda_dur, onset_dur = _shrink_to(coda_nat, onset_next_nat, coda_b, onset_b,
                                         gap, overlap, cap)
        return coda_dur, onset_dur

    # (2) floor が入らない: 区間を奪い合う音素の重み比で按分する。
    #
    # 区間に居る音素（子音と母音）の重みを w として
    #   1 単位 = dur / Σ w、音素 i = w_i × 1 単位。
    # 全音素が既定 1.0 なら等分になる。
    #
    # ★旧実装は「子音の合計 = ratio × dur、内訳は floor の比」だった。子音の取り分の総量が
    #   本数に依らず頭打ちになるので、子音が並ぶときだけ 1 本あたりが潰れた
    #   （100ms に s+cl+k で k が 8.9ms → 重み比なら 25ms）。重みは「母音に対して何倍取るか」
    #   であって割合ではないので、相手が増えれば区間の取り分も増える。
    n_c = len(coda_nat)
    n_o = len(onset_next_nat) if overlap > 0.0 else 0
    if n_c + n_o <= 0:
        return list(coda_nat), list(onset_next_nat)
    D = _DEFAULT_PHONEME_WEIGHT
    wc = list(coda_weight) if coda_weight is not None else [D] * n_c
    wo = (list(onset_next_weight) if onset_next_weight is not None
          else [D] * len(onset_next_nat))
    wc = (wc + [D] * n_c)[:n_c]
    wo = (wo + [D] * len(onset_next_nat))[:len(onset_next_nat)]
    wv = D if nucleus_weight is None else max(0.05, float(nucleus_weight))
    unit = max(0.0, dur) / (sum(wc) + sum(wo[:n_o]) + wv)      # 母音の重みも足す
    coda_dur = [unit * w for w in wc]
    if n_o > 0:
        # 次の onset は gap ぶんは只で使えるので、区間から取るのは重みぶんだけ。
        inside = unit * sum(wo)
        onset_dur = _scale_to_total([unit * w for w in wo], gap + inside)
    else:
        # 次の onset は gap に丸ごと収まっており、この区間を 1ms も奪っていない。
        onset_dur = list(onset_next_nat)
    return coda_dur, onset_dur


def _shrink_to(coda_nat: List[float], onset_next_nat: List[float],
               coda_floor: List[float], onset_next_floor: List[float],
               gap: float, overlap: float, cap: float) -> Tuple[List[float], List[float]]:
    """区間内の子音（coda 全部 ＋ onset の overlap 部分）の合計を cap まで縮める。
    各子音は自分の floor を割らない（floor で止まった分は他の子音がさらに縮む）。
    呼び出し側が Σfloor ≤ cap を保証しているので必ず収まる。"""
    if overlap <= 0.0:
        # onset は区間を奪っていない → coda だけを縮める
        return _shrink_floored(coda_nat, coda_floor, cap), list(onset_next_nat)
    nat = list(coda_nat) + [overlap]
    flo = list(coda_floor) + [max(0.0, float(sum(onset_next_floor)) - gap)]
    out = _shrink_floored(nat, flo, cap)
    coda_dur, overlap_final = out[:len(coda_nat)], out[-1]
    onset_dur = _scale_to_total(onset_next_nat, gap + overlap_final)
    return coda_dur, onset_dur


def _shrink_floored(nat: List[float], floor: List[float], cap: float) -> List[float]:
    """合計を cap にする。各要素は floor 以上を保つ（水位法）。"""
    n = len(nat)
    out = list(nat)
    fixed = [False] * n
    for _ in range(n + 1):
        free = [i for i in range(n) if not fixed[i]]
        if not free:
            break
        budget = cap - sum(out[i] for i in range(n) if fixed[i])
        tot = sum(out[i] for i in free)
        if tot <= budget or tot <= 0:
            break
        s = budget / tot
        hit = False
        for i in free:
            v = out[i] * s
            if v < floor[i]:
                out[i] = float(floor[i]); fixed[i] = True; hit = True
            else:
                out[i] = v
        if not hit:
            break
    return out


def _nucleus_weight(profile: ConsDurProfile, nucleus: List[str],
                    spk_id=None, style_id=None) -> float:
    """区間の母音側の重み。核が複数音素なら平均（区間では 1 つの塊として扱うため）。"""
    phs = [v for v in nucleus if v != PAU]
    if not phs:
        return _DEFAULT_PHONEME_WEIGHT
    return sum(profile.weight(v, spk_id, style_id) for v in phs) / len(phs)


def _shrink(coda_nat: List[float], onset_next_nat: List[float],
            gap: float, overlap: float, s: float) -> Tuple[List[float], List[float]]:
    """coda と（overlap>0 のときのみ）onset を係数 s で縮める。

    overlap==0（次 onset が gap に収まる）なら onset は自然長のまま（膨らませない）。
    """
    coda_dur = [c * s for c in coda_nat]
    if overlap > 0:
        overlap_final = overlap * s
        O_next_final = gap + overlap_final       # onset は gap + 圧縮後 overlap を張る
        onset_dur = _scale_to_total(onset_next_nat, O_next_final)
    else:
        onset_dur = list(onset_next_nat)
    return coda_dur, onset_dur


# ── メイン後処理 ───────────────────────────────────────────────────────────

def _to_syls(phrase: Phrase, ps,
             onset_nat: Optional[List[List[float]]],
             coda_nat: Optional[List[List[float]]],
             core_nat: Optional[List[List[float]]] = None) -> List[_Syl]:
    syls: List[_Syl] = []
    for k, note in enumerate(phrase.notes):
        groups = _syllabify(note.phonemes, ps)
        m = len(groups)
        for gi, g in enumerate(groups):
            onset, nucleus, coda = split_syllable(g, ps)
            # 呼び出し元が渡す自然長はノート単位（先頭 onset / 末尾 coda 相当）なので、
            # 最初のグループの onset と最後のグループの coda にだけ適用する。
            # 分割で新たに現れた内部子音は自然長が未知なのでフォールバックを使う。
            on_nat = onset_nat[k] if (onset_nat is not None and gi == 0) else None
            cd_nat = coda_nat[k] if (coda_nat is not None and gi == m - 1) else None
            if on_nat is None or len(on_nat) != len(onset):
                on_nat = [FALLBACK_CONS_DUR] * len(onset)
            if cd_nat is None or len(cd_nat) != len(coda):
                cd_nat = [FALLBACK_CONS_DUR] * len(coda)
            # 核を持たないノート（子音だけのノート）の自然長。ノート = 1 グループのときだけ対応が付く。
            co_nat = core_nat[k] if (core_nat is not None and m == 1) else None
            if co_nat is None or len(co_nat) != len(nucleus):
                co_nat = []
            syls.append(_Syl(
                onset=onset, nucleus=nucleus, coda=coda,
                pitch_midi=note.pitch_midi,
                dur_sec=note.dur_sec / m,                       # ノートをグループ数で等分
                gap_after_sec=note.gap_after_sec if gi == m - 1 else 0.0,
                onset_nat=list(on_nat), coda_nat=list(cd_nat), core_nat=list(co_nat),
                melisma=bool(getattr(note, 'melisma', False)) and gi == 0))
    return syls


def postprocess_durations(
    phrase: Phrase,
    profile: Optional[ConsDurProfile] = None,
    onset_nat: Optional[List[List[float]]] = None,
    coda_nat: Optional[List[List[float]]] = None,
    core_nat: Optional[List[List[float]]] = None,
    leading_pau_target: float = 0.25,
    rest_before: float = 1e9,
    phoneset=None,
    spk_id=None,
    style_id=None,
) -> PhonemeSequence:
    """フレーズ → 音素列＋各音素秒数（floor/圧縮適用済み）。

    onset_nat / coda_nat : ノートごとの子音自然長 [s]（子音長モデル由来）。
        None のとき全子音を FALLBACK_CONS_DUR とする。
    core_nat             : 核を持たないノート（子音だけのノート）の自然長 [s]。与えると尺を
        自然長の比で配分する。None なら等分（従来どおり）。
    rest_before          : 直前フレーズ末とのあいだの余白 [s]（先頭 onset 交渉に使用）。
        現状 svs/render.py は 1e9 を渡す（フレーズを独立に扱うので実際の余白を持たない）。
        その場合この交渉は常に「余裕あり」に落ち、leading pau = 目標値・子音 = 自然長になる。
    """
    ps = _ps(phoneset)
    profile = profile or default_profile(ps)
    syls = _to_syls(phrase, ps, onset_nat, coda_nat, core_nat)
    n = len(syls)

    # 核を持たないノート（子音だけのノート）にも floor を当てる。onset / coda と同じ順序
    # （floor → 尺合わせ）にするため、ここで core_nat を floor 済みの値に置き換える。
    # これをしないとノートが短いときに子音が下限を割る（例: sh が 80ms を下回る）。
    for x in syls:
        if x.core_nat and not any(ps.is_nucleus(v) or ps.is_pau(v) for v in x.nucleus):
            x.core_nat = _floor_each(x.nucleus, x.core_nat, profile,
                                     spk_id=spk_id, style_id=style_id)

    # ── リージョン圧縮：各ノート区間 j で coda_j と onset_(j+1) を確定 ──
    # floor 済み自然長は純関数なのでグループごとに1回だけ算出して使い回す。
    # coda_dur は下のループが全 j を、onset_dur は j>=1 をループが、j=0 を先頭交渉が
    # 確定させる（onset が空なら _Syl の既定 [] のままで正しい）。
    coda_floored = [_floor_each(x.coda, x.coda_nat, profile,
                                spk_id=spk_id, style_id=style_id) for x in syls]
    # 先頭グループの onset だけは **lead_c_min**（フレーズ頭の統計 p75）で floor する。
    # フレーズ先頭と中間では同じ音素でも持続長が別物なので、中間用の c_min（p5）を
    # 当てるのは不適切。index 0 はリージョン圧縮では使われず（あちらは j+1 のみ参照）、
    # 下の先頭ネゴだけが読む。
    onset_floored = [_floor_each(x.onset, x.onset_nat, profile, lead=(i == 0),
                                 spk_id=spk_id, style_id=style_id)
                     for i, x in enumerate(syls)]

    # floor そのもの。区間に収まらないときの配分比に使う。
    coda_floors = [[profile.c_min(c, spk_id, style_id) for c in x.coda] for x in syls]
    onset_floors = [[(profile.lead_c_min(c, spk_id, style_id) if i == 0
                      else profile.c_min(c, spk_id, style_id)) for c in x.onset]
                    for i, x in enumerate(syls)]

    for j in range(n):
        onset_next_floored = onset_floored[j + 1] if j + 1 < n else []
        onset_next_floors = onset_floors[j + 1] if j + 1 < n else []
        onset_next_phs = syls[j + 1].onset if j + 1 < n else []
        coda_dur, onset_next_dur = _compress_region(
            syls[j].dur_sec, syls[j].gap_after_sec,
            coda_floored[j], onset_next_floored, profile,
            coda_floor=coda_floors[j], onset_next_floor=onset_next_floors,
            coda_weight=[profile.weight(c, spk_id, style_id) for c in syls[j].coda],
            onset_next_weight=[profile.weight(c, spk_id, style_id)
                               for c in onset_next_phs],
            nucleus_weight=_nucleus_weight(profile, syls[j].nucleus, spk_id, style_id),
            v_min=profile.v_min(spk_id, style_id))
        syls[j].coda_dur = coda_dur
        if j + 1 < n:
            syls[j + 1].onset_dur = onset_next_dur

    # ── 先頭 onset × leading pau 交渉（onset_0 をここで確定）──
    leading_pau = phrase.leading_pau_sec
    if n > 0 and syls[0].onset:
        # onset_floored[0] は既に lead_c_min で floor 済みなので、ここでの引き上げは不要。
        onset0_floored = onset_floored[0]
        lead_pau, O0 = _lead_pau_cons(float(sum(onset0_floored)), float(rest_before),
                                      pau_target=leading_pau_target)
        leading_pau = lead_pau
        syls[0].onset_dur = _scale_to_total(onset0_floored, O0)

    # ── フラットな音素列へ組み立て（`_build_phoneme_arrays` の一般化）──
    return _build_sequence(syls, leading_pau, phrase.trailing_pau_sec, ps)


def _build_sequence(syls: List[_Syl], leading_pau: float,
                    trailing_pau: float, ps) -> PhonemeSequence:
    phonemes: List[str] = []
    durs: List[float] = []
    midis: List[int] = []
    vowels: List[bool] = []
    starts: List[bool] = []
    helds: List[bool] = []

    def append(ph: str, dur: float, midi: int, vow: bool, start: bool, held: bool = False) -> None:
        phonemes.append(ph)
        durs.append(dur)
        midis.append(midi)
        vowels.append(vow)
        starts.append(start)
        helds.append(held)

    append(PAU, leading_pau, 0, False, False)

    n = len(syls)
    for j, s in enumerate(syls):
        # onset クラスタ（先頭スロット＝ノートイベント先頭）
        for k, c in enumerate(s.onset):
            append(c, s.onset_dur[k], s.pitch_midi, False, k == 0)

        # 次 onset の食い込み量（overlap）と coda 分を差し引いた母音尺
        O_next = float(sum(syls[j + 1].onset_dur)) if j + 1 < n else 0.0
        overlap = max(0.0, O_next - s.gap_after_sec)
        coda_total = float(sum(s.coda_dur))
        # ノート間 gap のうち、次 onset に食われずに残る分。PAU_DROP_SEC 未満しか
        # 残らないなら pau を置かず、その時間を直前の核へ返す（尺は保存される）。
        effective_gap = max(0.0, s.gap_after_sec - O_next) if s.gap_after_sec > 0 else 0.0
        absorbed = 0.0
        if 0.0 < effective_gap < PAU_DROP_SEC:
            absorbed, effective_gap = effective_gap, 0.0
        nuc_total = max(_NUC_EPS, s.dur_sec - overlap - coda_total) + absorbed

        if s.nucleus:
            # 子音だけのノートは、自然長が分かっていればその比で配分する。分からなければ等分。
            # 等分だと 0.6 秒のノートに 'k s t' で破裂音 k に 200ms が割り当たり、自然長の
            # 4 倍以上になる。比例配分なら k:120 s:260 t:205 のように現実的な比率になる。
            # 実際の合計は nuc_total に一致させるので、尺は保存される（_scale_to_total と同じ考え方）。
            core_only = not any(ps.is_nucleus(v) or v == 'pau' for v in s.nucleus)
            w = None
            if core_only and s.core_nat and len(s.core_nat) == len(s.nucleus):
                # core_nat は postprocess_durations 側で floor 済み。ここは尺合わせだけ。
                w = _scale_to_total(s.core_nat, nuc_total)
            per = nuc_total / len(s.nucleus)
            first = True
            for i_v, v in enumerate(s.nucleus):
                d_v = w[i_v] if w is not None else per
                if v == 'pau':
                    append(PAU, d_v, 0, False, False)
                else:
                    is_start = first and not s.onset      # onset 無しなら最初の母音が先頭
                    # 伸ばしの音符の最初の核が、すぐ前の音素と同じ母音なら「前の母音の続き」
                    held = (is_start and s.melisma and ps.is_nucleus(v)
                            and bool(phonemes) and phonemes[-1] == v)
                    # 核でない音素（cl・子音のみのノート）に母音フラグを立てない
                    append(v, d_v, s.pitch_midi, ps.is_nucleus(v), is_start, held)
                first = False
        else:
            # 母音無し（子音のみのノート）— 尺を確保して pau で埋める（退化ケース）
            append(PAU, nuc_total, 0, False, not s.onset)

        # coda クラスタ（母音の後ろ・区間内）
        for k, c in enumerate(s.coda):
            append(c, s.coda_dur[k], s.pitch_midi, False, False)

        # ノート間 gap（次 onset が食った残りを中間 PAU で埋める）
        if effective_gap > 0:
            append(PAU, effective_gap, 0, False, False)

    append(PAU, trailing_pau, 0, False, False)

    return PhonemeSequence(
        phonemes=phonemes,
        dur_sec=np.asarray(durs, dtype=np.float32),
        note_midi=np.asarray(midis, dtype=np.int32),
        is_vowel=np.asarray(vowels, dtype=bool),
        note_event_starts=np.asarray(starts, dtype=bool),
        leading_pau_sec=leading_pau,
        held=np.asarray(helds, dtype=bool))


def to_frames(dur_sec: np.ndarray, frame_rate: float) -> Tuple[np.ndarray, int]:
    """音素秒数 → 各音素フレーム数 + 総フレーム数（`_dur_to_frames` と同一）。"""
    cum = np.concatenate([[0.0], np.cumsum(dur_sec)])
    bnd = np.round(cum * frame_rate).astype(int)
    ph_durs = np.maximum(np.diff(bnd), 1).astype(np.int32)
    return ph_durs, int(bnd[-1])
