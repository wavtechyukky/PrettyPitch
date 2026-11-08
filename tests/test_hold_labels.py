"""tools/hold_labels/rules.py — which vowels to split, where, and how to write them (no GAME needed)."""
import numpy as np
import pytest

from tools.hold_labels import rules

FR = 44100 / 256


def _step_cents(n_pre, n_post, lo=0.0, hi=200.0):
    return np.r_[np.full(n_pre, lo), np.full(n_post, hi)]


def test_best_split_diff_finds_the_step_and_ignores_short_or_flat_vowels():
    assert rules.best_split_diff(_step_cents(30, 30), FR) == pytest.approx(200.0)
    assert rules.best_split_diff(np.full(60, 100.0), FR) == 0.0
    assert rules.best_split_diff(_step_cents(3, 3), FR) == 0.0          # each side under 40 ms


def test_clusters_merge_close_boundaries_and_count_seeds():
    got = rules.clusters([[1.00, 2.0], [1.01], [1.02, 3.0]], 0.03)
    assert [(round(t, 3), n) for t, n in got] == [(1.01, 3), (2.0, 1), (3.0, 1)]


def test_keep_apart_drops_boundaries_near_the_edges_or_each_other():
    assert rules.keep_apart([1.05, 1.5, 1.55, 1.95], 1.0, 2.0, 0.08) == [1.5]


def test_steps_keep_only_real_pitch_changes():
    t = np.arange(120) / FR                                   # a 0.7 s vowel, step at 0.35 s
    cents = np.where(t < 0.35, 0.0, 150.0)
    voiced = np.ones_like(cents, bool)
    assert rules.step_at(cents, voiced, FR, 0.35, 0.0, 0.7) == pytest.approx(150.0)
    assert rules.keep_steps([10.35], 10.0, 0.0, 0.7, cents, voiced, FR, 80.0) == [10.35]
    flat = np.zeros_like(cents)
    assert rules.keep_steps([10.35], 10.0, 0.0, 0.7, flat, voiced, FR, 80.0) == []
    assert rules.step_at(cents, np.zeros_like(voiced), FR, 0.35, 0.0, 0.7) is None   # nothing voiced


def _vowel(diff=150.0, seeds=((0.35,), (0.351,), (0.36,), (), ())):
    return {"j": 3, "row": 7, "ph": "ay", "v0": 0.0, "v1": 0.7, "l0": 10.0, "l1": 10.7, "diff": diff,
            "seeds": [list(s) for s in seeds]}


def test_choose_splits_applies_every_rule():
    t = np.arange(120) / FR
    f0 = (np.where(t < 0.35, 0.0, 150.0), np.ones(120, bool))
    p = rules.Params()
    ts, src, dropped = rules.choose_splits(_vowel(), 5, p, f0, FR)
    assert src == "game" and dropped == 0 and len(ts) == 1 and ts[0] == pytest.approx(10.351, abs=1e-3)
    assert rules.choose_splits(_vowel(diff=10.0), 5, p, f0, FR)[0] == []                 # not a candidate
    assert rules.choose_splits(_vowel(seeds=((0.35,), (), (), (), ())), 5, p, f0, FR)[0] == []   # minority
    flat = (np.zeros(120), np.ones(120, bool))
    assert rules.choose_splits(_vowel(), 5, p, flat, FR)[1:] == ("game", 1)             # no step: dropped
    short = dict(_vowel(), v1=0.15, l1=10.15)
    assert rules.choose_splits(short, 5, p, f0, FR)[0] == []                             # 150 ms ≤ min_len


def test_overrides_replace_game_but_keep_the_length_rule():
    p = rules.Params()
    split = {"decision": "split", "splits_sec": [0.4, 0.2]}
    assert rules.choose_splits(_vowel(), 5, p, override=split)[:2] == ([10.2, 10.4], "override")
    assert rules.choose_splits(_vowel(), 5, p, override={"decision": "keep"})[0] == []
    short = dict(_vowel(), v1=0.15, l1=10.15)
    assert rules.choose_splits(short, 5, p, override=split)[0] == []


def test_split_lab_lines_writes_hold_rows_in_the_lab_unit():
    lines = ["0 1000000 pau", "", "1000000 8000000 ay", "8000000 9000000 z"]
    assert rules.split_lab_lines(lines, {1: [0.35, 0.5]}, "100ns") == [
        "0 1000000 pau", "1000000 3500000 ay", "3500000 5000000 -", "5000000 8000000 -", "8000000 9000000 z"]
    sec = ["0.0000000 0.1000000 pau", "0.1000000 0.8000000 a"]
    assert rules.split_lab_lines(sec, {1: [0.35]}, "sec") == [
        "0.0000000 0.1000000 pau", "0.1000000 0.3500000 a", "0.3500000 0.8000000 -"]


def test_insert_holds_keeps_frames_and_untouched_phonemes():
    ids = np.array([0, 5, 9, 0], np.int16)
    dur = np.array([20, 10, 60, 25], np.float32) / FR
    new_ids, new_dur = rules.insert_holds(ids, dur, FR, {2: [50, 70, 30]}, hold_id=54)
    assert new_ids.tolist() == [0, 5, 9, 54, 54, 0] and new_ids.dtype == ids.dtype
    assert np.round(new_dur * FR).astype(int).tolist() == [20, 10, 20, 20, 20, 25]
    assert new_dur[0] == dur[0] and new_dur[-1] == dur[-1]                              # copied as they were
    same_ids, same_dur = rules.insert_holds(ids, dur, FR, {2: [30, 90]}, hold_id=54)  # cuts on the edges
    assert same_ids.tolist() == ids.tolist() and np.array_equal(same_dur, dur)
