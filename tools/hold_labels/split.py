"""音符をまたぐ母音を、GAME の音符の境界で「ay -」のように分けたラベルとシャードを作る。

3 段に分ける。GAME（torch）を使うのは run だけで、分ける条件は write で何度でも変えられる。

    # 1. GAME を全部の句にかける（止めても、もう一度走らせると続きから）
    python -m tools.hold_labels.split run --recipe configs/recipes/aidol.yaml --shard data/aidol \\
        --game_root ../GAME --lang en --out work/aidol_hold
    # 2. 分ける母音と位置を決め、lab と splits.json と、その lab を読むレシピを書く
    python -m tools.hold_labels.split write --recipe configs/recipes/aidol.yaml --shard data/aidol --out work/aidol_hold
    # 3. 前処理済みのシャードに「-」を差し込んだシャードを作る（RMVPE を回し直さない）
    python -m tools.hold_labels.split shard --shard data/aidol --splits work/aidol_hold/splits.json --out data/aidol_hold

run の入力は、元のラベルで前処理したシャード（句の音素・長さ・F0）と、レシピの wav・lab。句が曲のどこかは、
pau 以外の音素の並びが lab と一致する所で決める（合わない句は飛ばして数を出す）。
3 の代わりに、write が書いたレシピ（<out>/recipe.yaml）で前処理をやり直してもよい（RMVPE を回し直す）。
どちらの場合も、学習の設定に model.hold_symbol: "-" を書く（guide/f0_model.md「音符エンコーダ」）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.hold_labels import rules  # noqa: E402


# ── 入力（レシピ・シャード・lab）────────────────────────────────────────────────

def load_recipe(path):
    r = yaml.safe_load(open(path, encoding="utf-8")) or {}
    from preprocess.run import _enumerate_songs
    songs = [s for s in _enumerate_songs(r, r["db_root"]) if s not in set(r.get("exclude", []))]
    return r, songs


def song_file(r, key, song) -> Path:
    from preprocess import recipes
    return Path(recipes.resolve_song_file(r["db_root"], r[key], song))


def load_shard(shard_dir):
    """→ (metadata, shard.npz を開いたもの, 番号 → 音素)。"""
    d = Path(shard_dir)
    meta = json.load(open(d / "metadata.json", encoding="utf-8"))
    phonemes = meta.get("phonemes") or [p for p, _ in sorted(meta["phoneme_vocab"].items(), key=lambda x: x[1])]
    return meta, np.load(d / "shard.npz"), phonemes


def phrase_song(name: str, prefix: str) -> str:
    """句の名前（<name_prefix><曲>_<番号 4 桁>）→ 曲の名前。"""
    return name[len(prefix):].rsplit("_", 1)[0] if name.startswith(prefix) else name.rsplit("_", 1)[0]


def lab_rows(path: Path, unit: str):
    """lab の行（3 つ以上の欄がある行。番号は rules.split_lab_lines と同じ数え方）→ [(始まり 秒, 終わり 秒, 元の記号)]。
    長さ 0 の行も番号を数えるために残す（前処理はその行を読み飛ばすので、照合では飛ばす）。"""
    scale = 1e-7 if unit == "100ns" else 1.0
    rows = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        parts = ln.split(maxsplit=2)
        if len(parts) == 3:
            rows.append((float(parts[0]) * scale, float(parts[1]) * scale, parts[2].strip()))
    return rows


def normalizer(r, ps):
    """前処理と同じ写像（レシピの phon_norm → 言語パックの sil_map → 無音は pau）。"""
    phon_norm = r.get("phon_norm") or {}

    def norm(p):
        p = phon_norm.get(p, ps.sil_map.get(p, p))
        return "pau" if ps.is_pau(p) else ps.normalize_ph(p)
    return norm


def match_phrase(phs, t0s, rows, norm):
    """句の pau 以外の音素の並びが lab と一致する所を探す（ずれの最も小さい所）。
    → (曲の時刻 − 句の時刻, 句の音素の番号 → lab の行の番号, 最大のずれ 秒) か None。"""
    seq = [(float(t0s[q]), p, q) for q, p in enumerate(phs) if p != "pau"]
    lab = [(i, s, norm(p)) for i, (s, e, p) in enumerate(rows) if e > s]   # 前処理と同じく長さ 0 の行は読まない
    lab = [x for x in lab if x[2] != "pau"]
    want = [p for _, p, _ in seq]
    hits = [i for i in range(len(lab) - len(want) + 1) if all(lab[i + m][2] == want[m] for m in range(len(want)))]
    if not seq or not hits:
        return None

    def drift(i):
        off = lab[i][1] - seq[0][0]
        return max(abs(lab[i + m][1] - off - seq[m][0]) for m in range(len(seq)))
    i = min(hits, key=drift)
    off = float(np.median([lab[i + m][1] - seq[m][0] for m in range(len(seq))]))
    dmax = max(abs(lab[i + m][1] - off - seq[m][0]) for m in range(len(seq)))
    return off, {q: lab[i + m][0] for m, (_, _, q) in enumerate(seq)}, dmax


def phrase_f0(z, name):
    cents = 1200 * np.log2(np.maximum(z[f"{name}|f0_interp"], 1e-3) / 440.0)
    return cents, z[f"{name}|uv"] > 0.5


# ── 1. run ──────────────────────────────────────────────────────────────────

def cmd_run(a):
    import librosa
    import soundfile as sf
    import torch
    from duration_model import phoneset
    from tools.hold_labels import game

    torch.set_grad_enabled(False)
    r, songs = load_recipe(a.recipe)
    ps = phoneset.load(r.get("lang", phoneset.DEFAULT_LANG))
    norm = normalizer(r, ps)
    meta, z, phonemes = load_shard(a.shard)
    fr = float(meta["frame_rate"])
    prefix = r.get("name_prefix", "")
    names = sorted(meta["phrases"], key=lambda n: (phrase_song(n, prefix), n))
    names = [n for n in names if phrase_song(n, prefix) in set(songs)]
    if a.limit > 0:
        names = names[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "game_raw.jsonl"
    done = set()
    if raw.exists():
        done = {json.loads(ln)["phrase"] for ln in raw.read_text(encoding="utf-8").splitlines() if ln.strip()}
    todo = [n for n in names if n not in done]
    model_dir = Path(a.model_dir) if a.model_dir else Path(a.game_root) / "pretrained" / a.model
    model, lang_map, fns = game.load(a.game_root, model_dir, a.device)
    lang = lang_map[a.lang] if a.lang != "none" else 0
    sr = model.inference_config.features.audio_sample_rate
    unit = r.get("lab_unit", "100ns")
    print(f"GAME {model_dir.name}、言語 {a.lang}、種 {a.seeds} 個、句 {len(names)}（済み {len(names) - len(todo)}、"
          f"残り {len(todo)}）、{a.device}", flush=True)
    cur, wav_song, rows = None, None, None
    t_start = time.time()
    with open(raw, "a", encoding="utf-8") as fo:
        for k, name in enumerate(todo, 1):
            song = phrase_song(name, prefix)
            if song != cur:
                y, s = sf.read(song_file(r, "wav", song), dtype="float32", always_2d=True)
                y = y.mean(axis=1)
                cur, wav_song = song, (librosa.resample(y, orig_sr=s, target_sr=sr) if s != sr else y)
                rows = lab_rows(song_file(r, "lab", song), unit)
            phs = [phonemes[int(i)] for i in z[f"{name}|phoneme_ids"]]
            durs = z[f"{name}|dur_sec"].astype(float)
            t0s = np.concatenate([[0.0], np.cumsum(durs)])
            m = match_phrase(phs, t0s, rows, norm)
            if m is None:
                fo.write(json.dumps({"phrase": name, "song": song, "miss": True}) + "\n")
                fo.flush()
                continue
            off, q2row, dmax = m
            s0, n = int(round(off * sr)), int(round((t0s[-1] + game.PAD_SEC) * sr))
            wav = np.zeros(n, np.float32)
            lo, hi = max(0, s0), min(len(wav_song), s0 + n)
            if hi > lo:
                wav[lo - s0:hi - s0] = wav_song[lo:hi]
            wav = wav / max(float(np.abs(wav).max()), 1e-6) * 0.8
            wd, _ = game.words(phs, durs, ps.is_nucleus)
            seeds = game.added_boundaries(model, fns, wav, wd, lang, a.seeds)
            cents, voiced = phrase_f0(z, name)
            fb = np.round(t0s * fr).astype(int)
            fb[-1] = len(cents)
            vowels = []
            for q, p in enumerate(phs):
                if not ps.is_nucleus(p):
                    continue
                idx = np.arange(fb[q], fb[q + 1])[voiced[fb[q]:fb[q + 1]]]
                diff = rules.best_split_diff(cents[idx], fr) if len(idx) >= 2 else 0.0
                v0, v1, row = float(t0s[q]), float(t0s[q + 1]), q2row[q]
                vowels.append({"j": q, "row": row, "ph": p, "v0": round(v0, 7), "v1": round(v1, 7),
                               "l0": round(rows[row][0], 7), "l1": round(rows[row][1], 7),
                               "diff": round(float(diff), 1),
                               "seeds": [[round(float(t), 3) for t in added if v0 < t < v1] for added in seeds]})
            fo.write(json.dumps({"phrase": name, "song": song, "off": round(off, 7), "drift": round(dmax, 4),
                                 "n_seeds": a.seeds, "vowels": vowels}, ensure_ascii=False) + "\n")
            fo.flush()
            if k % 25 == 0 or k == len(todo):
                el = time.time() - t_start
                print(f"  {k}/{len(todo)}（{el:.0f} 秒、残り約 {el / k * (len(todo) - k) / 60:.0f} 分）", flush=True)


# ── 2. write ────────────────────────────────────────────────────────────────

def load_overrides(paths):
    """人の判断 {"<句>#<音素の番号>": {"decision": "split"|"keep", "splits_sec": [句の中の秒, ...]}} を重ねて読む。"""
    out = {}
    for p in paths or ():
        out.update(json.load(open(p, encoding="utf-8")))
    return {k: v for k, v in out.items() if v.get("decision") in ("split", "keep")}


def cmd_write(a):
    params = rules.Params(margin=a.margin_ms / 1000, min_diff=a.min_diff, min_len=a.min_len_ms / 1000,
                          min_step=a.min_step)
    r, songs = load_recipe(a.recipe)
    out = Path(a.out)
    recs = [json.loads(ln) for ln in (out / "game_raw.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    ok = [x for x in recs if not x.get("miss")]
    overrides = load_overrides(a.overrides)
    meta, z, _ = load_shard(a.shard)
    fr = float(meta["frame_rate"])
    unit = r.get("lab_unit", "100ns")
    rows_of = {}                                     # 曲 → lab の行（母音の頭の正確な時刻を shard 段へ渡すため）
    splits, by_song = [], {}
    n_vowel = n_drop = n_over = 0
    for x in ok:
        f0 = None
        for v in x["vowels"]:
            n_vowel += 1
            ov = overrides.get(f"{x['phrase']}#{v['j']}")
            n_over += ov is not None
            if ov is None and params.min_step > 0 and f0 is None:
                f0 = phrase_f0(z, x["phrase"])
            ts, src, dropped = rules.choose_splits(v, x["n_seeds"], params, f0, fr, ov)
            n_drop += dropped
            if ts:
                off = v["l0"] - v["v0"]
                if x["song"] not in rows_of:
                    rows_of[x["song"]] = lab_rows(song_file(r, "lab", x["song"]), unit)
                l0, l1 = rows_of[x["song"]][v["row"]][:2]
                splits.append({"song": x["song"], "row": v["row"], "phrase": x["phrase"], "j": v["j"], "ph": v["ph"],
                               "l0": round(l0, 7), "l1": round(l1, 7), "splits": [round(t, 7) for t in ts],
                               "splits_phrase": [round(t - off, 7) for t in ts], "src": src})
                by_song.setdefault(x["song"], {})[v["row"]] = ts
    lab_dir = out / "lab"
    lab_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    for song in songs:
        src_path = song_file(r, "lab", song)
        if not src_path.exists():
            continue
        text = src_path.read_text(encoding="utf-8")
        lines = rules.split_lab_lines(text.splitlines(), by_song.get(song, {}), unit, a.hold)
        n_rows += sum(len(t) for t in by_song.get(song, {}).values())
        nl = "\r\n" if "\r\n" in text else "\n"          # 改行の形と最後の改行の有無は元の lab に合わせる
        with open(lab_dir / f"{song}.lab", "w", encoding="utf-8", newline="") as fo:
            fo.write(nl.join(lines) + (nl if text.endswith("\n") else ""))
    json.dump({"margin_ms": a.margin_ms, "min_diff_cents": a.min_diff, "min_len_ms": a.min_len_ms,
               "min_step_cents": a.min_step, "hold_symbol": a.hold, "overrides": [str(p) for p in a.overrides or ()],
               "splits": splits}, open(out / "splits.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # この lab を読むレシピ（wav は元のまま。前処理をやり直すときに使う）
    rh = dict(r)
    rh["name"] = f"{r['name']}_hold"
    rh["hook"] = r.get("hook", r["name"])
    rh["lab"] = str((lab_dir / "{song}.lab").resolve())
    rh["hold_symbol"] = a.hold
    with open(out / "recipe.yaml", "w", encoding="utf-8") as fo:
        fo.write(f"# tools/hold_labels/split.py write が {a.recipe} から作ったレシピ（伸ばしのラベル）\n")
        yaml.safe_dump(rh, fo, allow_unicode=True, sort_keys=False)
    n_game = sum(1 for s in splits if s["src"] == "game")
    print(f"句 {len(recs)}（lab と合わず飛ばした {len(recs) - len(ok)}）、母音 {n_vowel}: 分けた {len(splits)}"
          f"（GAME {n_game}、人の判断 {len(splits) - n_game}）、足した行 {n_rows}。人の判断のある母音 {n_over}、"
          f"段差 {params.min_step:.0f} c 未満で捨てた境界 {n_drop}")
    print(f"→ {lab_dir}（{len(songs)} 曲）、{out / 'splits.json'}、{out / 'recipe.yaml'}")


# ── 3. shard ────────────────────────────────────────────────────────────────

def cmd_shard(a):
    meta, z, phonemes = load_shard(a.shard)
    fr = float(meta["frame_rate"])
    sp = json.load(open(a.splits, encoding="utf-8"))
    hold = a.hold or sp.get("hold_symbol", "-")
    vocab = list(phonemes) + ([] if hold in phonemes else [hold])
    hold_id = vocab.index(hold)
    cuts = {}
    for s in sp["splits"]:
        a0 = int(round(float(s["l0"]) * fr))            # 前処理（phrase_cut）と同じく、曲の時刻をフレームに丸めて差を取る
        cuts.setdefault(s["phrase"], {})[int(s["j"])] = [int(round(float(t) * fr)) - a0 for t in s["splits"]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arrays, n_added = {}, 0
    for key in z.files:
        arrays[key] = z[key]
    for name, per in cuts.items():
        if f"{name}|phoneme_ids" not in arrays:
            continue
        ids, dur = arrays[f"{name}|phoneme_ids"], arrays[f"{name}|dur_sec"]
        bnd = np.concatenate([[0], np.round(np.cumsum(dur.astype(np.float64)) * fr).astype(int)])
        frames = {j: [int(bnd[j]) + c for c in cs] for j, cs in per.items()}
        new_ids, new_dur = rules.insert_holds(ids, dur, fr, frames, hold_id)
        n_added += len(new_ids) - len(ids)
        arrays[f"{name}|phoneme_ids"], arrays[f"{name}|dur_sec"] = new_ids, new_dur
    np.savez(out / "shard.npz", **arrays)
    meta = dict(meta)
    meta.update(phonemes=vocab, phoneme_vocab={p: i for i, p in enumerate(vocab)}, n_phonemes=len(vocab),
                hold_symbol=hold, hold_source={"shard": str(Path(a.shard)), "splits": str(Path(a.splits))})
    with open(out / "metadata.json", "w", encoding="utf-8") as fo:
        json.dump(meta, fo, ensure_ascii=False, indent=2)
    print(f"句 {len(meta['phrases'])}、分けた句 {len(cuts)}、足した「{hold}」{n_added} → {out}"
          f"（音素 {len(vocab)}、「{hold}」= {hold_id}）")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="GAME を全部の句にかける（game_raw.jsonl）")
    p.add_argument("--recipe", required=True)
    p.add_argument("--shard", required=True, help="元のラベルで前処理したシャード（data/<db>）")
    p.add_argument("--game_root", required=True, help="GAME を clone したフォルダ")
    p.add_argument("--model", default="GAME-1.0-medium", help="<game_root>/pretrained/ の下のモデルのフォルダ")
    p.add_argument("--model_dir", default=None, help="モデルのフォルダを直接指定する")
    p.add_argument("--lang", default="en", help="GAME の言語（en / ja / yue / zh、none で指定しない）")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="先頭から何句（0 = 全部）")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    p = sub.add_parser("write", help="分ける母音を決めて lab・splits.json・レシピを書く")
    p.add_argument("--recipe", required=True)
    p.add_argument("--shard", required=True)
    p.add_argument("--out", required=True, help="run と同じフォルダ")
    p.add_argument("--margin_ms", type=float, default=80.0)
    p.add_argument("--min_diff", type=float, default=30.0, help="候補の音高の差の下限（セント）")
    p.add_argument("--min_len_ms", type=float, default=200.0, help="これ以下の母音は分けない（ms）")
    p.add_argument("--min_step", type=float, default=80.0, help="境界の段差の下限（セント、0 で検査しない）")
    p.add_argument("--overrides", nargs="*", default=None, help="人の判断の JSON（後のものが優先）")
    p.add_argument("--hold", default="-", help="伸ばしの記号")
    p = sub.add_parser("shard", help="前処理済みのシャードに伸ばしの記号を差し込む")
    p.add_argument("--shard", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hold", default=None, help="伸ばしの記号（既定は splits.json のもの）")
    a = ap.parse_args()
    {"run": cmd_run, "write": cmd_write, "shard": cmd_shard}[a.cmd](a)


if __name__ == "__main__":
    main()
