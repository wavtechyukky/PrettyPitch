"""Run full preprocessing for one database from a recipe YAML.

    enumerate songs -> read lab -> per-phrase npz -> shard.npz

The recipe describes only what differs per database (paths, lab time unit, F0 range,
phoneme normalisation). Everything else is the common pipeline. Phoneme timing comes
straight from the `.lab`; **there is no musical score** — that is the whole point of
PrettyPitch.

    python -m preprocess.run --recipe configs/recipes/ritsu.yaml
"""
from __future__ import annotations

import argparse
import glob
import os

import yaml

from prettypitch.config import FrameSpec
from . import recipes
from .lab import process_file
from .shard import combine_db
from .vocab import Vocab


def _enumerate_songs(recipe: dict, db_root: str) -> list[str]:
    layout = recipe.get("layout", "per_song")
    if layout == "per_song":                       # {db_root}/{song}/{song}.*
        return sorted(os.path.basename(d) for d in glob.glob(os.path.join(db_root, "*"))
                      if os.path.isdir(d))
    if layout == "type_split":                     # {db_root}/{wav_dir}/{song}.wav
        wav_dir = os.path.join(db_root, recipe.get("wav_dir", "wav"))
        return sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(wav_dir, "*.wav")))
    raise ValueError(f"unknown layout: {layout}")


def _drop_fix(lab_fix, drop: tuple):
    """レシピの drop_phones（記号の頭。例: 中国語の曲の中の英単語の `en/`）: その記号の行を、曲ごとの lab_fix の後で
    `pau` にする。その行に触れるフレーズは `_drop_spans` で丸ごと除くので、その音声は shard に入らない。"""
    if not drop:
        return lab_fix

    def fix(segs):
        segs = lab_fix(segs) if lab_fix is not None else segs
        return [(s, e, "pau" if p.startswith(drop) else p) for s, e, p in segs]
    return fix


def _drop_spans(lab_path: str, lab_unit: str, lab_fix, drop: tuple) -> list:
    """drop_phones に当たる行の [[始まり, 終わり], ...]（秒）。process_file はこれに重なるフレーズをすべて除く。"""
    if not drop:
        return []
    from .lab import read_lab
    segs = read_lab(lab_path, lab_unit)
    if lab_fix is not None:
        segs = lab_fix(segs)
    return [[s, e] for s, e, p in segs if p.startswith(drop)]


def run_recipe(recipe_path: str, out_root: str = "data",
               frames: FrameSpec | None = None, limit: int | None = None, vocab=None) -> int:
    frames = frames or FrameSpec()
    r = yaml.safe_load(open(recipe_path, encoding="utf-8"))
    db_root = r["db_root"]
    out_dir = os.path.join(out_root, r["name"])

    exclude = set(r.get("exclude", []))
    songs = [s for s in _enumerate_songs(r, db_root) if s not in exclude]
    if limit:
        songs = songs[:limit]
    lab_unit = r.get("lab_unit", "100ns")
    phon_norm = r.get("phon_norm", {}) or {}
    prefix = r.get("name_prefix", "")
    hook = recipes.load(r.get("hook", r["name"]))     # per-singer code module, if any
    has_fix = hook is not None and hasattr(hook, "fix_lab")
    drop = tuple(r.get("drop_phones") or ())
    # 静音区間(pau / 長い cl)を無音化してから F0 を抽出する。既定 ON。
    silence_pau = bool(r.get("silence_pau", True))
    silence_fade_sec = float(r.get("silence_fade_sec", 0.05))
    print(f"静音化: {'ON' if silence_pau else 'OFF'}"
          + (f"  フェード {silence_fade_sec*1000:.0f} ms" if silence_pau else ""))

    n_ok = 0
    for s in songs:
        paths = {k: recipes.resolve_song_file(db_root, r[k], s) for k in ("wav", "lab")}
        missing = [k for k, p in paths.items() if not os.path.exists(p)]
        if missing:
            print(f"  [MISS {','.join(missing)}] {s}")
            continue
        name = f"{prefix}{s}"
        lab_fix = (lambda iv, _s=s: hook.fix_lab(_s, iv)) if has_fix else None
        # speaker/style identity is NOT baked into the shard — it is assigned per-DB at train
        # time from the config's data.spk_map / data.style_map. Nothing about identity belongs
        # in a recipe, which describes only how to *read* the database.
        try:
            process_file(name, paths["lab"], paths["wav"], out_dir, frames=frames,
                         f0_device=r.get("f0_device", "cpu"),
                         f0_octave_fix=bool(r.get("f0_octave_fix", True)),
                         silence_pau=silence_pau, silence_fade_sec=silence_fade_sec,
                         exclude_spans=_drop_spans(paths["lab"], lab_unit, lab_fix, drop) or None,
                         lab_unit=lab_unit, phon_norm=phon_norm, lab_fix=_drop_fix(lab_fix, drop), vocab=vocab)
        except Exception as e:  # noqa: BLE001 — keep going, report the failing song
            print(f"  [FAIL] {s}: {e}")
            continue
        n_ok += 1
    print(f"{r['name']}: processed {n_ok}/{len(songs)} songs")
    combine_db(out_dir, frames, vocab=vocab)
    return n_ok


def main():
    ap = argparse.ArgumentParser(description="Preprocess one database from a recipe YAML.")
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N songs")
    ap.add_argument("--hop", type=int, default=256, help="frame size (default 256)")
    ap.add_argument("--lang", default=None,
                    help="言語パックの上書き（既定はレシピの lang:）。dict/<lang>.yaml")
    ap.add_argument("--phonemes", default=None,
                    help="phoneme-list file (default: bundled Japanese dict/ja.phonemes)")
    args = ap.parse_args()
    # 言語はレシピの lang: が宣言する（DB は 1 言語）。--lang はその上書き。
    import yaml as _yaml
    from duration_model import phoneset as _ps_mod
    _r = _yaml.safe_load(open(args.recipe, encoding="utf-8")) or {}
    _lang = args.lang or _r.get("lang") or _ps_mod.DEFAULT_LANG
    _ps = _ps_mod.set_default(_ps_mod.load(_lang, args.phonemes))
    print(f"言語パック: {_ps.lang}  音素{_ps.n_phonemes} / 無音写像 {sorted(_ps.sil_map)}")
    run_recipe(args.recipe, out_root=args.out_root, frames=FrameSpec(hop=args.hop),
               limit=args.limit,
               vocab=Vocab.load(args.phonemes) if args.phonemes else Vocab(list(_ps.phonemes)))


if __name__ == "__main__":
    main()
