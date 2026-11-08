# PrettyPitch

**日本語**: [README.md](README.md)

An F0 (pitch curve) model for singing voice synthesis that needs no musical score to train. It is the sibling repository of the [LeapSinger](https://github.com/wavtechyukky/LeapSinger) acoustic model and shares its frame grid (hop 256 / 44.1 kHz) and phoneme set.

🎧 **Listen here → [PrettyPitch demo page](https://wavtechyukky.github.io/PrettyPitch/)** — three singers, a tempo sweep, and a 5×5 grid of the expression and vibrato knobs

![PrettyPitch overview](guide/fig/prettypitch_overview.svg)

## Updates

**2026-11-08**

- **Languages other than Japanese:** language packs and lyric-to-phoneme conversion (G2P) for Korean (Team CODA notation), Chinese (ACV / Opencpop notation) and English. In English lyrics, a word's syllables can be spread over the following `+` notes. → [Languages other than Japanese](#languages-other-than-japanese)
- **Note encoder (`model.encoder: notes`):** the input is split into two streams, one of phonemes and one of notes (older models take the phoneme stream only). Labels that write a held vowel with a hold symbol (e.g. `an -`) let it tell "w an~" from "w an an". (With older models, both can only be entered as "w an an", so they cannot be learned properly.) Recommended for newly trained models. → [guide/f0_model.en.md](guide/f0_model.en.md)
- **Hold-label tool:** uses GAME to split vowels held over several notes into labels like `ay -`. → [tools/hold_labels/](tools/hold_labels/README.en.md)
- **SVS rendering (`svs/`):** F0 models trained with a hold symbol now receive held vowels as that symbol. Other models get the same input as before.
- **Other:** phrase-final consonants can now be covered by the consonant-duration model (`final_consonants` in the language pack). F0 models with only one or two speakers can now be exported to ONNX.

Existing models (the bundled weights and their ONNX files) keep working with the new code as they are (it is backward compatible). New models trained with a hold symbol have the same ONNX input and output shapes, but to use them as intended you need a host that passes held vowels as that symbol (one that reads `hold_symbol` in the metadata). Older hosts still run them, but held notes are then treated as re-sung.

## What is new

Training uses no score (ust / MusicXML). **A wav and a phoneme label (.lab) are enough.**

Instead of note pitches, PrettyPitch estimates a **representative pitch value** for each *nucleus* phoneme (syllable nucleus: a i u e o A I U E O N) from the F0 that was actually sung. The estimator is a refinement of the perceived-pitch weighted mean of Blaauw & Bonada (2017): stability is measured by the F0 slope (|dF0/dt|), and for a short note that is sung as a brief peak (< 300 ms) the value is taken near the top of the peak.

This one substitution buys several things at once:

- **No score alignment.** Preprocessing uses the lab and the wav as they are.
- **Multilingual support is straightforward.** A language is defined by one language pack, `dict/<lang>.yaml`, named by `lang:` in the training config and in the preprocessing recipe. SVS rendering (`svs/`) additionally needs a lyric-to-phoneme step (`dict/ja.mora` for Japanese, the bundled G2P for Korean, Chinese and English — see [Languages other than Japanese](#languages-other-than-japanese)). It is not fully language-agnostic — see [Notes](#notes).
- **Off-pitch singers are fine as training data.** The teacher is the residual against the pitch that was actually sung, so out-of-tune singing does not corrupt training. At inference the model sings the pitch the user asks for.

The model itself is a 3-channel generative model. A DiffWave-style DDPM (default 10 steps, x0 direct prediction) and rectified flow (1 step) are both available through `model.arch` in the config. It generates plain (expression), vibrato and v/uv, reconstructed as `F0 = 2^(baseline + plain + vibrato)` (next section). Speakers use a low-dimensional bank (spk_dim=32); styles (e.g. Ritsu default / normal / soft) are additive embeddings.

Besides reading the input per phoneme (the default, used by the bundled weights), the model can use a **note encoder** (`model.encoder: notes`) that reads it as two streams, one of phonemes and one of notes. Trained on labels that write a vowel held over several notes with a hold symbol (`an -`; [tools/hold_labels/](tools/hold_labels/README.en.md) makes them with GAME), it tells "wan~" from "wan an". The inputs and outputs are the same, so the ONNX shape does not change either; to use a model trained with a hold symbol as intended, though, the host has to pass held vowels as that symbol ([guide/f0_model.en.md](guide/f0_model.en.md)). It is the recommended choice for newly trained models.

## The three F0 layers (baseline + plain + vibrato) and v/uv

![The baseline / plain / vibrato layers and v/uv](guide/fig/f0_layers_furusato.png)

F0 is handled in the log2 domain as the **sum of three layers**: `F0 = 2^(baseline + plain + vibrato)`. The figure shows the first phrase of *Furusato* sung by Namine Ritsu; each panel adds one layer.

**baseline** — the score pitch (during training, the nucleus representative values) laid out as a per-frame staircase and rounded off with a σ = 20 ms Gaussian. The model does not generate it. It is a deterministic floor, built by the same formula in training and inference, and the other two layers are learned as deviations from it. This is why the output follows the score pitch closely.

**plain (expression)** — the part of the deviation from the baseline that is not vibrato: the scoop at a note onset, the fall at a note end, the transitions between notes and the slow movement inside a note. The name comes from "the plain F0 with vibrato removed", but the content is the singing expression itself, so read it as **expression** if you prefer. The two-stage pitch correction after generation (Blaauw's method + an asymmetric Tukey window) touches only this layer.

**vibrato** — the 4.5–7.5 Hz oscillation. In the training data it is extracted from the residual (log2 F0 − baseline) by a tanh soft clip followed by a band-pass filter, and only outside the neighbourhood (±0.1 s) of consonants and phoneme boundaries, so that the rise of a scoop is not mistaken for vibrato. Whatever is removed stays in plain, so **the three layers always sum to the original singing**.

**v/uv** — the third channel, a per-frame voicing probability, binarized at 0.5 after generation. The orange bands in the fourth panel are the consonant regions the model judged unvoiced.

**Why separate them** — plain and vibrato are generated separately so that Synthesizer V's vibrato modulation can be reproduced: plain alone can be strengthened or weakened, and so can vibrato alone. Learning the vibrato-free F0 and the vibrato as two signals also makes each shape simpler to learn.

## Pipeline

```
wav + lab ──preprocess──▶ npz {phoneme_ids, dur_sec, f0_interp, uv}   ← all that is stored
                              │
              dataset.py derives per item (tunable without re-preprocessing):
                              ├─ nucleus representative values (continuous MIDI, Blaauw weights)
                              ├─ baseline = value staircase → Gaussian smooth (σ=20 ms)
                              └─ plain / vibrato decomposition (tanh softclip → 4.5-7.5 Hz filtfilt × gain)
                              │
                        train.py (diffusion + GAN)
                              │
                        infer.py: JSON score → baseline → sample → pitch correction → F0
```

Inference-time pitch correction is two-stage — Blaauw's method (a per-note constant offset from the representative value) then an asymmetric Tukey-window pull — with the user's input pitch (integer or fractional) as the target. Correction touches only the plain channel, so scoops and vibrato keep their shape.

## Usage

```bash
pip install -e .            # what inference needs (torch, onnxruntime, ...)
```

Everything else is an extra, by purpose: `.[train]` (TensorBoard), `.[data]` (gdown for the database downloads), `.[tuner]` (the vibrato-separation web app), `.[notebooks]` (Jupyter), `.[export]` (ONNX export), `.[test]` (pytest). They combine: `pip install -e ".[train,data]"`.

**Run every `python -m ...` command below from the repository root** — `pip install -e .` installs only the `prettypitch` package; `svs/`, `preprocess/` and the other top-level modules run from the clone.

### 1. Get data and preprocess

The bundled weights were trained on three databases (five recipes); one is enough to train.

```bash
pip install -e ".[data]"
python preprocess/download_scripts/download_oniku.py
python preprocess/download_scripts/download_natsume.py
python preprocess/download_scripts/download_ritsu.py                  # default voice
python preprocess/download_scripts/download_ritsu.py --voice normal
python preprocess/download_scripts/download_ritsu.py --voice soft
for db in oniku natsume ritsu ritsu_normal ritsu_soft; do
    python -m preprocess.run --recipe configs/recipes/$db.yaml
done
```

This writes `data/<db>/shard.npz` and `metadata.json` per database. F0 comes from RMVPE (despike + octave-error fix; the weights download on first run). pau / br regions are silenced before F0 extraction so that breath noise is not detected as voiced (`silence_pau` in the recipe, on by default). **Do not run RMVPE under multiprocessing** (the 181 MB model would load per process).

The recipe `configs/recipes/<db>.yaml` is **the one configuration file per database**. Besides the preprocessing settings (`lang`, `silence_pau`, `f0_octave_fix`, excluded songs) it holds the per-database values the loader uses (`reject_f0_below`, overrides under `decomp:` and `pitch:`) and how the consonant-duration floors are measured (`duration:`). Speaker and style ids are decided only by `data.spk_map` / `data.style_map` in the training config.

**Adding a speaker**: copy `configs/recipes/example_filled.yaml` (a made-up Japanese singer) or `example_filled_en.yaml` (a made-up English one) to `configs/recipes/<db>.yaml`. The five bundled recipes leave `decomp:`, `pitch:` and `duration:` empty, so these two are where you can see what goes in them and why. Neither runs as it stands — they exist to be copied. The `<db>` in the file name must equal the `data/<db>/` directory that `preprocess.run` writes (training looks the recipe up by that name). Add the name to `data.spk_map` / `data.style_map` in the training config with the next free id, and raise `model.n_styles` if you add a style.

Preprocessing is visually inspectable:

```bash
python -m tools.plot_decomp --db data/ritsu --out out/qa --limit 10
```

### 1.5 Tuning the vibrato separation (vibrato_separation_tuner)

The plain/vibrato decomposition hyperparameters (band-pass range, consonant/boundary margins, ...) can be tuned visually in a web app:

```bash
pip install -e .[tuner]
python vibrato_separation_tuner/app.py   # -> http://127.0.0.1:8017
```

It decomposes phrases through the exact functions the training dataloader uses and shows per-phrase cards (F0 + plain overlay, vibrato, the gain curve, excluded regions and the filter response). Every parameter is explained in its UI tooltip, and [guide/tuner.en.md](guide/tuner.en.md) walks through both tabs. Once you like a setting, copy it into the `decomp:` section of `configs/ja_example.yaml` (per-database overrides go into the recipe's `decomp:`). The shards store only raw data, so **no re-preprocessing is needed**. For a non-Japanese dataset, change `lang:` in the recipe and in the training config (e.g. `lang: en_full` -> `dict/en_full.yaml`; the bundled languages and example configs are listed in [Languages other than Japanese](#languages-other-than-japanese)). To swap only the phoneme table or the nucleus set, use `--phonemes` / `--nucleus` of `train` / `duration_model.build_profile`.

### 2. Train

```bash
python -m train --config configs/ja_example.yaml \
    --data_dirs data/oniku data/natsume data/ritsu data/ritsu_normal data/ritsu_soft \
    --run_name pp_3singer --device cuda
```

Re-running the same command resumes from the latest checkpoint. TensorBoard (`log/<run_name>/`) shows losses, cent RMSE, and F0/vibrato comparison figures.

### 3. Infer

```bash
python -m infer --ckpt checkpoints/f0_3singer.pt \
    --score examples/example_score.json --out out/test.npz --png out/test.png
```

`checkpoints/f0_3singer.pt` is the bundled weight (DDPM, 10 steps, 3 speakers × 3 styles). A checkpoint you trained yourself gets the same shape with `python -m tools.slim_ckpt --ckpt log/<run>/ckpt_010000.pt --out <out>.pt`, which drops the discriminator and the optimizers.

Scores are JSON. Nucleus phonemes require `midi` (fractional is fine); values on consonants or `cl` are optional — they reshape **only the baseline** (plain/vibrato/v-uv stay identical, which is what pitch tweaking wants).

```json
{"spk_id": 2, "style_id": 0, "phonemes": [
  {"ph": "pau", "dur": 0.5},
  {"ph": "k",   "dur": 0.08},
  {"ph": "a",   "dur": 0.42, "midi": 61},
  {"ph": "cl",  "dur": 0.10, "midi": 68},
  {"ph": "a",   "dur": 0.50, "midi": 65},
  {"ph": "pau", "dur": 0.3}]}
```

## Repository layout

```
dict/           ja.yaml (language pack: nucleus set, silence map, score literals) + ja.phonemes (shared with LeapSinger)
                + ja.mora (lyric→phoneme); the ko, zh_acv, zh_opencpop and en_full packs and the G2P dictionaries
                (en_cmudict.txt, zh_pinyin.json). Another language = add <lang>.yaml
configs/        ja_example.yaml (the config the bundled weights were trained with; every parameter
                annotated) + examples for other languages (ko_coda, zh_acv, zh_opencpop, en_aidol, en_aidol_notes)
                + recipes/<db>.yaml (one per database) + recipes/example_filled{,_en}.yaml
preprocess/     wav+lab → shard.npz (entry: run.py; never reads a score)
prettypitch/    pitch.py (representative values) baseline.py decomp.py correction.py losses.py models/
dataset.py      shard → teacher signals (values / baseline / decomposition derived here every time)
train.py        training (diffusion + multi-scale GAN, auto-resume, TensorBoard)
infer.py        JSON score → F0
duration_model/ consonant-duration model + phoneme sets (phoneset.py) + floor profiles
svs/            score → singing. score_io (UST/MusicXML) / duration_postprocess / render / plot
                + g2p_ko.py, g2p_zh.py, g2p_en.py (lyric→phoneme)
notebooks/      01 singing demo · 02 design notes · 03 how to drive the F0 model
guide/          the reference documents (README.en.md is the index): pipeline, F0 model,
                representative pitch, vibrato split, pitch correction, durations + fig/
docs/           the demo page published through GitHub Pages (not documentation)
tests/          pytest (infer runs on the example; every score loads) + fixtures/
examples/       example_score.json (F0 only) + six songs with lyrics (kirakiraboshi.{ust,musicxml}, furusato, ...; SVS demo)
checkpoints/    bundled weights (f0_3singer.pt, cons_dur_3singer.pt, nhv_v3_2_1.onnx) + CREDITS.txt and the Natsume Yuuri terms
                + leapsinger/ (where the LeapSinger release goes)
tools/          download_leapsinger.py (fetch the acoustic model) plot_decomp.py (visual QA of
                preprocessing/decomposition) slim_ckpt.py (shrink a ckpt for distribution)
                render_demos.py + make_demo_audio.py + make_demo_figures.py (rebuild docs/)
                build_zh_pinyin.py, build_en_cmudict.py (rebuild the G2P dictionaries)
                hold_labels/ (hold labels with GAME, for training the note encoder)
export/         f0.py (ONNX export of the F0 model, DDPM and reflow alike, with a memory benchmark) cons_dur.py (consonant-duration model)
```

## Synthesizing singing (SVS)

Combined with the [LeapSinger](https://github.com/wavtechyukky/LeapSinger) acoustic model and the NHVSing v3.2.1 vocoder, a score renders all the way to audio. The F0 model, the consonant-duration model and the vocoder (`nhv_v3_2_1.onnx`) are bundled in `checkpoints/`, so the only extra download is the LeapSinger acoustic model (details in `checkpoints/leapsinger/place_leapsinger_models_here.txt`):

```bash
python tools/download_leapsinger.py                       # the acoustic model (~100 MB)
git clone https://github.com/wavtechyukky/LeapSinger.git ../LeapSinger   # the inference code
```

The first line unpacks the latest LeapSinger [Release](https://github.com/wavtechyukky/LeapSinger/releases) into `checkpoints/leapsinger/`. The version is looked up through the API rather than pinned; `--tag` picks a specific one, `--zip` uses a file you already have, and it does nothing if the model is already there. The inference code on the second line is imported at run time, so pass `--leapsinger_root` if you keep it elsewhere.

```bash
python -m svs.render examples/kirakiraboshi.ust -o out/kirakiraboshi.wav --spk_id 2
```

Consonant lengths come from the bundled **consonant-duration model** (`checkpoints/cons_dur_3singer.pt`): one prediction per consonant from speaker, style, neighbouring phonemes and note length, floored by `duration_model/profiles/<ckpt>.json` (defaults: mean − σ of the measured lengths per consonant mid-phrase, with σ capped at 0.6 × mean; **p75 for a phrase-initial consonant**; 75 ms for vowels; per-speaker/style values are preferred when present). The floors are honoured whenever they fit; only when vowel floor + consonant floors exceed the note length is the note compressed at a fixed ratio. The statistic (`cons_floor_stat: pctl` for a percentile), the vowel floor and the ratio are set per database in the recipe's `duration:` section and baked into the profile by `python -m duration_model.build_profile`. Pass `--cons_dur_ckpt none` to fall back to a flat 80 ms per consonant and A/B the difference.

### Timing

**A phrase-initial consonant sounds before the note's time in the score.** Consonants are placed ahead of their syllable nucleus so that the nucleus lands exactly on the scored time; mid-phrase, a consonant likewise reaches back into the preceding note. This is the usual arrangement in singing synthesis — the same idea as SynthV's preutterance or DiffSinger's word division.

How far a phrase is pulled forward depends on its first consonant (Japanese: median 35 ms, max 273 ms; English: median 110 ms, max 470 ms). When a song begins too close to time zero to allow that, **the whole song is shifted later** by the amount needed. Silence is added at the start of the output, and the timing relative to the score is preserved.

Both ends of the output carry 0.25 s of silence by default (`pau_lead_sec` / `pau_trail_sec` in `score_to_phrases`).

[guide/duration_model.en.md](guide/duration_model.en.md) is the full reference for the duration model and its post-processing. [guide/README.en.md](guide/README.en.md) indexes every document.

The notebooks walk through each stage:

- `notebooks/01_svs_demo.ipynb` — score → singing, plotting durations, F0, mel, waveform and the mel re-analyzed from the output audio; switching speakers, morphing styles continuously (`style_mix`), and comparing expression / vibrato strengths
- `notebooks/02_design_notes.ipynb` — what each correction does and **why** (for people modifying it)
- `notebooks/03_f0_control.ipynb` — expression vs vibrato as independent knobs, intensity that changes over the course of a phrase, and `style_mix` (it listens to the result, so like 01 it needs the LeapSinger acoustic model)

## Languages other than Japanese

The bundled weights cover the three Japanese singers only, but the following language packs and lyric-to-phoneme converters are included. With your own data, the F0 and consonant-duration models train exactly as for Japanese. Each phoneme table has the same contents as LeapSinger's table of the same name.

| Language pack | Notation | Lyric → phoneme | Example config |
|---|---|---|---|
| `ja` | Japanese | mora table (`dict/ja.mora`) | `configs/ja_example.yaml` |
| `ko` | Korean (Team CODA notation) | rule-based G2P (`svs/g2p_ko.py`: liaison, nasalization, tensification, ...) | `configs/ko_coda.yaml` |
| `zh_acv` | Mandarin (ACV notation) | hanzi → pinyin (`dict/zh_pinyin.json`, phrase readings for polyphones) → rules (`svs/g2p_zh.py`) | `configs/zh_acv.yaml` |
| `zh_opencpop` | Mandarin (Opencpop notation) | same | `configs/zh_opencpop.yaml` |
| `en_full` | English | pronouncing dictionary (`dict/en_cmudict.txt`, CMUdict's first reading) | `configs/en_aidol.yaml`, `configs/en_aidol_notes.yaml` |

Lyrics may also be written as phonemes (`[hh eh l ow]` in English, space-separated phonemes otherwise). Write English words missing from the dictionary as phonemes.

**Hold and syllable marks**
- "-", "ー" and notes without lyrics hold the previous note's vowel. In Korean, Mandarin and English the final consonant moves to the last held note ("밤" "-" → b a | a M).
- Japanese, Mandarin and Korean put one syllable on each note, so "+" is a hold there too.
- In English, a word of several syllables is written on one note and its syllables go, in order, to the following "+" notes (`again` `+` → ax | g eh n). `+~` and `+*` hold; `+3` starts at the word's third phoneme.
- `lyric_style` decides `++` and what happens when a word has more syllables than notes: `sequential` (default) gives each + the next syllable and puts leftovers on the last note; `count` sings as many syllables as there are + signs and drops the rest.

**Held notes and the F0 model**: an F0 model trained on labels written with a hold symbol (`ay -`; `hold_symbol` in the checkpoint) gets held vowels as that symbol from svs ("skies" "-" → s k ay | - z). Other models get the repeated vowel exactly as before. For acoustic models trained on labels that never split a vowel at a note change, `--merge_held_vowels` passes a held vowel as one phoneme.

## Notes

- One frame = 256/44100 s (172.265625 fps) — the same grid as the LeapSinger acoustic model.
- `data.min_sec: 1.7` is the minimum length the vibrato filter (filtfilt) needs, plus a safety margin (a 0.3 s FIR needs 0.93 s).
- Old-generation checkpoints (hop 512, integer-MIDI input) cannot be resumed — the input representation changed.
- **What "multilingual" actually means.** Everything that changes with the language lives in the language pack `dict/<lang>.yaml`: the phoneme inventory and id space (`<lang>.phonemes`; line order = id, id 0 is `pau`), the nucleus set (`nucleus:`), devoiced-vowel normalization (`vowel_normalize:`), silence symbols (`silence:` / `sil_map:`), score reading (`score:` — the UST encoding, rest/melisma lyric literals, the default melisma vowel), the lyric→phoneme table (`.mora`) or the name of a rule-based G2P (`g2p:`), and whether phrase-final consonants are duration-model targets (`final_consonants:`). The hold symbol is not part of a pack; the recipe's `hold_symbol` and the training config's `model.hold_symbol` add it. The *number* of consonants is not fixed, so a different inventory loads fine. Korean, Mandarin and English packs are included, but the bundled weights cover the three Japanese singers only.

## Acknowledgments

Thanks to the datasets used to train this model, and to the related projects.

- Oniku Kurumi singing database (Oniku Kurumi) — https://onikuru.info/db-download/
- Natsume Yuuri (database production: アマノケイ / voice provider: 霧野蒼太) — https://ksdcm1ng.wixsite.com/njksofficial/enunu-nnsvs
- Namine Ritsu — https://www.canon-voice.com/voicebanks/
- Blaauw & Bonada (2017), Applied Sciences 7(12), 1313 (the weighting of the nucleus representative values and the second stage of the inference-time pitch correction follow Appendix B of this paper) — https://www.mdpi.com/2076-3417/7/12/1313
- DiffWave (the diffusion model design; our code is our own) — https://arxiv.org/abs/2009.09761
- Rectified Flow (the formulation behind `model.arch: reflow`) — https://arxiv.org/abs/2209.03003
- ConvNeXt (the 1-D conditioner; our code is our own) — https://arxiv.org/abs/2201.03545
- pitch-benchmark (Lars Nieradzik; the two files in `preprocess/algorithms/` come from it) — https://github.com/lars76/pitch-benchmark
- RMVPE (the F0 model itself) — https://github.com/Dream-High/RMVPE
- CREPE (`to_local_average_cents()`, reached via RMVPE) — https://github.com/marl/crepe
- pypinyin (source data of the hanzi → pinyin table `dict/zh_pinyin.json` used by the Mandarin G2P) — https://github.com/mozillazg/python-pinyin
- The CMU Pronouncing Dictionary (source data of the English dictionary `dict/en_cmudict.txt`) — https://github.com/cmusphinx/cmudict
- GAME (openvpi; the note extraction model `tools/hold_labels/` uses to make hold labels; not bundled) — https://github.com/openvpi/GAME

The distributed multi-speaker models display the credits above, following each database's terms. For Natsume Yuuri, we display **database production: アマノケイ / voice provider: 霧野蒼太**, and the "Terms of use for Natsume Yuuri's output audio" is bundled next to the models in `checkpoints/` (`checkpoints/CREDITS.txt` and `checkpoints/夏目悠李_利用規約.txt`).

## License

The code is MIT (`LICENSE`). The following is **not** covered by MIT; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the details.

- **The bundled weights `checkpoints/f0_3singer.pt` and `checkpoints/cons_dur_3singer.pt`** — trained on the Oniku Kurumi, Natsume Yuuri and Namine Ritsu singing databases and governed by each database's terms of use (personal, non-commercial use is the premise; credits and links are in `checkpoints/CREDITS.txt`, and the full Natsume Yuuri terms are in `checkpoints/夏目悠李_利用規約.txt`). Keep those two files with the models if you redistribute them.
- **The bundled vocoder `checkpoints/nhv_v3_2_1.onnx`** — weights built with our own [NHVSing](https://github.com/wavtechyukky/NHVSing) (MIT). The code is MIT, but **the weights were trained on a different and wider set of non-commercial datasets than the two above, so they cannot be used commercially.** The NHVSing README lists the datasets and their terms.

- **`preprocess/algorithms/base.py` and `preprocess/algorithms/rmvpe.py`** — both taken verbatim from [pitch-benchmark](https://github.com/lars76/pitch-benchmark) (MIT, Copyright (c) 2025 Lars Nieradzik). Their notice is in `LICENSES/pitch-benchmark-MIT.txt` and must be kept when redistributing. `base.py` is MIT like the rest of this repository.
- `rmvpe.py` is not: the RMVPE model inside it is [Apache-2.0 upstream](https://github.com/Dream-High/RMVPE), so that one file is distributed under Apache-2.0 (full text in `LICENSES/Apache-2.0.txt`). Its `to_local_average_cents()` originates in [CREPE](https://github.com/marl/crepe); that MIT notice is in `LICENSES/crepe-MIT.txt`. The `rmvpe.pt` weights are downloaded at run time and are not distributed here.

Training data (singing databases), and any model trained on them, follow their own terms.
