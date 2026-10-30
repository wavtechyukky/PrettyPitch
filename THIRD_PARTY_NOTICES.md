# Third-party notices

PrettyPitch is released under the MIT license (see [`LICENSE`](LICENSE)). A few things shipped in, or fetched by, this repository are **not** ours and carry different terms. They are listed here. If you redistribute PrettyPitch, carry these notices with it.

| What | Where | Terms |
|---|---|---|
| Pitch-algorithm base classes | `preprocess/algorithms/base.py` | MIT (pitch-benchmark) |
| RMVPE pitch estimator | `preprocess/algorithms/rmvpe.py` | Apache-2.0, plus the MIT notices below |
| RMVPE weights `rmvpe.pt` | downloaded at run time, **not** in this repo | see §3 |
| Bundled F0 and consonant-duration models | `checkpoints/f0_3singer.pt`, `checkpoints/cons_dur_3singer.pt` | each singing database's terms of use — see §4 and `checkpoints/CREDITS.txt` |
| Bundled neural vocoder | `checkpoints/nhv_v3_2_1.onnx` | non-commercial (its own training data) — see §5 |
| Hanzi → pinyin table | `dict/zh_pinyin.json` | MIT (pypinyin) — see §7 |
| English pronouncing dictionary | `dict/en_cmudict.txt` | BSD-style (CMUdict) — see §8 |
| Training data (singing databases) | **not** in this repo | each singing database's terms of use |

Everything else in this repository is our own work and is MIT.

---

## 1. pitch-benchmark — `preprocess/algorithms/base.py` and `rmvpe.py` (MIT)

Both files were taken **verbatim** from

> **pitch-benchmark** — <https://github.com/lars76/pitch-benchmark>
> The MIT License (MIT), Copyright (c) 2025 Lars Nieradzik

- `preprocess/algorithms/base.py` is their `algorithms/base.py`: `PitchAlgorithm`, `ContinuousPitchAlgorithm` and `ThresholdPitchAlgorithm`.
- `preprocess/algorithms/rmvpe.py` is their `algorithms/rmvpe.py`: `MelSpectrogram`, `ConvBlockRes`, `ResEncoderBlock`, `ResDecoderBlock`, `Encoder`, `Intermediate`, `Decoder`, `DeepUnet0`, `BiGRU`, `E2E0`, `to_local_average_cents`, `get_model_path` and `RMVPEPitchAlgorithm`.

Their MIT text and copyright notice are in [`LICENSES/pitch-benchmark-MIT.txt`](LICENSES/pitch-benchmark-MIT.txt), and MIT requires that notice to travel with the code.

**Our modifications**

- `base.py`: `PitchAlgorithm` no longer clips the estimated pitch to `[fmin, fmax]`. RMVPE never receives those bounds, so clipping pinned out-of-range frames to the boundary and invented flat sustained notes rather than rejecting them.
- `rmvpe.py`: the notice header was added. The code is unchanged.

## 2. RMVPE and CREPE — inside `rmvpe.py` (Apache-2.0, MIT)

The model that pitch-benchmark packaged in that file comes from

> **RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music**
> Haojie Wei, Xueke Cao, Tangpeng Dan, Yueguo Chen
> <https://arxiv.org/abs/2306.15412> — code at <https://github.com/Dream-High/RMVPE>

which is **Apache-2.0**, and `to_local_average_cents()` originates one step further back in

> **CREPE** — <https://github.com/marl/crepe>
> The MIT License (MIT), Copyright (c) 2018 Jong Wook Kim

Because the RMVPE code is Apache-2.0 upstream, `preprocess/algorithms/rmvpe.py` is **not** covered by this repository's MIT license. We distribute it under the Apache License, Version 2.0 ([`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)), whose conditions are a superset of the MIT conditions; keeping the Apache notice together with the two MIT copyright notices satisfies all three upstreams. CREPE's text is in [`LICENSES/crepe-MIT.txt`](LICENSES/crepe-MIT.txt).

## 3. RMVPE weights (`rmvpe.pt`)

**Not redistributed here.** `get_model_path()` downloads the file on first use from

    https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt

which is a third-party mirror (`lj1995` is the RVC author). The weights originate with the RMVPE authors. We do not restate terms for them, because the mirror states none — if you redistribute the weights yourself, check with the RMVPE authors first.

## 4. Bundled model weights and training data

Two trained models here were trained on the three singing databases:

- `checkpoints/f0_3singer.pt` — the F0 model (3 speakers × 3 styles)
- `checkpoints/cons_dur_3singer.pt` — the consonant-duration model

(The bundled vocoder was trained on a different set of data — see §5.)

Both were trained on three singing-voice databases: 御丹宮くるみ (Oniku Kurumi), 夏目悠李(Natsume Yuuri; database production アマノケイ, voice provider 霧野蒼太) and 波音リツ(Namine Ritsu). The recordings themselves are **not** distributed. Each database's terms of use govern what you may do with the models — typically requiring credit and restricting commercial use — and they are **not** covered by this repository's MIT license. `checkpoints/CREDITS.txt` lists the credits and links, and `checkpoints/夏目悠李_利用規約.txt` is the full Natsume Yuuri terms of use, bundled verbatim as those terms require. Keep both files with the models if you redistribute them.

If you train a model yourself with a singing-voice database, the same applies: check that database's terms before redistributing anything you train.

## 5. Bundled vocoder (`checkpoints/nhv_v3_2_1.onnx`)

The vocoder comes from [NHVSing](https://github.com/wavtechyukky/NHVSing), which is also ours and also MIT. The **code** is MIT like the rest.

The **weights** are a separate matter. They were not trained on the three databases above but on a wider set of non-commercial singing datasets, so **they cannot be used commercially**. The NHVSing README lists those datasets with links to each one's terms; check them at the source before redistributing anything built with this vocoder.

## 6. LeapSinger acoustic model

The LeapSinger acoustic model used by the singing demo is **not** distributed here; its inference code is imported at run time from a sibling checkout of [LeapSinger](https://github.com/wavtechyukky/LeapSinger) (ours, MIT) and its weights come from that project's release, under that project's terms.

## 7. pypinyin — `dict/zh_pinyin.json` (MIT)

The hanzi → pinyin table used by the Chinese G2P (`svs/g2p_zh.py`) is generated by `tools/build_zh_pinyin.py` from the data of

> **pypinyin** — <https://github.com/mozillazg/python-pinyin>
> The MIT License (MIT), Copyright (c) 2016 mozillazg, 闲耘

We keep each character's default reading and the phrases whose readings differ from the defaults, without tones. Their MIT text is in [`LICENSES/pypinyin-MIT.txt`](LICENSES/pypinyin-MIT.txt). pypinyin is needed only to rebuild the table, not at run time.

## 8. CMUdict — `dict/en_cmudict.txt` (BSD-style)

The pronouncing dictionary used by the English G2P (`svs/g2p_en.py`) is generated by `tools/build_en_cmudict.py` from

> **The CMU Pronouncing Dictionary** — <https://github.com/cmusphinx/cmudict>
> Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.

We keep each word's first pronunciation, written in the `dict/en_full.phonemes` notation (lower case, stress digits dropped, unstressed AH0 as `ax`). The license text is in [`LICENSES/cmudict-BSD.txt`](LICENSES/cmudict-BSD.txt).
