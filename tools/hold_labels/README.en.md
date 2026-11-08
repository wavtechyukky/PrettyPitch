# tools/hold_labels — making hold labels

**日本語**: [README.md](README.md)

The note encoder (`model.encoder: notes`, [guide/f0_model.en.md](../../guide/f0_model.en.md)) can tell "wan~" (an held over two notes) from "wan an" (an sung again) when it is trained on labels that split a vowel sung over several notes per note, writing every part after the first as the hold symbol (`an -`). This tool makes such labels with [GAME](https://github.com/openvpi/GAME), openvpi's note extraction model.

**When you do not need it**: labels that always split the vowel where the pitch changes (`a a`), like the bundled Japanese data, need nothing. Use it for labels such as AIdol's, which write a vowel as one row even when it spans several notes.

## Setup

GAME and its model are not part of this repository. The code is MIT; the model is **CC BY-NC-SA 4.0**.

```bash
git clone https://github.com/openvpi/GAME.git ../GAME          # tested at 69ba01d
# unpack GAME-1.0-medium.zip from https://github.com/openvpi/GAME/releases/tag/v1.0.0 into ../GAME/pretrained/
pip install -e ".[game]"                                        # omegaconf and pydantic, used by GAME
```

GAME runs on the CPU (a few seconds per phrase with 5 seeds). MPS on Apple Silicon uses a lot of memory, so keep `--device cpu` (the default).

## Usage

It reads the shard preprocessed from the original labels (`data/<db>`) and its recipe.

```bash
# 1. run GAME on every phrase (work/<db>_hold/game_raw.jsonl; rerunning continues where it stopped)
python -m tools.hold_labels.split run --recipe configs/recipes/aidol.yaml --shard data/aidol \
    --game_root ../GAME --lang en --out work/aidol_hold
# 2. decide which vowels to split and where (lab/, splits.json, recipe.yaml); rerun with other limits at will
python -m tools.hold_labels.split write --recipe configs/recipes/aidol.yaml --shard data/aidol --out work/aidol_hold
# 3. insert "-" into the preprocessed shard (no second RMVPE pass)
python -m tools.hold_labels.split shard --shard data/aidol --splits work/aidol_hold/splits.json --out data/aidol_hold
```

In the training config, write `model.hold_symbol: "-"` (it appends "-" to the phoneme table) and, for the note encoder, `model.encoder: notes`. [configs/en_aidol_notes.yaml](../../configs/en_aidol_notes.yaml) is an example. Instead of step 3 you can preprocess again with the `work/<db>_hold/recipe.yaml` written by step 2 (RMVPE runs again). The phonemes and F0 come out the same, but recomputing durations from the lab times rounds differently, so a boundary occasionally moves by one frame (on two AIdol songs: 4 boundaries in 3 of 55 phrases, none of them at an inserted "-"). Use step 3 if you want to stay aligned with the original shard.

`--lang` is GAME's language (`en`, `ja`, `yue`, `zh`, or `none`).

## Split rules

GAME gets the phrase audio and word boundaries (a word starts at a vowel and runs through the following consonants). Of the note boundaries GAME adds inside a word, those inside the vowel are used, and only if all of the following hold (`write` options change them):

| Rule | Default | Option |
|---|---|---|
| Candidate: largest difference of medians when the vowel's F0 is split in two | at least 30 cents | `--min_diff` |
| Vowel length | longer than 200 ms (short vowels make diphthongs and transitions look like notes) | `--min_len_ms` |
| Boundaries within 30 ms of each other across 5 seeds | a majority | (`--seeds` of `run`) |
| Distance from the vowel edges and from the previous kept boundary | at least 80 ms (scoops and falling endings are not notes) | `--margin_ms` |
| Step of the median F0 in 100 ms on each side of the boundary (20 ms around it excluded) | at least 80 cents (drops splits made on timbre) | `--min_step` |

On 300 randomly chosen AIdol vowels judged by hand, GAME (majority of 5 seeds, before the margin and other limits) made the same call on 290. To override with your own judgements, pass `--overrides <json> ...` holding `{"<phrase>#<phoneme index>": {"decision": "split" or "keep", "splits_sec": [seconds in the phrase, ...]}}`.

## Output

| File | Contents |
|---|---|
| `game_raw.jsonl` | GAME's result per phrase (boundaries per vowel, pitch difference, lab row index) |
| `lab/<song>.lab` | hold labels (every part after the first is `-`; units and line endings as in the original lab) |
| `splits.json` | the split vowels and positions (seconds in the song and in the phrase), and the limits used |
| `recipe.yaml` | a recipe reading `lab/` (`hold_symbol: "-"`) |
