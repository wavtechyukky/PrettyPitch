# How the duration model and the post-processing work

![Sharing out duration](fig/duration_model.png)

This document describes how `duration_model/` (training and inference) and `svs/duration_postprocess.py` (post-processing) behave. The two are inseparable. You cannot follow either one on its own, so they share one document.

---

## 1. What this model does

**It decides consonant length only.** A vowel gets what is left after the consonants take their share of the note length.

```
when 0.5 s notes ['k','a'] and ['t','a'] stand side by side
  pau:250  k:60  a:466.3  t:33.7  a:500  pau:250
                  └ 500 − 33.7. what is left after the t of the next note cuts in
```

**What eats into a vowel is the onset of the next note, or the coda of its own note. It is not its own onset.** The 60ms of `k` comes from the **previous vowel**, not from the vowel of its own note. At the start of a phrase there is no vowel to cut, so the onset competes with the lead pau for room instead. The phoneme order is `[lead pau][lead onset][nucleus ...]`, so the onset sits **after** the pau. See §6-④ and §8.

Adding a coda shrinks the note's own vowel. Here is an example with the Japanese `cl` (the geminate) placed in the coda.

```
['k','a','cl'] 0.5 s → pau:250  k:60  a:321.0  cl:136.0  t:42.9  a:500  pau:250
                                       └ the coda cl takes the vowel from 466.3 to 321.0
```

**Training needs no score (score-free).** The training data is wav plus phoneme labels (lab) only. The input pitch comes from the representative value of F0.

---

## 2. The model itself

`ConsonantDurModel` (`model.py`) is a small MLP.

| Configuration | Parameters |
|---|---|
| Japanese, 50 phonemes, 3 speakers, 3 styles, no `cluster_len` (the bundled weights) | **27,137** |
| The same with `cluster_len` added | 27,713 |
| English, 46 phonemes, 1 speaker, with `cluster_len` | 26,561 |

### Inputs

These are the keys `encode()` actually reads.

| Feature | Type | Meaning |
|---|---|---|
| `cons_id` | embedding | The target consonant |
| `prev_ph_id` / `next_ph_id` | embedding | The phoneme before and the phoneme after. Consonants and vowels share one phoneme space |
| `prev_note_dur` / `next_note_dur` | linear projection | Total time given to the neighbouring syllables. Passed through `log1p(seconds / 0.3)` before the projection |
| `pitch_midi` | embedding | Pitch of the following syllable (MIDI) |
| `pitch_diff` | linear projection | Pitch difference from the previous syllable. Clamped to +/-12 semitones, then divided by 12 |
| `prev_note_exists` | embedding | Whether a previous syllable exists |
| `cluster_len` | embedding | Length of the consonant cluster this consonant belongs to (§3) |
| `spk_id` / `style_id` | embedding | Speaker and style, only when you set them |
| `style_mix` | mixing ratio | Weighted sum of the style embeddings. Pass it instead of `style_id` to blend styles continuously |
| `prev2_ph_id` / `next2_ph_id` | embedding | **Not used by default.** Only with `--wide_context` |

Every feature is embedded into the hidden dimension and added. They are not concatenated.

**A consonant with no note after it** (a coda at the end of a phrase or before a rest) is left out of the training records by default and gets a flat `FALLBACK_CONS_DUR` (80 ms) at inference, because Japanese hardly has any. For languages where phrases normally end in a closed syllable, such as Korean, `final_consonants: true` in the language pack records and predicts it as the previous note's coda (pitch = the previous note's representative value, `pitch_diff` 0, `next_note_dur` 0 = no note after it). Among the bundled packs, `ko`, `zh_acv` and `en_full` set it.

### Output

The model regresses `log(length + 0.001)`. Inference computes `exp(output) - 0.001` and clamps it to a lower bound of 0.005 seconds.

---

## 3. What cluster_len is

It is **the number of consonants in a row on one side of the syllable nucleus**. It is a training option, and it is on unless you pass `--no_cluster_len`.

**Whether it helps splits by language.** We compared pairs that differ only in this flag, on the same data with the same seed. In English all 6 pairs from NUS and aidol were better with it, by 0.09 to 1.46ms of validation MAE. In Japanese both pairs were better without it, 0.13 to 0.22ms worse with it. A Japanese consonant cluster is at most 2 long even counting the geminate, so this feature can carry almost no information.

**The bundled Japanese weights do not use it.** When you train on English, leave it on at the default.

```
en  pau:0  s:3  t:3  r:3  ey:0  n:2  jh:2  pau:0      "strange"
en  pau:0  z:3  d:3  t:3  uw:0  pau:0                 "-sed to" (3 consonants pulled forward to the next nucleus)
ja  pau:0  k:1  a:0  cl:2  t:2  a:0  pau:0            「かった」 (katta) (the geminate cl plus t, length 2)
```

A nucleus (a vowel) gets 0. The test is `is_nucleus`, the same one `split_syllable` uses.

---

## 4. Keeping the features to the same meaning in training and inference

We keep them aligned by **running both sides through the same functions**.

| | Training | Inference |
|---|---|---|
| Phoneme sequence | From lab | From the score |
| Entry point | `duration_model/dataset.py` | `svs/cons_dur.py::_flatten` |
| Record creation | `features.extract_phrase_records` | **The same function** |
| Total time of a syllable | `features.syllable_span_durs` | **The same function** (`extract_phrase_records` reapplies it internally) |
| Cluster length | `features.cluster_lengths` | **The same function** |

If you do not pass `span_dur`, `extract_phrase_records` calls `syllable_span_durs` again internally. Both paths therefore produce arrays built by the same rule.

---

## 5. Syllable splitting

This is the entry point of the post-processing. It lives in `svs/duration_postprocess.py`.

### `_syllabify(phoneme sequence)` — split what is inside a note into groups

```
_syllabify(['k','a','k','a'])  = [['k','a'], ['k','a']]
_syllabify(['i','cl','t','a']) = [['i'], ['cl','t','a']]     consonants go to the onset of the next nucleus
_syllabify(['k','s','t'])      = [['k','s','t']]             no nucleus, so no split
_syllabify(['a','pau','k','a'])= [['a'], ['pau'], ['k','a']]  pau is a boundary too
```

The boundary test is whether the phoneme is a nucleus or a `pau`. That is why groups of only `pau` appear.

**Each returned group holds at most one nucleus.** Some groups hold none, such as `['cl']` or `['k','s','t']`. Only a group with a nucleus is a syllable.

When one note holds several morae (`かか` (kaka) = `k a k a`, `がっこう` (gakkou) = `g a cl k o u`), we split the note evenly by the number of groups. The note then behaves exactly like a row of one-mora notes.

### `split_syllable(phoneme sequence)` — split into (onset, nucleus, coda)

```
split_syllable(['k','a'])                  = (['k'], ['a'], [])
split_syllable(['s','t','r','ey','n','jh'])= (['s','t','r'], ['ey'], ['n','jh'])
split_syllable(['k','s','t'])              = ([], ['k','s','t'], [])     no nucleus
split_syllable(['cl'])                     = ([], ['cl'], [])
```

**When there is no nucleus at all, it returns `([], everything, [])`.** Pushing those phonemes into the onset would make them eat into the previous note, the span they left would fill with pau, and real phonemes would vanish from the output.

---

## 6. The post-processing flow

`postprocess_durations()` runs in this order.

### ① Turn the note into groups (`_to_syls`)

When one note holds several morae, we split the note length evenly by the number of groups `_syllabify` returns.

Through `svs/render.py` the earlier `split_phrase_syllables` has already cut every note down to one group, so this even split normally does not happen.

**We split a note into as many equal parts as it has nuclei, before we predict duration.** We predict consonant length after the split, so the consonants that the split exposes inside the note get predictions too. A multi-mora note like `かか` (kaka) produces exactly the same output as two `か` (ka) notes in a row.

### ② Apply the floor (`_floor_each`)

Each consonant takes the larger of the model value and the floor. **The onset of the first group of a phrase uses `lead_c_min`, the p75 statistic for phrase-initial consonants.** Everything else uses `c_min`, the mean minus σ.

The same phoneme has a different duration at the start of a phrase than in the middle, so we do not apply the mid-phrase `c_min` to the first one.

### ③ Region compression (`_compress_region`)

**The coda of note j and the onset of the next note fight over span j.**

```
  span of note j
  ├─ vowel ─┤─ coda_j ─┤─ the part onset_(j+1) eats into ─┤
```

An onset consonant is pulled forward so that the nucleus of the next note starts on time.

There is only one branch. It asks **whether the sum of the floors of the phonemes in the span fits inside the span**.

Below, `floor` means the threshold itself: `c_min` for a consonant, and a per-speaker, per-style value for a vowel. The value a consonant holds after step ② is `max(model value, floor)`.

| Rule | Condition | Behaviour |
|---|---|---|
| A It fits | span >= the sum of the floors of the phonemes in the span | **Guarantee the floor.** Each consonant keeps `max(model value, floor)`. If the consonant total exceeds span minus the vowel floor, shrink it to that limit. Even then no consonant goes below its own floor. The remainder goes to the vowel |
| B It does not fit | Below that | **Split by weight ratio.** One unit = span / Σ weights, and each phoneme = its own weight × one unit. `phoneme_weight` defaults to 1.0, so by default a vowel and a consonant get the same length |

You set a weight per phoneme, and you can set it for vowels as well as consonants. See `weight()` in `duration_model/profile.py`. Only the ratio matters, so setting one phoneme to 2.0 doubles that phoneme alone. Values clamp to the range 0.05 to 20, so a phoneme does not disappear even at 0.

Consonants with floors s=75, sh=80 and k=25 eat into the span. The vowel floor is 75ms, every weight is 1.0, and the model values equal the floors.

| Span | s | sh | k | Consonant total | Vowel | Rule |
|---|---|---|---|---|---|---|
| 400ms | 75.0 | 80.0 | 25.0 | 180.0 | 220.0 | A |
| 255ms | 75.0 | 80.0 | 25.0 | 180.0 | 75.0 | A (it fits exactly) |
| 200ms | 50.0 | 50.0 | 50.0 | 150.0 | 50.0 | B |
| 100ms | 25.0 | 25.0 | 25.0 | 75.0 | 25.0 | B |
| 50ms | 12.5 | 12.5 | 12.5 | 37.5 | 12.5 | B |

**Do not cap the consonant total in B.** The old code used consonant total = ratio × span, split inside by the ratio of the floors. The total share of the consonants did not depend on how many of them there were, so each one got crushed as more phonemes joined the split. With s+cl+k in 100ms, `k` came out at 8.9ms. A per-phoneme weight gives 25ms under the same conditions. A share follows from how many are splitting the span, not from a fixed cap.

**Do not use the ratio of the floors for the split either.** A floor is built as the lower bound of a distribution. That is not what a weight for sharing means. Fricatives have a high lower bound and plosives a low one, so splitting by the floor ratio crushes the plosives and nothing else.

The phonemes in the span are the vowel, every consonant of the coda, and the consonants of the next onset. The floor of the next onset counts only for **the part that does not fit in the rest between the notes (the gap)**. The wider the gap, the less the next onset affects the condition.

When the onset fits entirely in the gap, it has taken not one ms from this note span, so neither A nor B touches it. It does not grow, it is not drawn into the B split, and it keeps its floored value.

### ④ Adjust the phrase-initial consonant length (`_lead_pau_cons`)

The first onset of a phrase has no note in front of it, so it competes with the lead pau for room. **Consonant length wins.** The consonant always gets `min(consonant length, room)`, and the pau takes the remainder up to its target of 250ms. Consonant length here means `max(model value, lead_c_min)` after step ②, not the raw model value.

```
consonant length after step ② is 150ms
  room     ∞  → pau 250.0ms  consonant 150.0ms
  room 300ms  → pau 150.0ms  consonant 150.0ms
  room 200ms  → pau  50.0ms  consonant 150.0ms
  room 100ms  → pau   0.0ms  consonant 100.0ms   too little room, so the consonant is cut
```

**Today `svs/render.py` passes `rest_before=1e9`.** This negotiation therefore always lands on the plenty-of-room case. The pau becomes 250ms and the consonant keeps its value from step ②. The design treats each phrase independently, so it does not hold the real room left by the previous phrase.

### ⑤ Build the phoneme sequence (`_build_sequence`)

This flattens everything into a list of `(phoneme, seconds)`. Two things happen here.

**A note with no nucleus** (a consonant-only note) is shared over the note length in the ratio of the floored values.

```
['s','sh','k'] 600ms → pau:250 s:267.0 sh:266.4 k:66.5 a:400 pau:250
['s','sh','k'] 200ms → pau:250 s:89.6  sh:88.8  k:21.5 a:400 pau:250
```

**The floor is not honoured here.** We take the values that step ② raised to the floor and rescale them by ratio to fit the note length. In the 200ms example `k` is 21.5ms, below the 30ms floor of speaker 0. Unlike the compression in step ③, nothing stops it at the lower bound.

**An internal pau** is what is left after the next onset has eaten in. If less than `PAU_DROP_SEC` (12ms) remains, we place no pau and give that time back to the preceding nucleus. A silence of only 1 or 2 frames (5.80ms per frame at 44.1kHz with hop 256) does not hold up as a phoneme. It is also shorter than the smallest internal pau in the training data, 11.6ms for Japanese and 5.8ms for English.

Here are two 0.4 second notes (`['k','a']` then `['t','a']`) with the gap varied. `t` takes 44.2ms, so the boundary sits where the gap passes 56.2ms.

```
gap 56ms → pau:250 k:60 a:411.8 t:44.2 a:400 pau:250          no pau, and a grows instead
gap 60ms → pau:250 k:60 a:400 pau:15.8 t:44.2 a:400 pau:250   a 15.8ms pau remains
```

Total length is preserved. In the examples above the total goes from 1.416s to 1.420s, in proportion to the gap.

### ⑥ Convert to frames (`to_frames` / `_align_durs`)

This is the one step that `svs/render.py` calls from outside `postprocess_durations`. Every model downstream works in frames, so seconds become an integer frame count here.

`to_frames` **takes the cumulative sum first and rounds afterwards**. Rounding each phoneme separately piles up error and drifts further the later you go. Rounding the boundaries instead keeps each phoneme within one frame.

Every phoneme gets **at least 1 frame**, so the total length can grow here.

```
[0.25, 0.003, 0.002, 0.4, 0.25] s @ 172.27fps
  per phoneme [43, 1, 1, 69, 43] (sum 157)  total frames from the boundaries 156
```

Step ⑤ drops internal pau shorter than 12ms, but this still happens when other phonemes shorter than one frame (5.80ms) remain.

Finally `_align_durs` **stretches or shrinks the last phoneme only**, to match the frame count the F0 model produced. That absorbs the drift above. "Total length is preserved" in step ⑤ holds at the seconds stage. In frames, the trailing pau moves by a few frames.

---

## 7. The profile (floors and weights)

It lives at `duration_model/profiles/<ckpt name>.json`. `build_profile.py` builds it from the recipe and the training data, and inference reads only this JSON.

```json
{
  "enabled": true,
  "vowel_floor_sec": 0.075,
  "cons_floor_sec":      { "s": 0.085, "sh": 0.095, "k": 0.050, "t": 0.045, ... },
  "lead_cons_floor_sec": { "s": 0.110, "sh": 0.130, "k": 0.035, "t": 0.015, ... },
  "phoneme_weight":         { "k": 1.5, "i": 0.6, ... },
  "phoneme_weight_default": 1.0,
  "per_speaker": {
    "0":   { "cons_floor": { "s": 0.090, ... }, "lead_cons_floor": { "s": 0.115, ... } },
    "0:0": { ... },
    "1":   { ... }
  }
}
```

The floor values are excerpts from the bundled `cons_dur_3singer.json`. Inside `per_speaker` the keys become `cons_floor` and `lead_cons_floor`, without the `_sec` suffix.

- **`enabled`** = set it to `false` and no floor is applied at all. Consonants go in at their natural length, and the only cap left is the one that keeps `_NUC_EPS` (5ms) for the vowel. Use it to see what the floors do
- **Consonant floor** = the mean of the measured lengths of that consonant minus σ, where σ is capped at 0.6 times the mean. Rounded to 5ms, with 5ms as the lower bound
- **Lead floor** = the 75th percentile, because a phrase-initial consonant comes out longer. This one has no lower bound
- **Vowel floor** = a fixed 75ms. Statistics do not decide it
- **Share weight** = `phoneme_weight`, used by rule B of step ③. The default is 1.0, which splits evenly across every phoneme. **You can set it for consonants and for vowels**
- **Per speaker and style** = the lookup runs in three steps, `"<speaker>:<style>"`, then `"<speaker>"`, then the all-speaker table

The recipe holds the weights and the rule for building the floors. A recipe is one file per DB, so it is per speaker already. The recipe does not hold the floor values themselves. `build_profile.py` computes those from measurements on the training data.

```yaml
# configs/recipes/<db>.yaml
duration:
  phoneme_weight_default: 1.0
  phoneme_weight: { k: 1.5, i: 0.6 }
  cons_floor_stat: pctl     # default is mean-sd. pctl switches to a percentile
  cons_pctl: 5              # the percentile to use with pctl
  sd_cap: 0.6               # cap on σ as a ratio of the mean
  min_n: 30                 # samples needed for a per-speaker floor
  min_lead_n: 10            # samples needed for a lead floor
```

Any string other than `pctl` in `cons_floor_stat` gives `mean-sd`. There is no validation of the value.

The `duration:` block of the recipe is wired only into the path that builds `per_speaker`. **If you do not pass `--config` to `build_profile.py`, what you wrote there has no effect.** It prints a warning in that case. Command line arguments decide the all-speaker table.

It still reads the old keys. `cons_weight`, a consonants-only table, goes into the same table. `short_note_ratio`, the share r that the consonant side takes, converts to a weight as r/(1−r). The old default of 0.5 gives the same split as a weight of 1.0.

There are two thresholds, and they apply to different things.

| Target | Threshold | Default | When there are too few |
|---|---|---|---|
| `cons_floor_sec` in `per_speaker` | `min_n` | 30 | The key is not written, so it falls back to the all-speaker value |
| `lead_cons_floor_sec` (both) | `min_lead_n` | 10 | The key is not written, so it falls back to `c_min` |
| The all-speaker `cons_floor_sec` | `min_n` | 30 | **The key stays** and inherits the base value |

The all-speaker table inherits the base values and then overwrites them with statistics, so the key itself survives even for a phoneme with `n < min_n`. Only the per-speaker tables drop keys.

`min_n` defaults to 30. The mean minus σ uses every sample, so a per-speaker floor is stable at around this count. A percentile is unstable on few samples, so it needs more.

The built-in defaults are **50ms for a consonant and 75ms for a vowel**. You get the same values when there is no profile at all. `load_cons_dur` looks for a JSON with the same name as the ckpt, so **renaming the ckpt drops you to the defaults**.

The phonemes that fall back to the defaults are the ones with few samples, and in Japanese those are the palatalised consonants. They are also on the long side. Measured `py` averages 110ms, against 71ms for all consonants. A short fallback would hurt them badly. 50ms is the conservative value under that assumption.

---

## 8. Phrase placement (`svs/render.py`)

**A phrase-initial consonant sounds before the note start time in the score.** This is the same convention as an onset inside a phrase being pulled forward into the previous note.

```
waveform   [lead pau][lead onset][nucleus ...]
placement  place it at start_sec − lead pau − lead onset
           → the nucleus lands exactly on start_sec
```

How far it moves forward depends on which consonant starts the phrase.

If the time we want to place it at goes negative, for example at the start of a song, **we shift the whole song back by that amount**. The output gains silence at the front, but the relation to the score is preserved.

---

## 9. What each file does

| File | Role |
|---|---|
| `model.py` | `ConsonantDurModel` (the MLP we use) and ckpt loading |
| `features.py` | Feature extraction. **Shared by training and inference** |
| `dataset.py` | Sends the values it needs from the npz that the F0 model preprocess produced |
| `train.py` | Training script |
| `profile.py` / `build_profile.py` | Definition and generation of the floor profile |
| `phoneset.py` | Loads the language pack (phoneme table, nucleus phonemes, `g2p`, `final_consonants`); appends the hold symbol (`hold`) to the phoneme table and the nucleus set |
| `svs/cons_dur.py` | Connects the model to the post-processing |
| `svs/duration_postprocess.py` | The post-processing itself |
| `svs/score_io.py` | Score (UST / MusicXML) to Phrase |
| `svs/render.py` | Calls the above in order, converts to frames, and places the phrases |

---

## 10. Key constants

| Constant | Value | Location |
|---|---|---|
| `LOG_OFFSET` | 0.001 | `model.py` |
| `MAX_CLUSTER` | 8 | `model.py` |
| `FALLBACK_CONS_DUR` | 0.08 | `duration_postprocess.py` |
| `_NUC_EPS` | 0.005 | Same file |
| `PAU_DROP_SEC` | 0.012 | Same file |
| `VOWEL_FLOOR_SEC` | 0.075 | `profile.py` (the vowel floor) |
| `_DEFAULT_CONS_FLOOR` | 0.05 | Same file (the consonant floor when there is no profile) |
| `_DEFAULT_PHONEME_WEIGHT` | 1.0 | Same file (the share weight of a phoneme. If every phoneme is at the default, the split is even) |
| `leading_pau_target` | 0.25 | Default in `postprocess_durations` |
| `rest_before` | 1e9 | The value `render.py` passes |
| `split_silence_sec` | 0.5 | Default in `score_to_phrases` |
| `pau_lead_sec` / `pau_trail_sec` | 0.25 | Same |
