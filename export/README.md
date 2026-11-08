# export/ — ONNX 書き出し

F0 モデル（`export/f0.py`）と子音長モデル（`export/cons_dur.py`）の 2 本。どちらも「ラッパーで入出力を固定 → trace（dynamic axes）→ onnxsim → metadata（JSON サイドカー＋`metadata_props`）→ ORT と torch の照合 → 個人情報のバイト走査」の同じ手順。書き出し先 `export/out/` は追跡しない。

```
pip install -e ".[export]"          # onnx / onnxsim を追加（onnxruntime は基本依存）
```

コマンドはリポジトリのルートで実行する。

## F0 モデルの ONNX — `export/f0.py`

```
python -m export.f0 --ckpt checkpoints/f0_3singer.pt --out export/out/f0_3singer.onnx
python -m export.f0 --ckpt log/<reflow_run>/ckpt_010000.pt --out export/out/f0_reflow.onnx
python -m export.f0 --bench export/out/f0_3singer.onnx --seconds 240      # メモリ・時間の計測
```

DDPM（`arch: diffusion`）と rectified flow（`arch: reflow`）は同じスクリプト・同じ入出力。違いは乱数入力の枚数 `K`（metadata の `n_noise`。DDPM = n_steps = 10、reflow = 1）とグラフの大きさ（DDPM は 10 回の評価を展開: 2,265 ノード / reflow 358 ノード。ウェイトは共有されるのでファイルは 7.5 MB / 6.9 MB）。

**グラフに入れるもの** = `model.sample` と同じ計算（長さ展開 → ConvNeXt conditioner →話者 / スタイル → サンプラ → v/uv ヘッド）。**baseline とピッチ補正はホスト側**（`plain_scale` / `vib_scale` を生きたつまみのまま残すため。DiffSinger の dspitch と同じ切り方）。

| 入力 | 形 | 内容 |
|---|---|---|
| `tokens` | `[1, L]` int64 | 音素 id（metadata `phonemes` の行順） |
| `pitch_midi` | `[1, L]` float32 | 核の音高 [MIDI、連続値可] を後続の音素へ前方埋めした階段 |
| `durations` | `[1, L]` int64 | 音素ごとのフレーム数（各 ≥ 1、合計 = T） |
| `spk_id` | `[1]` int64 | |
| `style_mix` | `[1, n_styles]` float32 | one-hot なら `style_id` と同じ。連続混合可 |
| `noise` | `[1, K, 2, T]` float32 | 標準正規乱数。ホストが握るので seed 再現はホスト側 |

| 出力 | 形 | 内容 |
|---|---|---|
| `plain` | `[1, T]` | log2 領域の baseline からのずれ（`plain_std` 適用済み） |
| `vibrato` | `[1, T]` | 同上（`vib_std` 適用済み） |
| `vuv` | `[1, T]` | 有声確率 |

ホスト: `f0 = 2 ** (baseline_log2 + plain_scale·plain + vib_scale·vibrato)`、`voiced = (vuv > 0.5) & ~pau`。baseline は音高の階段を σ = `baseline_sigma_sec` のガウシアンで平滑したもの、ピッチ補正（Blaauw + 非対称 Tukey）は plain だけに掛ける。必要な定数はすべて metadata にある。

**長さ展開**は Python ループ（`InputEncoder.length_regulate` の `.item()`）を使わず、音素開始フレームへの scatter → cumsum でフレーム→音素 index を作る O(T+L) の形に書き換えてある（T×L の比較行列は作らない）。ラッパーと `model.sample`（seed 固定）の差は 3e-6 以下（float の丸め）。

**検証**: ORT vs torch は 0.6 / 3 / 12 / 40 s の 4 長さで最大 2.6e-6（両アーキ）。個人情報のバイト走査 OK。

**ノートエンコーダ（`model.encoder: notes`）のモデル**も、同じスクリプト・同じ入出力で書き出せる。グラフの中で伸ばしの記号を前の母音に足し込み、音素の列と音符の列を作る（長さ展開は `prettypitch/frontend.py` の、長さ 0 の音素があっても壊れない形）。metadata には次のキーが足される。どれも無いモデルは今までと同じ。

| キー | いつ | 内容 |
|---|---|---|
| `encoder` | `model.encoder` を指定したモデル | `notes` |
| `hold_symbol` | 伸ばしの記号で書いたラベルで学習したモデル（`model.hold_symbol`） | 伸ばしの記号（例 `-`）。ホストはこれがあるとき、伸ばしの音符で続く母音を記号で渡す。無いモデルには、今までどおり同じ母音を繰り返して渡す |
| `hold_spec` | 同上 | 記号の渡し方（記号の `pitch_midi` はその音符の音高、`durations` は母音のうちその音符の分） |

**メモリ（Apple Silicon・ORT CPU・子プロセスで RSS 監視）** — NHVSing / LeapSinger で起きた「中間テンソルが入力長に比例して爆発する」事故の再発確認。増え方は**長さに比例し、2 次の項なし**:

| 長さ | フレーム | DDPM 実行時の増分 | reflow 実行時の増分 | DDPM 時間 | reflow 時間 |
|---:|---:|---:|---:|---:|---:|
| 10 s | 1,793 | +53 MB | — | 0.14 s | — |
| 60 s | 10,391 | +313 MB（5.2 MB/s） | +228 MB（3.8 MB/s） | 0.76 s | 0.19 s |
| 240 s | 41,402 | +1,067 MB（4.4 MB/s） | +912 MB（3.8 MB/s） | 2.7 s | 0.64 s |
| 480 s | 82,743 | +1,635 MB（3.4 MB/s） | +1,827 MB（3.8 MB/s） | 5.1 s | 1.2 s |

（セッション作成までの常駐は約 250 MB。torch 側は 60 s で +153 MB / +118 MB。）フレーズ単位で合成する SVS（最長でも数十秒）では 100–200 MB の範囲。ホストが `SessionOptions.enable_mem_pattern = False` にすると、速度は変わらずに増分が約 4 割減る（reflow 480 s: +1,675 → +969 MB）。ステップ数は増分に効かない（DDPM と reflow がほぼ同じ）ので、残りは conditioner / denoiser の中間テンソルを ORT がどこまで再利用するかで決まる。

## 子音長モデルの ONNX — `export/cons_dur.py`

```
python -m export.cons_dur --ckpt <cons_dur ckpt> --out export/out/cons_dur.onnx
```

`.onnx` の隣に同名の `.json` を書く（同じ内容を ONNX の `metadata_props` にも入れる）。ホストが要るものは全部そこにある: 入力名の並び、`phonemes`（id = 行順。無声化母音の正規化は表の側で済んでいる）、`spk_map` / `style_map`、`profile_hint`（floor の JSON 名）。

入力は子音レコード列 [N]: `cons_id prev_ph_id next_ph_id pitch_midi prev_note_exists prev_note_dur next_note_dur pitch_diff` ＋ モデルが持てば `cluster_len prev2_ph_id next2_ph_id spk_id style_id`。出力は `dur_sec` [N]。持続長モデルは `ConsonantDurModel` の 1 種だけ。

`dur_sec` は `exp(x) − LOG_OFFSET` と `clamp(min=0.005)` まで含めた**自然長 [秒]**。レコード／音素列の組み立て（`duration_model/features.py`）はホスト側の仕事で、ONNX には入れない。エクスポート時に ORT と PyTorch を乱数入力と実フレーズ由来の入力で照合し、最後に**バイト走査**で絶対パス・メールアドレスが焼かれていないことを確かめる（配布中の `checkpoints/cons_dur_3singer.pt` → 0.12 MB、差 0.0000 ms、走査 OK）。

**ONNX は「自然長」だけを返し、floor と圧縮はホスト側に置く**。DiffSinger と同じ構成にする。

根拠: OpenVPI DiffSinger も `linguistic.onnx` + `dur.onnx` の 2 本だけを書き出し、ノート尺への合わせ込み（`RhythmRegulator`）は**グラフの外**に置いている。リスケールはホスト側がやる。

我々の後処理も同じ理由でグラフに入らない:

| 処理 | グラフに入れられるか |
|---|---|
| floor の適用（`max(自然長, c_min)`） | 入れられる（`Gather` + `Max`） |
| リージョン圧縮（`_compress_region`） | 4 分岐あり `where` で書けなくはない |
| 先頭 onset × leading pau の交渉 | **楽譜側の休符長が要る**（ONNX の入力の外） |
| 音節分解（`_syllabify`） | **音素表と核判定が要る**＝言語パック依存 |
| 話者別 floor | 表を焼くことになる。話者を増やすたびに再エクスポート |

後処理は「音素列を組み立てる」段階と不可分で、ONNX の入力である音素列が確定する**前**の処理を含む。切り出すとホスト側が音節分解だけ別途やることになり、かえって複雑になる。

したがって:
- ONNX の出力 = 子音の自然長 [秒]（`predict` と同じ `exp(x) - LOG_OFFSET`、`clamp(min=0.005)`）
- floor（`duration_model/profiles/<ckpt名>.json`）はホストが読む。**話者別は `per_speaker` キー**（`"<spk>"` / `"<spk>:<style>"` の 3 段フォールバック）。
- profile が無い場合はモデル出力をそのまま使う。`clamp(min=0.005)` だけが最終下限。
