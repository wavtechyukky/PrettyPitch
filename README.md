# PrettyPitch

**English**: [README.en.md](README.en.md)

学習に楽譜データが不要な、歌声合成のための、F0（ピッチカーブ）モデルです。音響モデル [LeapSinger](https://github.com/wavtechyukky/LeapSinger) の姉妹リポジトリで、同じフレーム設定（hop 256 / 44.1kHz）・同じ音素セットを共有します。

🎧 **試聴はこちら → [PrettyPitch デモページ](https://wavtechyukky.github.io/PrettyPitch/)**（3 話者・早口・抑揚とビブラートの 5×5）

![PrettyPitch の概要](guide/fig/prettypitch_overview.svg)

## 更新情報

**2026/11/8**

- **日本語以外の言語:** 韓国語（Team CODA の表記）・中国語（ACV / Opencpop の表記）・英語の言語パックと、歌詞を音素に直す変換（G2P）を同梱しました。英語では、単語の音節を続く「+」の音符へ配れます。→「[日本語以外の言語](#日本語以外の言語)」
- **ノートエンコーダ（`model.encoder: notes`）:** 入力を音素の列と音符の列の 2 本で読む、新しい読み方です。伸ばしの記号で書いたラベル（`ay -`）で学習すると、「wan〜」と「wan an」を区別できます。新しく学習するモデルにはこちらを勧めます。→ [guide/f0_model.md](guide/f0_model.md)
- **伸ばしのラベルを作るツール:** GAME を使って、音符をまたぐ母音を「ay -」のように分けたラベルを作ります。→ [tools/hold_labels/](tools/hold_labels/README.md)
- **歌声合成（`svs/`）:** 伸ばしの記号で学習した F0 モデルには、伸ばしの母音を記号で渡します。ほかのモデルには今までどおりです。
- **その他:** 句末の終声も子音長モデルの対象にできるようにしました（言語パックの `final_consonants`）。話者が 1〜2 人のモデルでも、F0 モデルを ONNX に書き出せるようにしました。

配布ウェイトと ONNX の入出力は変わらないので、今までのモデルはそのまま使えます。

## 何が新しいか

学習データに楽譜（ust / MusicXML）を使いません。**wav と音素ラベル（.lab）だけ**で学習できます。

ノートの音高の代わりに、母音など「核」となる音素（音節核: a i u e o A I U E O N）の区間から、実際に歌われた F0 の**代表値**を推定して使います。推定は Blaauw & Bonada (2017) の知覚ピッチ重みによる重み付き平均を改良したものです。安定度は F0 の傾き（|dF0/dt|）で測り、一瞬だけ山なりに歌われる短いノート（< 300 ms）は山の頂点付近を代表値にします。

この置き換えで、次のことが成り立ちます。

- **楽譜との照合作業がなくなる。** 前処理は lab と wav をそのまま使うだけ。
- **多言語対応がしやすい。** 言語は `dict/<lang>.yaml`（言語パック）1 つが定義し、学習 config の `lang:` と前処理レシピの `lang:` で指定します。SVS レンダリング（`svs/`）はさらに歌詞→音素の変換が要ります（日本語は `dict/ja.mora`、韓国語・中国語・英語は同梱の G2P。「[日本語以外の言語](#日本語以外の言語)」）。完全に言語非依存ではない点は「[注意](#注意)」を参照。
- **音程の良くない歌唱データでも学習できる。** 教師は「実際に歌われたピッチ」との残差なので、楽譜からズレた歌い方でも学習は壊れない。生成時はユーザーが指定した音高どおりに歌う。

モデル本体は 3ch の生成モデルです。DiffWave 型 DDPM（既定 10 step・x0 直接予測）と rectified flow（1 step）を config の `model.arch` で選べます。plain（抑揚）・vibrato（ビブラート）・v/uv の 3 チャンネルを生成し、`F0 = 2^(baseline + plain + vibrato)` で復元します（次節）。話者は低次元バンク（spk_dim=32）、スタイル（例: リツ 無印/normal/soft）は埋め込みの加算で切り替えます。

入力の読み方は、音素ごとに読む形（既定・配布ウェイト）のほかに、音素の列と音符の列の 2 本で読む**ノートエンコーダ**（`model.encoder: notes`）を選べます。音符をまたぐ母音を伸ばしの記号で書いたラベル（`ay -`。GAME を使って [tools/hold_labels/](tools/hold_labels/README.md) で作れます）で学習すると、「wan〜」と「wan an」を区別できます。入出力は同じなので、ONNX の形も変わりません（[guide/f0_model.md](guide/f0_model.md)）。新しく学習するモデルにはこちらを勧めます。

## F0 の 3 層構成（baseline + plain + vibrato）と v/uv

![baseline / plain / vibrato の 3 層と v/uv](guide/fig/f0_layers_furusato.png)

F0 は対数（log2）の領域で **3 つの層の和**として扱います: `F0 = 2^(baseline + plain + vibrato)`。図は「ふるさと」の 1 フレーズ目を波音リツで生成したもので、上から順に各層を足していきます。

**baseline（土台）** — 楽譜の音高（学習時は核音素の代表値）をフレームごとの階段にし、σ = 20 ms のガウシアンで角を丸めたものです。モデルはこれを生成しません。学習でも推論でも同じ式で作る確定的な土台で、残りの 2 層は「baseline からのずれ」として学習されます。音高が楽譜どおりにある程度沿うのは、これを土台にするためです。

**plain（抑揚）** — baseline からの差分のうち、ビブラートを除いた成分です。ノート頭のしゃくり、語尾の落ち、ノート間のつなぎ、ノート内の揺れがここに入ります。名前は「ビブラートを除いた素の F0」から来ていますが、中身は歌唱表現そのものなので**expression** と読み替えて構いません。生成後の 2 段ピッチ補正（Blaauw 法 + 非対称 Tukey 窓）はこの層にだけ掛かります。

**vibrato** — 4.5–7.5 Hz の振動成分です。学習データでは、残差（log2 F0 − baseline）を tanh でソフトクリップしてからバンドパスフィルタを使い、子音の周辺と音素境界の周辺（±0.1 s）以外から抽出します（しゃくりの立ち上がりが偽ビブラートになるのを防ぐため）。取り除いた分は plain 側に残るので、**3 層の和は元の F0 の歌い方に一致**します。

**v/uv** — 3 つ目のチャンネルで、フレームごとの有声確率です。生成後に 0.5 で二値化します。図 4 段目の橙の帯が、モデルが無声と判断した子音区間です。

**分けるねらい** — この機構は Synthesizer V の vibrato modulation を再現できるように plain と vibrato を別々に生成しており、plain だけ強く、あるいは弱くすることもできますし、vibrato だけ強弱をつけることもまた可能です。また、ビブラートを除いた F0 と、ビブラートのみをそれぞれ学習することで、学習する F0 の形状が単純になるという狙いもあります。

## パイプライン

```
wav + lab ──preprocess──▶ npz {phoneme_ids, dur_sec, f0_interp, uv}   ← これだけ保存
                              │
              dataset.py が毎回導出（再前処理なしで調整可能）:
                              ├─ 核音素の代表値（連続MIDI, Blaauw重み）
                              ├─ baseline = 値の階段 → ガウシアン平滑(σ=20ms)
                              └─ plain / vibrato 分解（tanh softclip → 4.5-7.5Hz filtfilt × gain）
                              │
                        train.py（拡散 + GAN）
                              │
                        infer.py: JSON譜面 → baseline → sample → ピッチ補正 → F0
```

生成時のピッチ補正は Blaauw 法（代表値との差の定数オフセット）+ 非対称 Tukey 窓補正の 2 段で、ユーザーが入力した音高（整数でも小数でも）が目標値になります。補正は plain だけに掛かるので、しゃくり・ビブラートの形は保たれます。

## 使い方

```bash
pip install -e .            # 推論に要るもの（torch / onnxruntime など）
```

追加でいるものは用途別に extras にしてあります: `.[train]`（学習: TensorBoard）、`.[data]`（DB のダウンロード: gdown）、`.[tuner]`（ビブラート分離の Web アプリ）、`.[notebooks]`（Jupyter）、`.[export]`（ONNX 書き出し）、`.[test]`（pytest）。`pip install -e ".[train,data]"` のように並べて書けます。

**以下の `python -m ...` はすべてリポジトリのルートで実行してください**（`pip install -e .` が入れるのは `prettypitch` パッケージだけで、`svs/` や `preprocess/` などはリポジトリ直下のモジュールとして動きます）。

### 1. データ取得と前処理

配布ウェイトの学習には 3 つのデータベース（5 レシピ）を使います。1 つだけでも学習はできます。

```bash
pip install -e ".[data]"
python preprocess/download_scripts/download_oniku.py
python preprocess/download_scripts/download_natsume.py
python preprocess/download_scripts/download_ritsu.py                  # 無印（キレ）
python preprocess/download_scripts/download_ritsu.py --voice normal
python preprocess/download_scripts/download_ritsu.py --voice soft
for db in oniku natsume ritsu ritsu_normal ritsu_soft; do
    python -m preprocess.run --recipe configs/recipes/$db.yaml
done
```

DB ごとに `data/<db>/shard.npz` と `metadata.json` ができます。F0 は RMVPE（despike + オクターブ誤り補正つき、初回にウェイトを自動ダウンロード）。pau / br の区間は F0 を抽出する前に無音化します（息継ぎの吐息が有声と誤検出されるのを防ぐ。レシピの `silence_pau`、既定 ON）。**RMVPE はマルチプロセスで動かさないでください**（181MB のモデルがプロセス毎に載ります）。

レシピ `configs/recipes/<db>.yaml` は **DB ごとの唯一の設定ファイル**です。前処理の設定（`lang`・`silence_pau`・`f0_octave_fix`・除外曲）に加え、ローダーが DB 別に使う値（`reject_f0_below`・`decomp:`・`pitch:` の上書き）と、子音長 floor の統計の取り方（`duration:`）もここに書きます。話者 / スタイルの ID は学習 config の `data.spk_map` / `data.style_map` だけが決めます。

**新しい話者を足すとき**: `configs/recipes/example_filled.yaml`（架空の日本語話者）か `example_filled_en.yaml`（架空の英語話者）を写して `configs/recipes/<db>.yaml` を作ります。同梱 5 話者のレシピは `decomp:` / `pitch:` / `duration:` が空なので、何をどう書けるかはこの 2 本を見てください（そのままでは動きません。写して書き換える用です）。ファイル名の `<db>` は `preprocess.run` が書き出す `data/<db>/` の名前と同じにしてください（学習時はこの名前でレシピが引かれます）。学習 config の `data.spk_map` / `data.style_map` にその名前を連番で足し、スタイルを増やすなら `model.n_styles` も増やします。

前処理の結果は目視確認できます:

```bash
python -m tools.plot_decomp --db data/ritsu --out out/qa --limit 10
```

### 1.5 学習で使用するビブラート抽出と F0 の代表値推定のハイパーパラメータの調整

plain/vibrato 分解と代表値推定のハイパーパラメータは、Web アプリで目視調整できます:

```bash
pip install -e .[tuner]
python vibrato_separation_tuner/app.py   # → http://127.0.0.1:8017
```

<p align="center">
  <img src="guide/fig/tuner_vibrato_separation.png" width="49%" alt="ビブラート分離タブ">
  <img src="guide/fig/tuner_pitch_representative.png" width="49%" alt="代表値タブ">
</p>

学習と同一の関数で計算した結果をフレーズ毎のカードで表示します。タブが 2 つあり、左がビブラート分離（F0+plain 重ね・vibrato・ゲインカーブ・除外領域・フィルタ特性図）、右が核音素ごとの代表値（代表値の横線・平均律からのずれ・どのフレームが値を決めたかの重み）です。良い設定が見つかったら `configs/ja_example.yaml` の `decomp:` に転記してください（DB 別の上書きはレシピの `decomp:`）。npz は生データのみなので**前処理の再実行は不要**です。

使い方の詳細は [guide/tuner.md](guide/tuner.md) にあります。日本語以外のデータセットでは、レシピと学習 config の `lang:` を変えてください（例 `lang: en_full` → `dict/en_full.yaml`。同梱の言語と設定の例は「[日本語以外の言語](#日本語以外の言語)」）。音素表や核音素だけを差し替えたいときは、`train` / `duration_model.build_profile` の `--phonemes` / `--nucleus` を使います。

### 2. 学習

```bash
python -m train --config configs/ja_example.yaml \
    --data_dirs data/oniku data/natsume data/ritsu data/ritsu_normal data/ritsu_soft \
    --run_name pp_3singer --device cuda
```

同じコマンドを再実行すると最新チェックポイントから自動再開します。TensorBoard（`log/<run_name>/`）に損失・cent RMSE・F0/ビブラート比較図が出ます。

### 3. 推論

```bash
python -m infer --ckpt checkpoints/f0_3singer.pt \
    --score examples/example_score.json --out out/test.npz --png out/test.png
```

`checkpoints/f0_3singer.pt` は同梱の配布ウェイト（DDPM 10 step・3 話者 × 3 スタイル）。自分で学習したものは `python -m tools.slim_ckpt --ckpt log/<run>/ckpt_010000.pt --out <out>.pt` で判別器・optimizer を落として同じ形にできます。

譜面は JSON です。核音素には `midi`（連続値可）が必須、子音や `cl` への指定は任意で、指定すると **baseline の形だけ**が変わります（plain/vibrato/v-uv は不変 = 調声向け）。

```json
{"spk_id": 2, "style_id": 0, "phonemes": [
  {"ph": "pau", "dur": 0.5},
  {"ph": "k",   "dur": 0.08},
  {"ph": "a",   "dur": 0.42, "midi": 61},
  {"ph": "cl",  "dur": 0.10, "midi": 68},
  {"ph": "a",   "dur": 0.50, "midi": 65},
  {"ph": "pau", "dur": 0.3}]}
```

## リポジトリ構成

```
dict/           ja.yaml（言語パック：核音素・無音写像・楽譜記号） + ja.phonemes（音素・LeapSinger と共通）
                + ja.mora（歌詞→音素）。ko・zh_acv・zh_opencpop・en_full のパックと、G2P の辞書
                （en_cmudict.txt・zh_pinyin.json）。他言語は <lang>.yaml を足すだけ
configs/        ja_example.yaml（配布ウェイトを学習した設定・全パラメータ注釈つき）
                + 他言語の例（ko_coda・zh_acv・zh_opencpop・en_aidol・en_aidol_notes）
                + recipes/<db>.yaml（DB ごと） + recipes/example_filled{,_en}.yaml（書き方の例）
preprocess/     wav+lab → shard.npz（run.py が入口。楽譜は一切読まない）
prettypitch/    pitch.py（代表値） baseline.py decomp.py correction.py losses.py models/
dataset.py      shard → 教師（代表値・baseline・分解は毎回ここで導出）
train.py        学習（拡散 + MultiScale GAN・自動再開・TensorBoard）
infer.py        JSON譜面 → F0
duration_model/ 子音長モデル + 音素集合（phoneset.py）・floor プロファイル
svs/            楽譜→歌声の統合。score_io（UST/MusicXML）/ duration_postprocess / render / plot
                + g2p_ko.py・g2p_zh.py・g2p_en.py（歌詞→音素）
notebooks/      01 歌唱デモ・02 設計意図・03 F0 の使い方
guide/          資料一式（README.md が目次）。全体構成・F0 モデル・代表値・ビブラート分離・
                ピッチ補正・持続長モデル・チューナー + fig/（図）
docs/           GitHub Pages で公開するデモページ（資料ではありません）
tests/          pytest（infer が example で動く・全楽譜が読める）+ fixtures/
examples/       example_score.json（F0 単体用） + 歌詞入り楽譜（kirakiraboshi.{ust,musicxml}・
                furusato・haru_no_ogawa 等 + ievan_polkka_{200,250,300}（早口テスト））
checkpoints/    同梱ウェイト（f0_3singer.pt・cons_dur_3singer.pt・nhv_v3_2_1.onnx）+ CREDITS.txt・夏目悠李_利用規約.txt
                + leapsinger/（LeapSinger の Release を展開する場所）
tools/          download_leapsinger.py（音響モデルの取得） plot_decomp.py（前処理/分解の目視QA）
                slim_ckpt.py（配布用に ckpt を縮める） render_demos.py + make_demo_audio.py +
                make_demo_figures.py（docs/ のデモページを作り直す）
                build_zh_pinyin.py・build_en_cmudict.py（G2P の辞書を作り直す）
                hold_labels/（GAME で伸ばしのラベルを作る。ノートエンコーダの学習用）
export/         f0.py（F0 モデルの ONNX export。DDPM / reflow 共通・メモリ計測つき） cons_dur.py（子音長モデル）
```

## 歌声を合成する（SVS）

音響モデル [LeapSinger](https://github.com/wavtechyukky/LeapSinger) とボコーダ NHVSing v3.2.1 を組み合わせると、楽譜から歌声まで通ります。F0・子音長・ボコーダ（`nhv_v3_2_1.onnx`）は `checkpoints/` に同梱してあるので、追加で用意するのは LeapSinger の音響モデルだけです（詳しくは `checkpoints/leapsinger/place_leapsinger_models_here.txt`）:

```bash
python tools/download_leapsinger.py                       # 音響モデル（約 100MB）
git clone https://github.com/wavtechyukky/LeapSinger.git ../LeapSinger   # 推論コード
```

1 行目は LeapSinger の [Release](https://github.com/wavtechyukky/LeapSinger/releases) の最新版を `checkpoints/leapsinger/` に展開します（版は API で引くので固定していません。`--tag` で指定、`--zip` で手元の zip を使用、既に置いてあれば何もしません）。2 行目の推論コードは実行時に読み込むので、別の場所に置くなら `--leapsinger_root` で渡してください。

```bash
python -m svs.render examples/kirakiraboshi.ust -o out/kirakiraboshi.wav --spk_id 2
```

子音の長さは同梱の**子音長モデル**（`checkpoints/cons_dur_3singer.pt`）が予測します。話者・スタイル・前後の音素・ノート長から 1 子音ずつ推定し、`duration_model/profiles/<ckpt名>.json` の floor で下限を守ります（既定: フレーズ途中の子音は実測長の平均 − σ（σ は平均の 0.6 倍で頭打ち）、**フレーズ先頭の子音は p75**、母音は 75 ms。話者・スタイル別の値があればそちらを優先）。floor はなるべく保証し、母音 floor + 子音 floor がノート長に収まらないときだけ、区間を**音素ごとの重み比で按分**します（`phoneme_weight`・既定 1.0 = 全音素が等分）。統計量（`cons_floor_stat: pctl` で百分位も可）・母音 floor・音素ごとの重みはレシピの `duration:` で DB ごとに変えられ、`python -m duration_model.build_profile` でプロファイルに焼かれます。`--cons_dur_ckpt none` にすると全子音 80 ms の固定長にフォールバックするので、効果を A/B で確かめられます。

### タイミングの仕様

**フレーズ先頭の子音は、楽譜のノート開始時刻より前に鳴ります。** 音節核（母音）がノートの時刻ちょうどに来るよう、子音はその手前へ置かれます。フレーズの途中でも同じで、子音は前のノートへ食い込みます。歌唱合成では一般的な扱いで、SynthV の preutterance や DiffSinger の word 分割と同じ考え方です。

前倒しされる量はフレーズ先頭の子音によって変わります（日本語で中央値 35 ms・最大 273 ms、英語で中央値 110 ms・最大 470 ms）。曲頭でその余白が足りない場合は、**曲全体を必要な分だけ後ろへずらします**。出力の冒頭に無音が増えますが、楽譜との相対関係は保たれます。

出力の両端には既定で 0.25 秒の無音が入ります（`score_to_phrases` の `pau_lead_sec` / `pau_trail_sec`）。

詳しくは [guide/duration_model.md](guide/duration_model.md) を参照してください。

各段の中身は `notebooks/` を参照してください。

- `notebooks/01_svs_demo.ipynb` — 楽譜→歌声。持続長・F0・mel・波形・出力音声の再分析 mel を可視化。話者切替、スタイルの連続変化（`style_mix`）、抑揚 / ビブラートの強さの比較
- `notebooks/02_design_notes.ipynb` — 各段の補正の中身と**なぜ**そうしているか（改造する人向け）
- `notebooks/03_f0_control.ipynb` — 抑揚とビブラートの独立制御、フレーズの中で時間変化する強度、`style_mix`（音声で確かめるセルがあるので、01 と同じく LeapSinger の音響ウェイトが要ります）

## 日本語以外の言語

配布ウェイトは日本語 3 話者だけですが、次の言語パックと歌詞→音素の変換を同梱しています。データを用意すれば、日本語と同じ手順で F0 モデルと子音長モデルを学習できます。音素表は LeapSinger の同名の表と同じ中身です。

| 言語パック | 表記 | 歌詞 → 音素 | 設定の例 |
|---|---|---|---|
| `ja` | 日本語 | モーラ表（`dict/ja.mora`） | `configs/ja_example.yaml` |
| `ko` | 韓国語（Team CODA の表記） | 規則の G2P（`svs/g2p_ko.py`。連音・鼻音化・濃音化など） | `configs/ko_coda.yaml` |
| `zh_acv` | 中国語（ACV の表記） | 漢字 → 拼音（`dict/zh_pinyin.json`、熟語で多音字を読み分ける）→ 規則（`svs/g2p_zh.py`） | `configs/zh_acv.yaml` |
| `zh_opencpop` | 中国語（Opencpop の表記） | 同上 | `configs/zh_opencpop.yaml` |
| `en_full` | 英語 | 発音辞書（`dict/en_cmudict.txt`、CMUdict の最初の読み） | `configs/en_aidol.yaml`・`configs/en_aidol_notes.yaml` |

歌詞には音素をそのまま書くこともできます（英語は `[hh eh l ow]`、ほかは空白で区切った音素）。辞書に無い英単語は、音素で書いてください。

**伸ばしと音節の記号**
- 「-」「ー」と歌詞の無い音符は、前の音符の母音を伸ばします。韓国語・中国語・英語では、終声は伸ばしの最後の音符へ移ります（「밤」「-」→ b a | a M）。
- 日中韓は 1 音符に 1 音節なので、「+」も伸ばしと同じです。
- 英語では、複数の音節の単語を 1 つの音符に書き、続く「+」の音符へ音節を前から配ります（`again` `+` → ax | g eh n）。`+~`・`+*` は伸ばし、`+3` は単語の 3 番目の音素から始めます。
- `++` と、音符より音節が多いときの扱いは `lyric_style` で選べます。`sequential`（既定）は + ごとに次の 1 音節で、余った音節は最後の音符にまとめます。`count` は + の数だけ音節を歌い、余った音節は歌いません。

**伸ばしの音符と F0 モデル**: 伸ばしの記号で書いたラベル（`ay -`）で学習した F0 モデル（ckpt の `hold_symbol`）には、svs は伸ばしの母音を記号で渡します（「skies」「-」→ s k ay | - z）。それ以外のモデルには、今までどおり同じ母音を繰り返して渡します。母音を音符ごとに分けないラベルで学習した音響モデルには、`--merge_held_vowels` で伸ばしの母音を 1 つにまとめて渡せます。

## 注意

- 1 フレーム = 256/44100 秒（172.265625 fps）。LeapSinger の音響モデルと同じ格子です。
- `data.min_sec: 1.7` はビブラート分離フィルタ（filtfilt）が要る最小長に安全側の余裕を足した値です（0.3 s の FIR なら下限は 0.93 秒）。
- 旧世代（hop512・整数 MIDI 入力）のチェックポイントからの再開はできません（入力表現が別物）。
- **多言語対応の実態**: 言語ごとに差し替えるものは `dict/<lang>.yaml`（言語パック）に集めてあります。音素インベントリ/ID 空間（`<lang>.phonemes`・行順が ID・id 0 は `pau`）、核音素（`nucleus:`）、無声化母音の正規化（`vowel_normalize:`）、無音記号（`silence:` / `sil_map:`）、楽譜の読み取り（`score:` — UST の文字コード、休符/メリスマ記号、メリスマの既定母音）、歌詞→音素（`.mora`、または規則の G2P の名前 `g2p:`）、句末の終声を子音長モデルの対象にするか（`final_consonants:`）です。伸ばしの記号はパックに書かず、レシピの `hold_symbol` と学習 config の `model.hold_symbol` で足します。子音の**個数**は固定していないので別インベントリでも動きます。韓国語・中国語・英語のパックも同梱していますが、配布ウェイトは日本語 3 話者のみです。

## 謝辞

本モデルの学習に使わせていただいたデータセットと、関連するプロジェクトに感謝します。

- 御丹宮くるみ歌声データベース（御丹宮くるみ） — https://onikuru.info/db-download/
- 夏目悠李（歌声 DB 制作: アマノケイ／音声提供者: 霧野蒼太） — https://ksdcm1ng.wixsite.com/njksofficial/enunu-nnsvs
- 波音リツ — https://www.canon-voice.com/voicebanks/
- Blaauw & Bonada (2017), Applied Sciences 7(12), 1313（核音素の代表値の重み付けと、生成時のピッチ補正の 2 段目はこの論文の Appendix B を元にしています） — https://www.mdpi.com/2076-3417/7/12/1313
- DiffWave（拡散モデルの設計を参考にしました。コードは自前実装です） — https://arxiv.org/abs/2009.09761
- Rectified Flow（`model.arch: reflow` の定式化） — https://arxiv.org/abs/2209.03003
- ConvNeXt（条件付けネットワークの 1D 版。コードは自前実装です） — https://arxiv.org/abs/2201.03545
- pitch-benchmark（Lars Nieradzik。`preprocess/algorithms/` の 2 ファイルはここからの取り込みです） — https://github.com/lars76/pitch-benchmark
- RMVPE（F0 抽出モデル本体） — https://github.com/Dream-High/RMVPE
- CREPE（RMVPE 経由で `to_local_average_cents()` を利用） — https://github.com/marl/crepe
- pypinyin（中国語の G2P の漢字 → 拼音の表 `dict/zh_pinyin.json` の元データ） — https://github.com/mozillazg/python-pinyin
- The CMU Pronouncing Dictionary（英語の発音辞書 `dict/en_cmudict.txt` の元データ） — https://github.com/cmusphinx/cmudict
- GAME（openvpi。`tools/hold_labels/` が伸ばしのラベルを作るのに使う音符抽出モデル。同梱していません） — https://github.com/openvpi/GAME

配布する多話者モデルには、各データベースの規約に従って上記のクレジットを表示します。夏目悠李については **歌声 DB 制作: アマノケイ／音声提供者: 霧野蒼太** を表示し、「夏目悠李の出力音声に関する利用規約」をモデルと同じ `checkpoints/` に同梱しています（`checkpoints/CREDITS.txt` と `checkpoints/夏目悠李_利用規約.txt`）。

## ライセンス

コードは MIT です（`LICENSE`）。ただし次は MIT の対象外です。詳細は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) にあります。

- **同梱ウェイト `checkpoints/f0_3singer.pt` と `checkpoints/cons_dur_3singer.pt`** — 御丹宮くるみ・夏目悠李・波音リツの歌声データベースで学習したモデルで、各データベースの利用規約に従います（個人・非商用が前提。クレジットと URL は `checkpoints/CREDITS.txt`、夏目悠李の規約全文は `checkpoints/夏目悠李_利用規約.txt`）。再配布するときはこの 2 ファイルをモデルと一緒に配布してください。
- **同梱ボコーダ `checkpoints/nhv_v3_2_1.onnx`** — 私たちの [NHVSing](https://github.com/wavtechyukky/NHVSing)（MIT）で作ったウェイトです。コードは MIT ですが、**ウェイトは上の 2 つとは別の、より広い非商用データセット群で学習しているため商用利用できません。**データセットの一覧と各条件は NHVSing の README を参照してください。

- **`preprocess/algorithms/base.py` と `preprocess/algorithms/rmvpe.py`** — どちらも [pitch-benchmark](https://github.com/lars76/pitch-benchmark)（MIT, Copyright (c) 2025 Lars Nieradzik）からそのまま取り込んだファイルです。著作権表示は `LICENSES/pitch-benchmark-MIT.txt` にあり、再配布時はこれを保持してください。`base.py` は本リポジトリと同じ MIT です。
- `rmvpe.py` は違います。中の RMVPE モデルは [上流](https://github.com/Dream-High/RMVPE) が **Apache-2.0** なので、このファイルだけ Apache-2.0 で配布します（全文 `LICENSES/Apache-2.0.txt`）。`to_local_average_cents()` は [CREPE](https://github.com/marl/crepe) 由来で、その MIT 表示は `LICENSES/crepe-MIT.txt` にあります。ウェイト `rmvpe.pt` は初回実行時にダウンロードするもので、本リポジトリには含みません。

学習データ（歌唱 DB）と、それで学習したモデルは各配布元の規約に従ってください。
