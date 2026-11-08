# tools/hold_labels — 伸ばしのラベルを作る

**English**: [README.en.md](README.en.md)

ノートエンコーダ（`model.encoder: notes`、[guide/f0_model.md](../../guide/f0_model.md)）は、音符をまたぐ母音を音符ごとに分け、2 つ目からを伸ばしの記号で書いたラベル（`an -`）で学習すると、「wan〜」（an が 2 つの音符にまたがる）と「wan an」（言い直し）を区別できます。このツールは、そのラベルを [GAME](https://github.com/openvpi/GAME)（openvpi の音符抽出モデル）で作ります。

**要らない場合**: 同梱の日本語のデータのように、音高が変わる所で母音を必ず分けてある（`a a`）ラベルには不要です。AIdol のように、音符をまたいでも母音を 1 行で書いてあるラベルで使います。

## 準備

GAME の本体とモデルは、このリポジトリに入っていません。コードは MIT、モデルは **CC BY-NC-SA 4.0** です。

```bash
git clone https://github.com/openvpi/GAME.git ../GAME          # 確かめた版: 69ba01d
# https://github.com/openvpi/GAME/releases/tag/v1.0.0 の GAME-1.0-medium.zip を ../GAME/pretrained/ に展開
pip install -e ".[game]"                                        # GAME が使う omegaconf・pydantic
```

GAME は CPU で動きます（5 回で 1 句あたり数秒）。Apple Silicon の MPS はメモリを大きく使うので、`--device cpu`（既定）のままにしてください。

## 使い方

元のラベルで前処理したシャード（`data/<db>`）と、そのレシピを使います。

```bash
# 1. GAME を全部の句にかける（work/<db>_hold/game_raw.jsonl。止めても、もう一度走らせると続きから）
python -m tools.hold_labels.split run --recipe configs/recipes/aidol.yaml --shard data/aidol \
    --game_root ../GAME --lang en --out work/aidol_hold
# 2. 分ける母音と位置を決める（lab/・splits.json・recipe.yaml）。条件を変えて何度でも走らせられる
python -m tools.hold_labels.split write --recipe configs/recipes/aidol.yaml --shard data/aidol --out work/aidol_hold
# 3. 前処理済みのシャードに「-」を差し込む（RMVPE を回し直さない）
python -m tools.hold_labels.split shard --shard data/aidol --splits work/aidol_hold/splits.json --out data/aidol_hold
```

学習の設定には `model.hold_symbol: "-"` を書き（音素表の末尾に「-」が足されます）、ノートエンコーダなら `model.encoder: notes` も書きます。例は [configs/en_aidol_notes.yaml](../../configs/en_aidol_notes.yaml) です。3 の代わりに、2 が書いた `work/<db>_hold/recipe.yaml` で前処理をやり直すこともできます（RMVPE を回し直します）。音素と F0 は同じになりますが、lab の時刻から長さを計算し直すときの丸めで、境界がまれに 1 フレームずれます（AIdol の 2 曲で試したところ、55 句のうち 3 句・4 か所。差し込んだ「-」の所ではない）。元のシャードとそろえたいときは 3 を使ってください。

`--lang` は GAME の言語（`en`・`ja`・`yue`・`zh`、`none` で指定しない）です。

## 分ける条件

GAME には句の音声と単語の境界（母音の頭で始まり、後ろの子音までを含む）を渡し、GAME が単語の中に足した音符の境界のうち、母音の中のものを使います。次の条件をすべて満たすものだけ分けます（`write` の引数で変えられます）。

| 条件 | 既定 | 引数 |
|---|---|---|
| 候補: 母音の F0 を 2 つに分けたときの中央値の差 | 30 セント以上 | `--min_diff` |
| 母音の長さ | 200 ms より長い（短い母音は二重母音や移りの動きを音符と取り違えやすい） | `--min_len_ms` |
| 種を変えた 5 回のうち、30 ms 以内に出た境界の数 | 過半数 | （`run` の `--seeds`） |
| 母音の端と、前に採った境界からの距離 | 80 ms 以上（入りのしゃくり・語尾の落ちを音符にしない） | `--margin_ms` |
| 境界の前後 100 ms（両側 20 ms を除く）の F0 の中央値の段差 | 80 セント以上（音色の動きで分けたものを捨てる） | `--min_step` |

AIdol で無作為に選んで手で判断した 300 個の母音では、GAME（種 5 回の過半数、端の余白などの条件をかける前）が 290 個で同じ判断をしました。人の判断で上書きしたいときは `--overrides <json> ...` に `{"<句>#<音素の番号>": {"decision": "split" か "keep", "splits_sec": [句の中の秒, ...]}}` を渡します。

## 出力

| ファイル | 中身 |
|---|---|
| `game_raw.jsonl` | 句ごとの GAME の結果（母音ごとの境界、音高の差、lab の行の番号） |
| `lab/<曲>.lab` | 伸ばしのラベル（分けた行は 2 つ目からが `-`。単位・改行は元の lab と同じ） |
| `splits.json` | 分けた母音と位置（曲の中の秒と句の中の秒）、使った条件 |
| `recipe.yaml` | `lab/` を読むレシピ（`hold_symbol: "-"`） |
