# MOSS-Image 絵柄 LoRA

MOSS-Image（SFT、FLUX.2 VAE、Qwen3.5 9B テキストエンコーダー）を、今の絵柄に寄せるための LoRA の学習と評価に使ったスクリプトと設定。

## ファイル

| ファイル | 役割 |
|---|---|
| `dataset_scan.py` | rating=good の Generation を refines の chain ごとに辿り、`good_chains.json` に書く |
| `dataset_extract.py` | `good_chains.json` から 1 絵 1 枚（deliver の装飾が乗る前の段階）を選んで画像を取得し、manifest を書く |
| `moss_style_v1.yaml` / `moss_style_v2.yaml` | ai-toolkit の学習設定。v1 は 512/768 で 2500 steps、v2 は 1024 で 2000 steps |
| `run.py` | ComfyUI に MOSS のグラフを投げて保存する。arm ごとに絵柄 LoRA・強さ・4-step LoRA・hires・画像参照を切り替える |
| `ingest.py` | `run.py` の出力を chimera に import し、semantic とタグを付ける |
| `arms/` | 試した arm の定義（プロンプトと設定） |

学習データ（画像と説明文）と学習済み LoRA は git に入れず、非公開の R2 に置いている。

## 分かったこと

- 画像参照（`TextEncodeQwenImageEdit`、プロンプト先頭に vision トークンが要る）で移るのは中身だけで、絵柄は移らない。
- プロンプトにはキャラの見た目の説明が要る。無いと別人になる。
- 絵師タグ（`@oshiki hitoshi`）は効かない。
- 絵柄 LoRA は強さ 1.0 がいちばん今の絵柄に近い。強めると oshiki 単体を強めたような絵になり、目の下の影も濃くなる。
- 4-step LoRA（6 steps、CFG 1）に 1280x2048 の hires（denoise 0.45）を足すと、1 枚 40〜50 秒で線が締まる。
- 学習データが足りず、崩壊が多い。
