# 依存ライブラリとライセンス

img-forge 本体は MIT ライセンスです。動作には以下のパッケージが必要で、それぞれのライセンスは下表のとおりです（`pip install -r requirements.txt` で入るもの。配布物には同梱していません）。

2026-08-07 時点で各パッケージのメタデータを確認した内容です。

| パッケージ | 用途 | ライセンス |
|---|---|---|
| torch / torchvision | 計算の土台 | BSD-3-Clause |
| diffusers | 画像生成パイプライン | Apache-2.0 |
| transformers | テキストエンコーダ | Apache-2.0 |
| accelerate | デバイス配置 | Apache-2.0 |
| safetensors | モデルファイルの読み込み | Apache-2.0 |
| peft | LoRA適用 | Apache-2.0 |
| huggingface_hub | モデル取得 | Apache-2.0 |
| flask | ダッシュボード | BSD-3-Clause |
| pillow | 画像処理 | MIT-CMU |
| numpy | 数値計算 | BSD-3-Clause |
| spandrel | GAN拡大モデルの読み込み | MIT |
| dghs-imgutils | 顔検出・美的採点 | MIT |

## 意図的に使っていないもの

- **ultralytics（AGPL-3.0）** — 顔検出でよく使われるライブラリですが、**AGPL-3.0 は本体の MIT ライセンスと整合しません**。AGPL のコードを取り込んだものを配布すると、全体を AGPL で公開する義務が生じるという解釈が一般的です。「MIT だから自由に使える」と受け取った方が、それを自分の製品に組み込んだときに困ることになります。そのため顔検出は MIT の `dghs-imgutils` に寄せています
- **spandrel_extra_arches** — 中身に非商用ライセンスのアーキテクチャが多数含まれるため、依存に加えません

## 実行時にダウンロードされるモデル

以下は配布物に含まれず、初回実行時に自動取得されます。**それぞれ別のライセンスです。**

| モデル | 取得元 | ライセンス |
|---|---|---|
| sdxl-vae-fp16-fix | Hugging Face (madebyollin) | MIT |
| RealESRGAN_x4plus_anime_6B | GitHub (xinntao/Real-ESRGAN) | BSD-3-Clause |
| アニメ顔検出モデル | Hugging Face (deepghs) | 配布ページを参照 |
| 美的採点モデル | Hugging Face (deepghs) | 配布ページを参照（一部 openrail・用途制限あり） |

生成に使う **SDXL系モデル本体のライセンスは、これらとは別です。** `models.json` の `license` 欄と、各モデルの配布ページを必ず確認してください。
