#!/usr/bin/env python3
"""img-forge: モデル追加ツール（Civitai / Hugging Face → ~/AI-Models → models.json登録）

例:
  Civitai:      python fetch_model.py --name wai14 --civitai-version 123456
  HuggingFace:  python fetch_model.py --name hassaku --hf-repo "Raelina/..." --hf-file "model.safetensors"
  登録だけ:     python fetch_model.py --name mymodel --local ~/AI-Models/foo.safetensors

Civitaiのversion IDは、モデルページ右側の「Download」ボタンのリンク
（civitai.com/api/download/models/XXXXX）のXXXXXの部分。
ログイン必須モデルは環境変数 CIVITAI_TOKEN にAPIキーを設定（civitai.com/user/account で発行）。
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

FORGE = Path(__file__).resolve().parent
REGISTRY = FORGE / "models.json"
MODELS_DIR = Path(os.path.expanduser("~/AI-Models"))

DEFAULT_QUALITY = "masterpiece, best quality"
DEFAULT_NEG = ("worst quality, low quality, lowres, bad anatomy, bad hands, extra digits, "
               "missing fingers, watermark, signature, username, blurry, jpeg artifacts")


def download(url, dest, headers=None):
    cmd = ["curl", "-L", "--fail", "--progress-bar", "-o", str(dest)]
    for h in headers or []:
        cmd += ["-H", h]
    cmd.append(url)
    print(f"[fetch] {url}\n     → {dest}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="img-forge fetch_model")
    ap.add_argument("--name", required=True, help="models.jsonに登録するキー名")
    ap.add_argument("--label", default=None, help="表示名（省略時はname）")
    ap.add_argument("--civitai-version", help="CivitaiのモデルversionID")
    ap.add_argument("--hf-repo", help="HuggingFaceリポジトリID")
    ap.add_argument("--hf-file", help="リポジトリ内のsafetensorsファイル名")
    ap.add_argument("--local", help="既にあるローカルファイルを登録するだけ")
    ap.add_argument("--quality", default=DEFAULT_QUALITY, help="品質タグ（モデル配布ページ推奨に合わせる）")
    ap.add_argument("--negative", default=DEFAULT_NEG)
    ap.add_argument("--vpred", action="store_true", help="V-predictionモデルの場合に指定")
    ap.add_argument("--civitai-id", help="CivitaiのモデルID（追加済み判定と配布ページのリンクに使う）")
    ap.add_argument("--license", default="配布ページで確認すること",
                    help="ライセンス（配布ページの記載。生成画像の商用可否を含めて書く）")
    ap.add_argument("--popularity", default=None, help="人気度メモ（DL数・👍等）")
    ap.add_argument("--as-lora", action="store_true",
                    help="LoRAとして~/AI-Models/lora/へ保存（台帳登録なし・generate.py --lora 名前 で使用）")
    args = ap.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    if args.as_lora:
        lora_dir = MODELS_DIR / "lora"
        lora_dir.mkdir(exist_ok=True)
        path = lora_dir / f"{args.name}.safetensors"
        if args.civitai_version:
            url = f"https://civitai.com/api/download/models/{args.civitai_version}"
            headers = []
            token = os.environ.get("CIVITAI_TOKEN")
            if token:
                headers.append(f"Authorization: Bearer {token}")
            download(url, path, headers)
        elif args.hf_repo and args.hf_file:
            download(f"https://huggingface.co/{args.hf_repo}/resolve/main/{args.hf_file}", path)
        else:
            sys.exit("--as-loraは--civitai-versionか--hf-repo+--hf-fileと併用")
        size_mb = path.stat().st_size / 1e6
        if size_mb < 5:
            sys.exit(f"⚠️ {size_mb:.0f}MBしかない。DL失敗（ログイン必須等）の可能性。中身を確認して。")
        print(f"[fetch] LoRA保存完了: {path.name} ({size_mb:.0f}MB) → generate.py --lora {args.name}:0.8 で使用可")
        return

    if args.local:
        path = Path(os.path.expanduser(args.local)).resolve()
        if not path.exists():
            sys.exit(f"ファイルが見つからない: {path}")
    elif args.civitai_version:
        path = MODELS_DIR / f"{args.name}.safetensors"
        url = f"https://civitai.com/api/download/models/{args.civitai_version}"
        headers = []
        token = os.environ.get("CIVITAI_TOKEN")
        if token:
            headers.append(f"Authorization: Bearer {token}")
        download(url, path, headers)
    elif args.hf_repo and args.hf_file:
        path = MODELS_DIR / args.hf_file
        download(f"https://huggingface.co/{args.hf_repo}/resolve/main/{args.hf_file}", path)
    else:
        sys.exit("--civitai-version / --hf-repo+--hf-file / --local のいずれかを指定")

    size_gb = path.stat().st_size / 1e9
    if size_gb < 1:
        sys.exit(f"⚠️ ファイルが{size_gb:.1f}GBしかない。DL失敗（ログイン必須モデル等）の可能性。中身を確認して。")

    reg = json.loads(REGISTRY.read_text())
    reg[args.name] = {
        "label": args.label or args.name,
        "source": "local",
        "path": str(path).replace(os.path.expanduser("~"), "~"),
        "license": args.license,
        "quality": args.quality,
        "negative": args.negative,
        "vpred": args.vpred,
    }
    if args.popularity:
        reg[args.name]["popularity"] = args.popularity
    if args.civitai_id:
        reg[args.name]["civitaiId"] = str(args.civitai_id)
        reg[args.name]["page"] = f"https://civitai.com/models/{args.civitai_id}"
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n")
    print(f"[fetch] 登録完了: {args.name} ({size_gb:.1f}GB) → generate.py -m {args.name} で使用可")


if __name__ == "__main__":
    main()
