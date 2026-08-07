#!/usr/bin/env python3
"""img-forge: ローカル画像生成CLI（SDXL系・Apple Silicon MPS）

品質スタック: 正解値既定(モデル別sampler/CFG/steps・clip skip) → hires fix(アニメGAN拡大)
→ 顔レタッチ(検出→部分描き直し・既定ON) → 自動採点(8枚以上で自動)

例:
  1枚焼く:   python generate.py -m femix-hassaku -p "1girl, cafe barista, green apron"
  清書:      python generate.py -m femix-hassaku -p "..." --hires
  夜間100枚: python generate.py -m femix-hassaku -p "..." -c 100 --hires --job overnight
"""
import argparse, datetime, gc, json, os, random, sys, time
from pathlib import Path

FORGE = Path(__file__).resolve().parent
REGISTRY = FORGE / "models.json"
AUX = Path(os.path.expanduser("~/AI-Models/aux"))
LORA_DIR = Path(os.path.expanduser("~/AI-Models/lora"))
ESRGAN_PATH = AUX / "RealESRGAN_x4plus_anime_6B.pth"
# 補助モデルの配布元。初回に無ければここから取りに行く（各数MB〜18MB）。
# 落とせなくても本体の生成は続行し、その機能だけ無効化する（読者環境を止めないため）
AUX_SOURCES = {
    ESRGAN_PATH: ("アニメ用GAN拡大モデル",
                  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
                  "RealESRGAN_x4plus_anime_6B.pth"),
}
# clip_skipは渡さない: diffusersのSDXL実装は最初からpenultimate層（A1111のClip skip 2相当）を使う。
# 追加でclip_skipを渡すと層がさらにズレて品質劣化・黒画像の原因になる（2026-08-02実測）
CLIP_SKIP = None
# SDXL純正VAEはfp16でNaN→黒画像を出すことがある。修正版VAEに差し替えるのが定石
VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"


def ensure_aux(path):
    """補助モデルが無ければ取得する。取れたらTrue、取れなければFalse（呼び側で機能を落とす）。"""
    if path.exists():
        return True
    label, url = AUX_SOURCES[path]
    print(f"[img-forge] {label}が未取得。ダウンロードします（初回のみ）→ {path}")
    try:
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)
        print(f"[img-forge] {label}を取得しました（{path.stat().st_size / 1e6:.1f}MB）")
        return True
    except Exception as e:
        print(f"[img-forge] ⚠ {label}を取得できませんでした（{e}）。この機能はスキップします。\n"
              f"           手動で置く場合はここへ: {path}\n"
              f"           入手先: {url}")
        return False


def load_registry():
    reg = json.loads(REGISTRY.read_text())
    return {k: v for k, v in reg.items() if not k.startswith("_")}


def make_scheduler(pipe, name):
    """A1111名→diffusersスケジューラ（公式対応表準拠）"""
    from diffusers import (DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler,
                           EulerDiscreteScheduler)
    cfg = pipe.scheduler.config
    if name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(cfg)
    if name == "euler":
        return EulerDiscreteScheduler.from_config(cfg)
    if name == "dpmpp_2m_karras":
        # 50step未満のDPM++はkarras必須＋euler_at_final（diffusers公式Tips）
        return DPMSolverMultistepScheduler.from_config(
            cfg, use_karras_sigmas=True, euler_at_final=True)
    raise ValueError(f"unknown sampler: {name}")


def build_pipeline(cfg, sampler):
    import torch
    from diffusers import StableDiffusionXLPipeline

    src, path = cfg["source"], os.path.expanduser(cfg["path"])
    if src in ("local", "hf-single"):
        pipe = StableDiffusionXLPipeline.from_single_file(path, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(path, torch_dtype=torch.float16)

    if cfg.get("vpred"):
        from diffusers import EulerDiscreteScheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, prediction_type="v_prediction", rescale_betas_zero_snr=True)
    else:
        pipe.scheduler = make_scheduler(pipe, sampler)

    from diffusers import AutoencoderKL
    pipe.vae = AutoencoderKL.from_pretrained(VAE_FIX, torch_dtype=torch.float16)

    pipe = pipe.to("mps")
    pipe.enable_attention_slicing()  # 64GB未満のMacでは公式推奨（速度も改善傾向）
    return pipe


class GanUpscaler:
    """R-ESRGAN 4x+ Anime6B（アニメ用GAN拡大）。失敗時はLANCZOSにフォールバック。

    注意: モデルをMPSに常駐させるとSDXL側のimg2imgがメモリ圧でNaN→黒画像になる
    （2026-08-02実測）。使う瞬間だけMPSへ載せ、終わったら即CPUへ降ろして
    torch.mps.empty_cache() で掃除する。
    """

    def __init__(self):
        self.model = None

    def _load(self):
        import torch
        if not ensure_aux(ESRGAN_PATH):
            raise FileNotFoundError("GAN拡大モデルが無い")  # 呼び側でLANCZOSへフォールバック
        from spandrel import ModelLoader
        self.model = ModelLoader().load_from_file(str(ESRGAN_PATH))
        self.model.eval()
        self.torch = torch

    def upscale(self, pil_img, target_w, target_h):
        from PIL import Image as PILImage
        try:
            if self.model is None:
                self._load()
            import numpy as np
            self.model.to("mps")
            t = self.torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).unsqueeze(0)
            t = (t.float() / 255.0).to("mps")
            with self.torch.no_grad():
                out = self.model(t)  # 4x
            out = (out.clamp(0, 1) * 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
            up = PILImage.fromarray(out)
            return up.resize((target_w, target_h), PILImage.LANCZOS)
        except Exception as e:
            print(f"[img-forge] GAN拡大失敗→LANCZOS代替: {e}", flush=True)
            return pil_img.resize((target_w, target_h), PILImage.LANCZOS)
        finally:
            if self.model is not None:
                self.model.to("cpu")
                self.torch.mps.empty_cache()


class FaceFixer:
    """顔検出(YOLO)→顔領域だけ高解像度で描き直して貼り戻す（ADetailer相当）"""

    def __init__(self, base_pipe, torch_mod):
        from diffusers import AutoPipelineForInpainting
        self.pipe = AutoPipelineForInpainting.from_pipe(base_pipe)  # 重み共有・追加メモリなし
        self.torch = torch_mod
        # 顔検出はMITライセンスのimgutilsを使う。
        # ultralyticsはAGPL-3.0で、importしたまま本体をMITで配ると表示が事実と食い違う
        from imgutils.detect.face import detect_faces
        self._detect_faces = detect_faces

    def fix(self, img, prompt, negative, cfg_scale, steps, seed):
        from PIL import Image as PILImage, ImageDraw
        # 返り値は [((x1, y1, x2, y2), ラベル, 信頼度), ...]
        # 検出器が動かない環境（Python 3.14ではimgutilsが内部でast.Strを使い失敗する）でも
        # 生成そのものは止めない。顔補正だけスキップする
        try:
            res = self._detect_faces(img, conf_threshold=0.3)
        except Exception as e:
            if not getattr(self, "_warned", False):
                print(f"[img-forge] ⚠ 顔検出が使えないため顔補正をスキップします（{type(e).__name__}: {e}）\n"
                      f"           Python 3.13以前の環境では動作します（--no-face-fix で警告を消せます）")
                self._warned = True
            return img, 0
        boxes = []
        img_area = img.width * img.height
        for (x1, y1, x2, y2), _label, _score in res:
            x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
            area = (x2 - x1) * (y2 - y1)
            # 極小(ノイズ)と巨大(顔アップ絵)はスキップ
            if area < img_area * 0.002 or area > img_area * 0.45:
                continue
            boxes.append((x1, y1, x2, y2))
        if not boxes:
            return img, 0
        mask = PILImage.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        for x1, y1, x2, y2 in boxes:
            pw, ph = (x2 - x1) * 0.25, (y2 - y1) * 0.25  # 25%広めに取る
            draw.rectangle([max(0, x1 - pw), max(0, y1 - ph),
                            min(img.width, x2 + pw), min(img.height, y2 + ph)], fill=255)
        mask = self.pipe.mask_processor.blur(mask, blur_factor=12)
        fixed = self.pipe(
            prompt=prompt, negative_prompt=negative,
            image=img, mask_image=mask,
            strength=0.38,  # 0.5以上は別人化（コミュニティ定石）
            guidance_scale=cfg_scale, num_inference_steps=steps,
            padding_mask_crop=32,  # 顔領域だけ切り出して高解像度で描く（ADetailerの核心）
            clip_skip=CLIP_SKIP,
            generator=self.torch.Generator("mps").manual_seed(seed),
        ).images[0]
        return fixed, len(boxes)


def score_folder(outdir):
    """アニメ美的採点（deepghs・CPU実行）→ scores.json"""
    from imgutils.metrics import anime_dbaesthetic
    scores = {}
    files = sorted(outdir.glob("*.png"))
    for p in files:
        try:
            pct = anime_dbaesthetic(str(p), fmt=("percentile",))
            scores[p.name] = round(float(pct[0] if isinstance(pct, (tuple, list)) else pct), 4)
        except Exception as e:
            print(f"[img-forge] 採点失敗 {p.name}: {e}", flush=True)
    (outdir / "scores.json").write_text(json.dumps(scores, indent=1))
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    print("[img-forge] 採点結果 上位5:", flush=True)
    for name, s in ranked[:5]:
        print(f"  {s:.2f}  {name}", flush=True)
    return scores


def main():
    ap = argparse.ArgumentParser(description="img-forge generate")
    ap.add_argument("-m", "--model", default="femix-hassaku", help="models.jsonのキー")
    ap.add_argument("-p", "--prompt", help="本文プロンプト（品質タグは自動付与）")
    ap.add_argument("--prompt-file", help="1行1プロンプトのファイル（各行×count枚）")
    ap.add_argument("-n", "--negative", default="", help="追加ネガティブ（既定に追記）")
    ap.add_argument("--no-quality", action="store_true", help="品質タグ自動付与をやめる")
    ap.add_argument("-W", "--width", type=int, default=832)
    ap.add_argument("-H", "--height", type=int, default=1216)
    ap.add_argument("--steps", type=int, default=None, help="省略時はモデル推奨値")
    ap.add_argument("--cfg", type=float, default=None, help="省略時はモデル推奨値（6超は非推奨）")
    ap.add_argument("--sampler", choices=["euler_a", "euler", "dpmpp_2m_karras"], default=None,
                    help="省略時はモデル推奨値")
    ap.add_argument("--seed", type=int, default=-1, help="-1でランダム開始（連番で+1ずつ）")
    ap.add_argument("-c", "--count", type=int, default=1, help="プロンプトごとの枚数")
    ap.add_argument("--job", default=None, help="ジョブ名（出力フォルダ名に使用）")
    ap.add_argument("--out", default=None, help="出力先（既定: output/<日付_ジョブ名>）")
    ap.add_argument("--hires", action="store_true",
                    help="2段階生成（GAN拡大→描き直し）。清書用・所要約2倍")
    ap.add_argument("--hires-scale", type=float, default=1.5)
    ap.add_argument("--hires-denoise", type=float, default=0.4)
    ap.add_argument("--no-face-fix", action="store_true", help="顔レタッチ段を切る")
    ap.add_argument("--score", action="store_true", help="採点を強制実行（8枚以上は自動）")
    ap.add_argument("--lora", action="append", default=[],
                    help="LoRA適用『名前:強度』（複数可・強度省略時0.8）。実体は~/AI-Models/lora/<名前>.safetensors")
    ap.add_argument("--vary", default=None,
                    help="微調整ガチャ: 指定画像を出発点にimg2imgで「ほぼ同じだけど少し違う」兄弟を量産する"
                         "（惜しい絵の救済。強さは--variation-strength）")
    ap.add_argument("--variation-strength", type=float, default=0.45,
                    help="--vary時の変化の強さ=img2imgのdenoise（0.35微修正〜0.6大きめ。1.0で別画像）")
    args = ap.parse_args()

    if not args.prompt and not args.prompt_file:
        ap.error("-p か --prompt-file のどちらかは必須")

    registry = load_registry()
    if args.model not in registry:
        sys.exit(f"モデル '{args.model}' は未登録。候補: {', '.join(registry)}")
    cfg = registry[args.model]
    sampler = args.sampler or cfg.get("sampler", "euler_a")
    steps = args.steps or cfg.get("steps", 26)
    cfg_scale = args.cfg or cfg.get("cfg", 5.5)

    prompts = []
    if args.prompt:
        prompts.append(args.prompt.strip())
    if args.prompt_file:
        prompts += [l.strip() for l in Path(args.prompt_file).read_text().splitlines()
                    if l.strip() and not l.startswith("#")]

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    job = args.job or "quick"
    outdir = Path(os.path.expanduser(args.out)) if args.out else FORGE / "output" / f"{stamp}_{job}"
    outdir.mkdir(parents=True, exist_ok=True)

    base_seed = args.seed if args.seed >= 0 else random.randrange(2**31)
    total = len(prompts) * args.count
    print(f"[img-forge] model={args.model} sampler={sampler} steps={steps} cfg={cfg_scale} "
          f"hires={args.hires} facefix={not args.no_face_fix} total={total}枚 out={outdir}", flush=True)

    t0 = time.time()
    pipe = build_pipeline(cfg, sampler)
    print(f"[img-forge] モデル読み込み {time.time()-t0:.0f}s", flush=True)

    import torch

    if args.lora:
        names, weights = [], []
        for spec in args.lora:
            n, _, w = spec.partition(":")
            path = LORA_DIR / f"{n}.safetensors"
            if not path.exists():
                sys.exit(f"LoRAが見つからない: {path}（~/AI-Models/lora/ に置く）")
            adapter = n.replace("-", "_")
            pipe.load_lora_weights(str(path), adapter_name=adapter)
            names.append(adapter)
            weights.append(float(w) if w else 0.8)
        if sum(weights) >= 2.0:
            print(f"[img-forge] ⚠️ LoRA合計weight {sum(weights):.1f}（2.0以上は焦げやすい）", flush=True)
        pipe.set_adapters(names, adapter_weights=weights)
        print(f"[img-forge] LoRA適用: {list(zip(names, weights))}", flush=True)
    hires_pipe, gan = None, None
    if args.hires:
        from diffusers import StableDiffusionXLImg2ImgPipeline
        hires_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        gan = GanUpscaler()
    face_fixer = None
    if not args.no_face_fix:
        try:
            face_fixer = FaceFixer(pipe, torch)
        except Exception as e:
            print(f"[img-forge] ⚠ 顔レタッチを初期化できませんでした（{e}）。顔補正なしで続行します")

    # 微調整ガチャ（--vary）: 指定画像を出発点に、低denoiseのimg2imgで兄弟を量産する。
    # 構図・色は元画像から引き継がれ、シード違いで細部だけ変わる（Fooocusの"Vary"と同方式）。
    # ※txt2imgのlatents指定によるslerp補間方式は、この環境（diffusers+MPS）で
    #   latents引数自体が壊れた絵を返すため不採用（2026-08-03実測）
    vary_img, vary_pipe = None, None
    if args.vary:
        from PIL import Image as PILImage
        from diffusers import StableDiffusionXLImg2ImgPipeline
        src = Path(os.path.expanduser(args.vary))
        if not src.exists():
            sys.exit(f"--varyの画像が見つからない: {src}")
        vary_img = PILImage.open(src).convert("RGB").resize((args.width, args.height))
        vary_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        print(f"[img-forge] 微調整ガチャ: {src.name} denoise={args.variation_strength}", flush=True)

    manifest = (outdir / "manifest.jsonl").open("a")
    done = 0
    for pi, body in enumerate(prompts):
        full = body if args.no_quality else f"{cfg['quality']}, {body}"
        neg = cfg["negative"] + (", " + args.negative if args.negative else "")
        for ci in range(args.count):
            seed = base_seed + pi * 10000 + ci
            t1 = time.time()
            if vary_pipe is not None:
                img = vary_pipe(
                    full, negative_prompt=neg,
                    image=vary_img, strength=args.variation_strength,
                    guidance_scale=cfg_scale, num_inference_steps=steps,
                    clip_skip=CLIP_SKIP,
                    generator=torch.Generator("mps").manual_seed(seed),
                ).images[0]
            else:
                img = pipe(
                    full, negative_prompt=neg,
                    width=args.width, height=args.height,
                    guidance_scale=cfg_scale, num_inference_steps=steps,
                    clip_skip=CLIP_SKIP,
                    # 低解像度学習画像の特徴から遠ざける負条件（diffusers公式Tips）
                    negative_original_size=(512, 512),
                    negative_target_size=(args.width, args.height),
                    generator=torch.Generator("mps").manual_seed(seed),
                ).images[0]
            if hires_pipe is not None:
                hw = int(args.width * args.hires_scale) // 8 * 8
                hh = int(args.height * args.hires_scale) // 8 * 8
                img = hires_pipe(
                    prompt=full, negative_prompt=neg,
                    image=gan.upscale(img, hw, hh),
                    strength=args.hires_denoise,
                    guidance_scale=cfg_scale, num_inference_steps=steps,
                    clip_skip=CLIP_SKIP,
                    generator=torch.Generator("mps").manual_seed(seed),
                ).images[0]
            n_faces = 0
            if face_fixer is not None:
                img, n_faces = face_fixer.fix(img, full, neg, cfg_scale, steps, seed)
            name = (f"{args.model}_p{pi:02d}_s{seed}_vary.png" if vary_pipe is not None
                    else f"{args.model}_p{pi:02d}_s{seed}.png")
            # A1111互換の生成パラメータをPNG自体に記録（コピーしても情報が残る）
            from PIL.PngImagePlugin import PngInfo
            pnginfo = PngInfo()
            params = (f"{full}\nNegative prompt: {neg}\n"
                      f"Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg_scale}, "
                      f"Seed: {seed}, Size: {args.width}x{args.height}, Model: {args.model}")
            if args.hires:
                params += (f", Hires upscale: {args.hires_scale}, "
                           f"Denoising strength: {args.hires_denoise}")
            if args.lora:
                params += f", Lora: {' '.join(args.lora)}"
            pnginfo.add_text("parameters", params)
            img.save(outdir / name, pnginfo=pnginfo)
            done += 1
            manifest.write(json.dumps({
                "file": name, "model": args.model, "prompt": full, "negative": neg,
                "seed": seed, "size": f"{args.width}x{args.height}",
                "steps": steps, "cfg": cfg_scale, "sampler": sampler,
                "hires": args.hires, "face_fixed": n_faces, "lora": args.lora,
                "vary_from": args.vary, "vary_strength": args.variation_strength if args.vary else None,
            }, ensure_ascii=False) + "\n")
            manifest.flush()
            print(f"[img-forge] {done}/{total} {name} 顔補正{n_faces}箇所 ({time.time()-t1:.0f}s)",
                  flush=True)
            if done % 20 == 0:
                gc.collect(); torch.mps.empty_cache()

    print(f"[img-forge] 生成完走 {total}枚 / {time.time()-t0:.0f}s → {outdir}", flush=True)
    if args.score or total >= 8:
        score_folder(outdir)


if __name__ == "__main__":
    main()
