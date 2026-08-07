#!/usr/bin/env python3
"""img-forge: モデル・LoRAの「味見サンプル」を焼く（未作成分だけ）

同じプロンプト・同じシードで焼くので、テイストの違いだけが見える。
モデル/LoRAを追加したら再実行すれば足りない分だけ焼かれる。
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import LORA_DIR, build_pipeline, load_registry

FORGE = Path(__file__).resolve().parent
SAMPLES = FORGE / "samples"
SAMPLES.mkdir(exist_ok=True)

PROMPT = "1girl, solo, upper body, gentle smile, holding coffee cup, cafe interior, soft lighting"
SEED = 123
W, H, STEPS = 640, 832, 16


def bake(pipe, cfg, dest, torch):
    img = pipe(
        f"{cfg['quality']}, {PROMPT}", negative_prompt=cfg["negative"],
        width=W, height=H, guidance_scale=cfg.get("cfg", 5.5),
        num_inference_steps=STEPS,
        generator=torch.Generator("mps").manual_seed(SEED),
    ).images[0]
    img.save(dest)


def main():
    import torch
    registry = load_registry()

    # ① モデルの見本
    for name, cfg in registry.items():
        dest = SAMPLES / f"model_{name}.png"
        if dest.exists():
            continue
        t0 = time.time()
        print(f"[samples] モデル見本: {name}", flush=True)
        try:
            pipe = build_pipeline(cfg, cfg.get("sampler", "euler_a"))
            pipe.set_progress_bar_config(disable=True)
            bake(pipe, cfg, dest, torch)
            print(f"[samples] {name} 完了 ({time.time()-t0:.0f}s)", flush=True)
            del pipe
            import gc; gc.collect(); torch.mps.empty_cache()
        except Exception as e:
            print(f"[samples] {name} 失敗: {e}", flush=True)

    # ② LoRAの見本（既定モデルに1個ずつ挿して焼く）
    base_name = next(iter(registry))
    loras = sorted(LORA_DIR.glob("*.safetensors")) if LORA_DIR.exists() else []
    missing = [p for p in loras if not (SAMPLES / f"lora_{p.stem}.png").exists()]
    if missing:
        cfg = registry[base_name]
        pipe = build_pipeline(cfg, cfg.get("sampler", "euler_a"))
        pipe.set_progress_bar_config(disable=True)
        for p in missing:
            t0 = time.time()
            adapter = p.stem.replace("-", "_")
            print(f"[samples] LoRA見本: {p.stem}（ベース={base_name}）", flush=True)
            try:
                pipe.load_lora_weights(str(p), adapter_name=adapter)
                pipe.set_adapters([adapter], adapter_weights=[0.9])
                bake(pipe, cfg, SAMPLES / f"lora_{p.stem}.png", torch)
                pipe.disable_lora()
                print(f"[samples] {p.stem} 完了 ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"[samples] {p.stem} 失敗（形式非対応の可能性）: {e}", flush=True)

    print("[samples] 全見本そろった", flush=True)


if __name__ == "__main__":
    main()
