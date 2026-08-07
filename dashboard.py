#!/usr/bin/env python3
"""img-forge ダッシュボード（ローカル専用・手動起動）

起動:  .venv/bin/python dashboard.py   → http://localhost:3942
       ポートを変えたい場合: IMG_FORGE_PORT=4000 .venv/bin/python dashboard.py
停止:  Ctrl+C（またはプロセスkill）

タブ: 生成 / レビュー（採用・ボツ振り分け） / モデル（手持ち一覧・Civitaiカタログ検索→追加DL）
生成・DLジョブは共通FIFOキューで1本ずつ実行（GPU/回線の取り合い防止）。
レビューの「ボツ」は output/_trash/ へ移動（即削除はしない）。採用は output/_picks/ へコピー。
"""
import datetime, json, os, random, shutil, subprocess, threading, urllib.parse, urllib.request
from pathlib import Path

from flask import Flask, jsonify, request, send_file, abort

FORGE = Path(__file__).resolve().parent
OUTPUT = FORGE / "output"
PICKS = OUTPUT / "_picks"
TRASH = OUTPUT / "_trash"
# 外部の画像置き場もギャラリーに映す（パスは "@名前/ファイル" で参照）
# 例: {"icons": Path(os.path.expanduser("~/Pictures/my-icons"))} と書くと
#     そのフォルダの画像もレビュータブに並ぶ。不要なら空のままでよい。
EXTRA_DIRS = {}
JOBS = FORGE / "jobs"
REGISTRY = FORGE / "models.json"
PYTHON = FORGE / ".venv" / "bin" / "python"

app = Flask(__name__)
queue, current, lock = [], {"job": None}, threading.Lock()


def worker():
    import time
    while True:
        with lock:
            job = queue.pop(0) if queue else None
            current["job"] = job
        if not job:
            time.sleep(1)
            continue
        log = open(job["log"], "w")
        proc = subprocess.Popen(job["cmd"], stdout=log, stderr=subprocess.STDOUT)
        job["pid"] = proc.pid
        proc.wait()
        job["done"] = True


threading.Thread(target=worker, daemon=True).start()


def enqueue(cmd, jobname, summary):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    jid = f"{stamp}_{jobname}"
    JOBS.mkdir(exist_ok=True)
    job = {"id": jid, "cmd": [str(c) for c in cmd], "log": str(JOBS / f"{jid}.log"),
           "done": False, "summary": summary}
    with lock:
        queue.append(job)
    return jid, len(queue)


def load_registry():
    reg = json.loads(REGISTRY.read_text())
    return {k: v for k, v in reg.items() if not k.startswith("_")}


# ---------- 生成 ----------

@app.get("/api/registry")
def registry():
    return jsonify(load_registry())


@app.post("/api/submit")
def submit():
    d = request.json
    jobname = (d.get("job") or "dash").strip() or "dash"
    cmd = [PYTHON, FORGE / "generate.py",
           "-m", d["model"], "-p", d["prompt"],
           "-W", int(d.get("width", 832)), "-H", int(d.get("height", 1216)),
           "--steps", int(d.get("steps", 26)), "--cfg", float(d.get("cfg", 6.0)),
           "-c", int(d.get("count", 1)), "--job", jobname,
           "--seed", int(d.get("seed", -1))]
    if d.get("negative"):
        cmd += ["-n", d["negative"]]
    if d.get("hires"):
        cmd += ["--hires"]
    if d.get("facefix") is False:
        cmd += ["--no-face-fix"]
    for spec in d.get("loras") or []:
        cmd += ["--lora", spec]
    jid, qlen = enqueue(cmd, jobname, f"生成 {d['model']} x{d.get('count', 1)}")
    return jsonify({"id": jid, "queued": qlen})


@app.get("/api/state")
def state():
    job = current["job"]
    tail = ""
    if job and Path(job["log"]).exists():
        tail = "\n".join(Path(job["log"]).read_text().splitlines()[-8:])
    with lock:
        q = [j["summary"] for j in queue]
    return jsonify({"running": (None if not job or job.get("done") else
                                {"id": job["id"], "summary": job["summary"]}),
                    "queue": q, "log": tail})


# ---------- レビュー ----------

_dim_cache = {}  # path -> (mtime, w, h)。ヘッダ読みのみで高速だがrglob毎回は重いので記憶


def dims_of(p):
    try:
        mt = p.stat().st_mtime
        hit = _dim_cache.get(str(p))
        if hit and hit[0] == mt:
            return hit[1], hit[2]
        from PIL import Image
        with Image.open(p) as im:
            w, h = im.size
        _dim_cache[str(p)] = (mt, w, h)
        return w, h
    except Exception:
        return None, None


@app.get("/api/jobs")
def jobs():
    """バッチ（ジョブフォルダ）一覧。レビュー画面の日付ジャンプ用"""
    out = []
    for d in OUTPUT.iterdir() if OUTPUT.exists() else []:
        if not d.is_dir() or d.name in ("_trash", "_picks"):
            continue
        pngs = list(d.glob("*.png"))
        if not pngs:
            continue
        out.append({"job": d.name, "count": len(pngs),
                    "mtime": max(p.stat().st_mtime for p in pngs)})
    out.sort(key=lambda j: j["mtime"], reverse=True)
    n_picks = len(list(PICKS.rglob("*.png"))) if PICKS.exists() else 0
    return jsonify({"jobs": out, "picks": n_picks})


@app.get("/api/images")
def images():
    limit = int(request.args.get("limit", 120))
    which = request.args.get("view", "all")  # all | picks
    job_filter = request.args.get("job", "")  # ジョブフォルダ名で絞り込み（空=全部）
    # ジョブフォルダごとのscores.json（自動採点結果）を拾う
    score_maps = {}

    def score_of(p):
        d = p.parent
        if d not in score_maps:
            sj = d / "scores.json"
            try:
                score_maps[d] = json.loads(sj.read_text()) if sj.exists() else {}
            except Exception:
                score_maps[d] = {}
        return score_maps[d].get(p.name)

    entries = []  # (mtime, relpath, picked, score, job, path)
    def job_hit(jobname):
        if not job_filter:
            return True
        if job_filter.startswith("date:"):  # date:YYYYMMDD → その日の全バッチ
            return jobname.startswith(job_filter[5:])
        return jobname == job_filter

    if which == "picks":
        # 採用コピーを元のバッチに紐付け直す（日付・バッチ絞り込みと点数を効かせる）
        orig_job = {}
        for d in OUTPUT.iterdir() if OUTPUT.exists() else []:
            if d.is_dir() and d.name not in ("_trash", "_picks"):
                for q in d.glob("*.png"):
                    orig_job[q.name] = d.name
        for p in PICKS.rglob("*.png"):
            jobname = orig_job.get(p.name, "採用済み")
            if not job_hit(jobname):
                continue
            entries.append((p.stat().st_mtime, str(p.relative_to(OUTPUT)), True,
                            score_of(OUTPUT / jobname / p.name), jobname, p))
    else:
        for p in OUTPUT.rglob("*.png"):
            rel = p.relative_to(OUTPUT)
            if rel.parts[0] in ("_trash", "_picks"):
                continue
            jobname = rel.parts[0] if len(rel.parts) > 1 else "その他"
            if not job_hit(jobname):
                continue
            entries.append((p.stat().st_mtime, str(rel), (PICKS / p.name).exists(),
                            score_of(p), jobname, p))
        for tag, base in EXTRA_DIRS.items():
            if job_filter and job_filter != f"@{tag}":
                continue
            if base.exists():
                for p in base.glob("*.png"):
                    entries.append((p.stat().st_mtime, f"@{tag}/{p.name}",
                                    (PICKS / p.name).exists(), None, f"@{tag}", p))
    entries.sort(key=lambda e: e[0], reverse=True)
    total = len(entries)
    if limit <= 0:  # 0以下=無制限（絞り込み時に全部出す用）
        limit = total
    items = []
    for _, rel, picked, score, jobname, p in entries[:limit]:
        w, h = dims_of(p)
        items.append({"path": rel, "picked": picked, "score": score,
                      "job": jobname, "w": w, "h": h})
    return jsonify({"total": total, "items": items})


def _parse_a1111(raw):
    """自前で書いたA1111互換parametersテキストを分解する"""
    lines = raw.split("\n")
    prompt, negative, tail = raw, "", ""
    for i, l in enumerate(lines):
        if l.startswith("Negative prompt: "):
            prompt = "\n".join(lines[:i])
            rest = "\n".join(lines[i:])
            neg_part, _, tail = rest.partition("\nSteps: ")
            negative = neg_part[len("Negative prompt: "):]
            tail = "Steps: " + tail if tail else ""
            break
    kv = {}
    if tail:
        import re
        for m in re.finditer(r"([A-Z][\w ]*?): ([^,]+)(?:, |$)", tail):
            kv[m.group(1)] = m.group(2).strip()
    return prompt, negative, kv


def collect_meta(p):
    """画像1枚の生成パラメータを集める（PNGチャンク優先→manifest.jsonlフォールバック）。
    見つからなければNone。呼び戻し用に品質タグ・既定ネガを剥がしたbody/extraNegativeも付ける"""
    entry = None
    from PIL import Image
    try:
        with Image.open(p) as im:
            raw = (im.text or {}).get("parameters")
    except Exception:
        raw = None
    if raw:
        prompt, negative, kv = _parse_a1111(raw)
        entry = {"prompt": prompt, "negative": negative,
                 "model": kv.get("Model"), "seed": kv.get("Seed"),
                 "steps": kv.get("Steps"), "cfg": kv.get("CFG scale"),
                 "size": kv.get("Size"), "sampler": kv.get("Sampler"),
                 "hires": "Hires upscale" in kv,
                 "lora": kv.get("Lora", "").split() if kv.get("Lora") else []}
    else:
        # 旧画像: 同フォルダ（_picksなら全ジョブ横断）のmanifest.jsonlから探す
        candidates = [p.parent / "manifest.jsonl"]
        if p.parent in (PICKS, TRASH):
            candidates = sorted(OUTPUT.glob("*/manifest.jsonl"), reverse=True)
        for mf in candidates:
            if not mf.exists():
                continue
            for line in mf.read_text().splitlines():
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("file") == p.name:
                    entry = {"prompt": d.get("prompt", ""), "negative": d.get("negative", ""),
                             "model": d.get("model"), "seed": str(d.get("seed", "")),
                             "steps": str(d.get("steps", "")), "cfg": str(d.get("cfg", "")),
                             "size": d.get("size"), "sampler": d.get("sampler"),
                             "hires": bool(d.get("hires")),
                             "lora": d.get("lora") or []}
                    break
            if entry:
                break
    if not entry:
        return None
    # 品質タグ・既定ネガを剥がして「本文だけ」も返す（フォーム呼び戻し用）
    reg = load_registry().get(entry.get("model") or "", {})
    body, extra_neg = entry["prompt"], entry["negative"]
    q = reg.get("quality", "")
    if q and body.startswith(q):
        body = body[len(q):].lstrip(", ")
    dn = reg.get("negative", "")
    if dn and extra_neg.startswith(dn):
        extra_neg = extra_neg[len(dn):].lstrip(", ")
    entry.update({"found": True, "body": body, "extraNegative": extra_neg})
    return entry


@app.get("/api/meta")
def image_meta():
    p = resolve_img(request.args.get("path", ""))
    if p is None:
        abort(404)
    entry = collect_meta(p)
    return jsonify(entry if entry else {"found": False})


@app.post("/api/variation")
def variation():
    """当たり画像の微調整ガチャ: シード固定＋ノイズslerp補間で兄弟を量産する"""
    d = request.json or {}
    p = resolve_img(d.get("path", ""))
    if p is None:
        abort(404)
    meta = collect_meta(p)
    if not meta or not meta.get("seed") or not meta.get("model"):
        return jsonify({"error": "生成パラメータが読めない画像（外部画像など）は兄弟を作れない"}), 400
    if meta["model"] not in load_registry():
        return jsonify({"error": f"モデル {meta['model']} が未登録"}), 400
    count = max(2, min(int(d.get("count", 8)), 16))
    strength = max(0.2, min(float(d.get("strength", 0.45)), 0.7))  # img2imgのdenoise
    try:
        w, h = (meta.get("size") or "832x1216").split("x")
    except ValueError:
        w, h = 832, 1216
    # hiresは付けない（探索は速度優先。当たりが出たら設定呼び戻し→hiresで清書する運用）
    cmd = [PYTHON, FORGE / "generate.py", "-m", meta["model"],
           "-p", meta.get("body") or meta["prompt"],
           "-W", int(w), "-H", int(h), "--seed", int(meta["seed"]) + 1, "-c", count,
           "--vary", str(p), "--variation-strength", strength,
           "--job", f"var-{p.stem[:24]}"]
    if meta.get("extraNegative"):
        cmd += ["-n", meta["extraNegative"]]
    # img2imgは「ステップ数×強さ」の回数しか描き直さないため、実行回数が
    # モデル推奨ステップを下回らないよう逆算して増やす（低ステップ元画像の溶け防止）
    reg_steps = load_registry()[meta["model"]].get("steps", 26)
    want = max(int(float(meta.get("steps") or reg_steps)), reg_steps)
    cmd += ["--steps", min(int(want / strength + 0.999), 70)]
    if meta.get("cfg"):
        cmd += ["--cfg", float(meta["cfg"])]
    for spec in meta.get("lora") or []:
        cmd += ["--lora", spec]
    jid, qlen = enqueue(cmd, f"var-{p.stem[:16]}", f"兄弟{count}枚（差{strength}） {p.name}")
    return jsonify({"id": jid, "queued": qlen, "count": count})


def resolve_img(rel: str):
    """通常出力と@外部置き場の両対応でパスを解決（範囲外はNone）"""
    if rel.startswith("@"):
        tag, _, name = rel[1:].partition("/")
        base = EXTRA_DIRS.get(tag)
        if not base:
            return None
        p = (base / name).resolve()
        return p if str(p).startswith(str(base.resolve())) and p.exists() else None
    p = (OUTPUT / rel).resolve()
    return p if str(p).startswith(str(OUTPUT.resolve())) and p.exists() else None


@app.post("/api/review")
def review():
    d = request.json
    src = resolve_img(d["path"])
    if src is None:
        abort(404)
    if d["action"] == "keep":
        PICKS.mkdir(exist_ok=True)
        shutil.copy2(src, PICKS / src.name)
    elif d["action"] == "unkeep":  # 採用の取り消し（_picksのコピーを外すだけ・元画像は残る）
        f = PICKS / src.name
        if f.exists():
            f.unlink()
    elif d["action"] == "trash":
        TRASH.mkdir(exist_ok=True)
        shutil.move(str(src), TRASH / src.name)
    else:
        abort(400)
    return jsonify({"ok": True})


@app.get("/img/<path:rel>")
def img(rel):
    p = resolve_img(rel)
    if p is None:
        abort(404)
    return send_file(p)


# ---------- スタイル（当たりプリセット） ----------

STYLES = FORGE / "styles.json"


def load_styles():
    try:
        return json.loads(STYLES.read_text())
    except Exception:
        return {}


@app.get("/api/styles")
def styles_list():
    return jsonify(load_styles())


@app.post("/api/styles/save")
def styles_save():
    d = request.json
    name = (d.get("name") or "").strip()
    if not name:
        abort(400)
    styles = load_styles()
    styles[name] = {"model": d.get("model"), "loras": d.get("loras") or [],
                    "styleTags": d.get("styleTags", ""), "negative": d.get("negative", "")}
    STYLES.write_text(json.dumps(styles, ensure_ascii=False, indent=2) + "\n")
    return jsonify({"ok": True, "count": len(styles)})


@app.post("/api/styles/delete")
def styles_delete():
    name = (request.json or {}).get("name")
    styles = load_styles()
    if name not in styles:
        abort(404)
    styles.pop(name)
    STYLES.write_text(json.dumps(styles, ensure_ascii=False, indent=2) + "\n")
    return jsonify({"ok": True})


# ---------- XYZスイープ（新モデルの品定め） ----------
# 設定を総当たりで焼いて比較表にする実験装置。X軸・Y軸に model / cfg / steps を割り当て、
# シード固定で1セル1枚ずつキューに積む。結果は /sweep/<id> の比較ページで見る。

AXIS_JP = {"model": "モデル", "cfg": "CFG", "steps": "ステップ"}


@app.post("/api/sweep")
def sweep_start():
    d = request.json or {}
    reg = load_registry()

    def axis(ax):
        p = (ax or {}).get("param")
        vals = [v for v in ((ax or {}).get("values") or []) if str(v).strip()]
        if p == "model":
            vals = vals or list(reg)  # 空なら手持ち全モデル
            bad = [v for v in vals if v not in reg]
            if bad:
                return p, None, f"未登録モデル: {', '.join(bad)}"
        elif p in ("cfg", "steps"):
            if not vals:
                return p, None, f"{AXIS_JP[p]}の値が空（カンマ区切りで入れてね）"
        else:
            return p, None, "軸はmodel/cfg/stepsのどれか"
        return p, vals, None

    px, xv, e1 = axis(d.get("x"))
    py, yv, e2 = axis(d.get("y"))
    if e1 or e2:
        return jsonify({"error": e1 or e2}), 400
    if px == py:
        return jsonify({"error": "X軸とY軸は別の項目にしてね"}), 400
    if len(xv) * len(yv) > 30:
        return jsonify({"error": f"セル数{len(xv)*len(yv)}は多すぎ（30まで）"}), 400
    if not (d.get("prompt") or "").strip():
        return jsonify({"error": "プロンプトが空"}), 400

    seed = random.randrange(2**31)  # スイープ内は全セル同シード固定
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    jobname = "".join(c for c in (d.get("job") or "sweep") if c.isalnum() or c in "-_")[:20] or "sweep"
    sid = f"{stamp}_XYZ-{jobname}"
    sdir = OUTPUT / sid
    sdir.mkdir(parents=True, exist_ok=True)

    cells = []
    for yi, yval in enumerate(yv):
        for xi, xval in enumerate(xv):
            axes = {px: xval, py: yval}
            model = axes.get("model", d.get("model"))
            if model not in reg:
                return jsonify({"error": f"モデル未登録: {model}"}), 400
            cell_dir = sdir / f"x{xi}_y{yi}"
            cmd = [PYTHON, FORGE / "generate.py", "-m", model, "-p", d["prompt"],
                   "-W", int(d.get("width", 832)), "-H", int(d.get("height", 1216)),
                   "--seed", seed, "-c", 1, "--out", str(cell_dir), "--job", sid]
            if d.get("negative"):
                cmd += ["-n", d["negative"]]
            if "cfg" in axes:
                cmd += ["--cfg", float(axes["cfg"])]
            if "steps" in axes:
                cmd += ["--steps", int(float(axes["steps"]))]
            if d.get("facefix") is False:
                cmd += ["--no-face-fix"]
            label = f"スイープ {model}" + (f" cfg{axes['cfg']}" if axes.get("cfg") else "") \
                    + (f" steps{axes['steps']}" if axes.get("steps") else "")
            enqueue(cmd, f"sweep-x{xi}y{yi}", label)
            cells.append({"x": xi, "y": yi, "model": model,
                          "cfg": float(axes["cfg"]) if axes.get("cfg") else reg[model].get("cfg"),
                          "steps": int(float(axes["steps"])) if axes.get("steps") else reg[model].get("steps"),
                          "dir": f"x{xi}_y{yi}", "file": f"{model}_p00_s{seed}.png"})

    (sdir / "sweep.json").write_text(json.dumps(
        {"id": sid, "seed": seed, "prompt": d["prompt"],
         "x": {"param": px, "values": xv}, "y": {"param": py, "values": yv},
         "cells": cells}, ensure_ascii=False, indent=2))
    with lock:
        qlen = len(queue)
    return jsonify({"id": sid, "cells": len(cells), "queued": qlen})


@app.get("/sweep/<path:sid>")
def sweep_page(sid):
    """スイープの比較表ページ（焼き待ちセルはリロードで埋まる）"""
    sj = (OUTPUT / sid / "sweep.json").resolve()
    if not str(sj).startswith(str(OUTPUT.resolve())) or not sj.exists():
        abort(404)
    d = json.loads(sj.read_text())
    xp, yp = d["x"]["param"], d["y"]["param"]
    header = "".join(f"<th>{v}</th>" for v in d["x"]["values"])
    rows = ""
    for yi, yval in enumerate(d["y"]["values"]):
        tds = ""
        for xi, _ in enumerate(d["x"]["values"]):
            c = next(c for c in d["cells"] if c["x"] == xi and c["y"] == yi)
            rel = f"{sid}/{c['dir']}/{c['file']}"
            if (OUTPUT / sid / c["dir"] / c["file"]).exists():
                img = f'<a href="/img/{rel}" target="_blank"><img src="/img/{rel}" loading="lazy"></a>'
            else:
                img = '<div class="wait">焼き待ち…<br>（リロードで更新）</div>'
            tds += (f'<td>{img}<div class="meta">{c["model"]}｜CFG {c["cfg"]}｜steps {c["steps"]}<br>'
                    f'<button onclick=\'tune({json.dumps(c["model"])},{c["cfg"]},{c["steps"]})\'>これを当たりに登録</button></div></td>')
        rows += f"<tr><th>{yval}</th>{tds}</tr>"
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>XYZスイープ {sid}</title>
<link rel="icon" type="image/png" href="/favicon.png">
<style>body{{background:#0f1117;color:#e6e9f0;font-family:-apple-system,sans-serif;padding:20px}}
h1{{font-size:16px}}.sub{{color:#8b93a7;font-size:12px;margin:6px 0 14px;line-height:1.7}}
table{{border-collapse:collapse}}td,th{{border:1px solid #2a2f3d;padding:8px;vertical-align:top;font-size:12px}}
img{{width:220px;display:block;border-radius:6px}}
.meta{{font-size:11px;color:#8b93a7;margin-top:6px;max-width:220px;line-height:1.7}}
.wait{{width:220px;height:322px;display:flex;align-items:center;justify-content:center;text-align:center;color:#8b93a7;font-size:12px;background:#181b24;border-radius:6px}}
button{{margin-top:4px;padding:5px 10px;border:1px solid #2a2f3d;background:#181b24;color:#e6e9f0;border-radius:6px;cursor:pointer;font-size:11px}}
button:hover{{background:#7c9cff;color:#0b0d12}}</style></head><body>
<h1>XYZスイープ: {sid}</h1>
<div class="sub">シード {d["seed"]} 固定｜X軸={AXIS_JP[xp]} / Y軸={AXIS_JP[yp]}｜プロンプト: {d["prompt"][:140]}<br>
「これを当たりに登録」を押すと、そのモデルの既定CFG/ステップが更新されて次回から自動で入る（メモはモデルタブに表示）</div>
<table><tr><th>{AXIS_JP[yp]}＼{AXIS_JP[xp]}</th>{header}</tr>{rows}</table>
<script>
async function tune(model,cfg,steps){{
  const note=prompt('当たりメモ（models.jsonに残る・モデルタブに表示）','XYZスイープで当たり判定');
  if(note===null)return;
  const r=await(await fetch('/api/models/tune',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{model,cfg,steps,note}})}})).json();
  alert(r.ok?model+' の既定を CFG '+cfg+' / steps '+steps+' に更新！次からこの設定が自動で入る':'失敗: '+(r.error||''));
}}
</script></body></html>"""


@app.post("/api/models/tune")
def model_tune():
    """スイープで見つけた当たり設定をモデルの既定値＋メモとしてmodels.jsonに書き込む"""
    d = request.json or {}
    reg = json.loads(REGISTRY.read_text())
    name = d.get("model")
    if name not in reg or name.startswith("_"):
        return jsonify({"ok": False, "error": "モデル未登録"}), 404
    if d.get("cfg") is not None:
        reg[name]["cfg"] = float(d["cfg"])
    if d.get("steps") is not None:
        reg[name]["steps"] = int(d["steps"])
    if d.get("note"):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d")
        reg[name]["notes"] = f"{stamp} {d['note']}（当たり: CFG {reg[name].get('cfg')} / steps {reg[name].get('steps')}）"
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n")
    return jsonify({"ok": True})


# ---------- モデル管理 ----------

LORA_DIR = Path(os.path.expanduser("~/AI-Models/lora"))


@app.get("/api/loras")
def loras():
    if not LORA_DIR.exists():
        return jsonify([])
    try:
        meta = json.loads((FORGE / "lora_meta.json").read_text())
    except Exception:
        meta = {}
    return jsonify([{"name": p.stem, "sizeMb": round(p.stat().st_size / 1e6),
                     "thumb": meta.get(p.stem, {}).get("thumb"),
                     "page": meta.get(p.stem, {}).get("page"),
                     "desc": meta.get(p.stem, {}).get("desc"),
                     "base": meta.get(p.stem, {}).get("base")}
                    for p in sorted(LORA_DIR.glob("*.safetensors"))])


@app.post("/api/loras/delete")
def lora_delete():
    p = (LORA_DIR / f"{request.json.get('name', '')}.safetensors").resolve()
    if not str(p).startswith(str(LORA_DIR.resolve())) or not p.exists():
        abort(404)
    p.unlink()
    return jsonify({"ok": True})


@app.get("/api/models")
def models():
    out = []
    for name, cfg in load_registry().items():
        path = os.path.expanduser(cfg["path"])
        is_file = cfg["source"] in ("local", "hf-single")
        exists = Path(path).exists() if is_file else True
        size_gb = round(Path(path).stat().st_size / 1e9, 1) if is_file and exists else None
        out.append({"name": name, "label": cfg.get("label", name), "source": cfg["source"],
                    "path": cfg["path"], "sizeGb": size_gb, "exists": exists,
                    "vpred": cfg.get("vpred", False),
                    "popularity": cfg.get("popularity"),
                    "notes": cfg.get("notes"),
                    "thumb": cfg.get("thumb"), "page": cfg.get("page")})
    return jsonify(out)


@app.get("/api/civitai")
def civitai_search():
    q = request.args.get("query", "")
    sort = request.args.get("sort", "Most Downloaded")
    base = request.args.get("base", "")  # 例: Illustrious / SDXL 1.0 / NoobAI
    mtype = request.args.get("type", "Checkpoint")  # Checkpoint | LORA
    params = [("limit", "16"), ("types", mtype), ("nsfw", "false"), ("sort", sort)]
    if q:
        params.append(("query", q))
    if base:
        params.append(("baseModels", base))
    url = "https://civitai.com/api/v1/models?" + urllib.parse.urlencode(params)
    # CivitaiのCDNはブラウザ風UA以外を503で弾くことがある
    req = urllib.request.Request(
        url, headers={"User-Agent": "img-forge/1.0 (+https://github.com/kino176222/img-forge)"}
    )
    try:
        # 15秒返らないのはCivitai API側の実質障害とみなして打ち切る
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
    except Exception as e:
        return jsonify({"error": f"Civitai APIに届かない: {e}"}), 502
    items = []
    for m in data.get("items", []):
        v = (m.get("modelVersions") or [{}])[0]
        # 作例を最大4枚（画像のみ・きわどい作例は除外）。CDNのwidthパラメータで軽量化
        samples = []
        for im in v.get("images", []):
            if im.get("type", "image") != "image":
                continue
            lvl = im.get("nsfwLevel")
            if isinstance(lvl, int) and lvl > 1:
                continue
            if isinstance(im.get("nsfw"), str) and im["nsfw"] != "None":
                continue
            u = im.get("url", "")
            samples.append(u.replace("/width=", "/w=").replace("/original=true", "")
                           if "/width=" in u else u)
            if len(samples) >= 4:
                break
        stats = m.get("stats") or {}
        items.append({
            "id": m.get("id"), "name": m.get("name"),
            "base": v.get("baseModel"), "versionId": v.get("id"),
            "version": v.get("name"),
            "downloads": stats.get("downloadCount"),
            "thumbs": stats.get("thumbsUpCount"),
            "commercial": m.get("allowCommercialUse"),
            "samples": samples,
            "page": f"https://civitai.com/models/{m.get('id')}",
        })
    return jsonify(items)


@app.post("/api/models/add")
def model_add():
    d = request.json
    name = "".join(c for c in d["name"].lower().replace(" ", "-") if c.isalnum() or c == "-")[:40]
    if not name or not d.get("versionId"):
        abort(400)
    cmd = [PYTHON, FORGE / "fetch_model.py", "--name", name,
           "--civitai-version", d["versionId"], "--label", d.get("label", name)]
    if d.get("popularity"):
        cmd += ["--popularity", d["popularity"]]
    if d.get("kind") == "lora":
        cmd += ["--as-lora"]
        jid, qlen = enqueue(cmd, f"dl-lora-{name}", f"LoRA DL {name}（数十MB）")
    else:
        jid, qlen = enqueue(cmd, f"dl-{name}", f"モデルDL {name}（数GB・数分〜）")
    return jsonify({"id": jid, "queued": qlen, "name": name})


@app.post("/api/models/delete")
def model_delete():
    d = request.json
    reg = json.loads(REGISTRY.read_text())
    name = d.get("name")
    if name not in reg or name.startswith("_"):
        abort(404)
    cfg = reg.pop(name)
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n")
    freed = None
    if cfg["source"] in ("local", "hf-single"):
        p = Path(os.path.expanduser(cfg["path"]))
        if p.exists():
            freed = round(p.stat().st_size / 1e9, 1)
            p.unlink()
    return jsonify({"ok": True, "freedGb": freed})


@app.get("/favicon.png")
def favicon():
    return send_file(FORGE / "favicon.png")


@app.get("/")
def index():
    return PAGE


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>img-forge 画像鍛冶場</title>
<link rel="icon" type="image/png" href="/favicon.png">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0f1117;--card:#181b24;--line:#2a2f3d;--tx:#e6e9f0;--mut:#8b93a7;--acc:#7c9cff;--ok:#5ad8a6;--ng:#ff7d7d}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--tx);font-family:-apple-system,sans-serif;padding:20px}
h1{font-size:18px;margin-bottom:10px}h1 span{color:var(--mut);font-size:12px;font-weight:normal;margin-left:8px}
.tabs{display:flex;gap:8px;margin-bottom:14px}
.tabs button{padding:8px 18px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--mut);cursor:pointer;font-size:13px}
.tabs button.on{background:var(--acc);color:#0b0d12;font-weight:bold;border-color:var(--acc)}
.wrap{display:grid;grid-template-columns:380px 1fr;gap:16px}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
label{display:block;font-size:11px;color:var(--mut);margin:10px 0 4px}
select,textarea,input{width:100%;background:#0d0f15;color:var(--tx);border:1px solid var(--line);border-radius:8px;padding:8px;font-size:13px}
textarea{min-height:90px;resize:vertical}
.row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
button.go{width:100%;margin-top:14px;padding:12px;background:var(--acc);color:#0b0d12;font-weight:bold;border:0;border-radius:8px;font-size:14px;cursor:pointer}
#state{font-size:12px;color:var(--mut);white-space:pre-wrap;margin-top:12px;font-family:Menlo,monospace;max-height:160px;overflow:auto}
.gal{display:flex;flex-wrap:wrap;gap:8px}
.gal .it{position:relative;border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer;height:var(--rowh,220px)}
.gal .it img{width:100%;height:100%;object-fit:cover;display:block}
.gal .fill{flex-grow:1000000;height:0}
.gal .ops{position:absolute;bottom:8px;right:8px;display:flex;gap:6px;opacity:0;transition:.15s}
.gal .it:hover .ops{opacity:1}
.gal .ops button{width:34px;height:34px;border-radius:99px;border:0;cursor:pointer;font-size:14px;display:flex;align-items:center;justify-content:center;background:rgba(10,12,18,.72);backdrop-filter:blur(6px);color:#fff;padding:0}
.gal .ops button.keep:hover{background:var(--ok);color:#06331f}
.gal .ops button.trash:hover{background:var(--ng);color:#5c0f0f}
.keep{background:var(--ok);color:#083}.trash{background:var(--ng);color:#600}
.badge{position:absolute;top:6px;left:6px;background:var(--ok);color:#052;font-size:11px;font-weight:bold;padding:2px 8px;border-radius:99px}
.badge.sc{left:auto;right:6px;background:#ffd54a;color:#653}
.jsec{display:flex;align-items:baseline;gap:8px;font-size:13px;font-weight:bold;margin:18px 0 8px;padding-bottom:4px;border-bottom:1px solid var(--line)}
.jsec span{color:var(--mut);font-size:11px;font-weight:normal}
.jsec:first-child{margin-top:0}
button.more{display:block;margin:16px auto 4px;padding:10px 24px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--tx);cursor:pointer;font-size:13px}
.facecrop{object-fit:cover;object-position:50% 12%}
.refrow{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--tx);line-height:1.4}
.refrow img{width:44px;height:44px;object-fit:cover;object-position:50% 12%;border-radius:6px;flex-shrink:0}
.refrow span b,.refcol span b{display:inline;font-size:10px;color:var(--mut);font-weight:normal;margin-right:4px}
.refcol{display:flex;flex-direction:column;gap:4px;font-size:11px;color:var(--tx)}
.refcol img{width:100%;max-height:260px;object-fit:contain;background:#0d0f15;border-radius:8px}
.chip{padding:6px 12px;border:1px solid var(--line);border-radius:99px;background:var(--card);color:var(--tx);cursor:pointer;font-size:12px}
.chip.on{background:var(--acc);color:#0b0d12;font-weight:bold;border-color:var(--acc)}
.seg{display:inline-flex;background:#0d0f15;border:1px solid var(--line);border-radius:10px;padding:3px;gap:3px}
.seg button{padding:7px 12px;border:0;border-radius:8px;background:none;color:var(--mut);cursor:pointer;font-size:12px;white-space:nowrap}
.seg button.on{background:#2a2f3d;color:var(--tx);font-weight:bold}
.ric{display:inline-block;border:1.5px solid currentColor;border-radius:2px;margin-right:6px;vertical-align:-2px}
.r23{width:8px;height:12px}.r32{width:12px;height:8px}.r169{width:14px;height:8px}.r11{width:10px;height:10px}
.tgl{position:relative;width:40px;height:22px;appearance:none;-webkit-appearance:none;background:#2a2f3d;border:0;border-radius:99px;cursor:pointer;transition:.15s;flex-shrink:0}
.tgl:checked{background:var(--acc)}
.tgl::after{content:'';position:absolute;top:2px;left:2px;width:18px;height:18px;border-radius:50%;background:#fff;transition:.15s}
.tgl:checked::after{transform:translateX(18px)}
#mbanner{position:relative;border-radius:10px;overflow:hidden;margin-top:6px;cursor:zoom-in;display:none;background:#0d0f15}
#mbanner img{width:100%;max-height:380px;object-fit:contain;display:block}
#mbanner::after{content:'';position:absolute;inset:0;background:linear-gradient(180deg,transparent 55%,rgba(5,7,12,.85));pointer-events:none}
#mbanner .nm{position:absolute;left:12px;bottom:9px;right:12px;font-weight:bold;font-size:14px;z-index:1;text-shadow:0 1px 6px rgba(0,0,0,.9)}
details.adv{margin-top:12px;border:1px solid var(--line);border-radius:10px;background:#12141c}
details.adv summary{padding:10px 12px;cursor:pointer;font-size:12px;color:var(--mut)}
details.adv .inner{padding:0 12px 12px}
.ic{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;vertical-align:-2px}
.icf{fill:currentColor}
.revbar{position:sticky;top:0;z-index:5;box-shadow:0 4px 16px rgba(0,0,0,.45)}
.mrow{display:flex;align-items:center;gap:12px;padding:10px;border:1px solid var(--line);border-radius:10px;margin-bottom:8px}
.mrow .nm{font-weight:bold}.mrow .sub{font-size:11px;color:var(--mut)}
.mrow button{margin-left:auto;background:none;border:1px solid var(--line);color:var(--ng);border-radius:6px;padding:5px 10px;cursor:pointer;font-size:12px}
.cgrid{display:grid;grid-template-columns:1fr;gap:12px;margin-top:12px}
.cit{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#12141c}
.strip{display:flex;gap:2px;overflow-x:auto}
.strip img{height:220px;flex:1;min-width:0;object-fit:cover;display:block;cursor:zoom-in}
.cit .b{padding:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.cit .n{font-size:13px;font-weight:bold;line-height:1.3}.cit .s{font-size:11px;color:var(--mut)}
.cit button{margin-left:auto;padding:8px 18px;border:0;border-radius:6px;background:var(--acc);color:#0b0d12;font-weight:bold;cursor:pointer;font-size:12px}
.hint{font-size:11px;color:var(--mut);margin-top:8px;line-height:1.6}
#lb{display:none;position:fixed;inset:0;background:rgba(5,7,12,.92);z-index:99}
#lb.on{display:flex}
#lbmain{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;padding:14px;min-width:0}
#lbimg{max-width:100%;max-height:calc(100vh - 92px);object-fit:contain;border-radius:8px}
#lbbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:center}
#lbbar button{padding:10px 16px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--tx);cursor:pointer;font-size:13px;font-weight:bold}
#lbbar button.keep,#lbbar button.trash{border:0}
#lbcnt{font-size:12px;color:var(--mut);min-width:52px;text-align:center}
#lbside{width:300px;flex-shrink:0;background:var(--card);border-left:1px solid var(--line);padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:10px}
#lbside.off{display:none}
#lbmeta{white-space:pre-wrap;word-break:break-word;font-family:Menlo,monospace;font-size:11px;line-height:1.7;color:var(--tx);user-select:text}
</style></head><body>
<h1>img-forge<span>ローカル画像鍛冶場 — 課金なし・無制限</span></h1>
<div class="tabs">
  <button data-t="gen" class="on">生成</button>
  <button data-t="rev">レビュー</button>
  <button data-t="mdl">モデル</button>
</div>

<div id="tab-gen" class="wrap">
<div class="card">
  <label title="当たりの組み合わせを名前付きで保存するプリセット。【保存されるもの】モデル／LoRA（強度ツマミ込み）／画風タグ（プロンプト末尾に自動合体）／追加ネガティブ。【保存されないもの】本文プロンプト（構図・シーン・人物＝毎回書く）と、ステップ・CFG（モデル推奨値が自動）。【使い方】スタイルを選ぶ→本文に描きたい内容を書く→焼く">スタイル（当たりプリセット）</label>
  <div style="display:flex;gap:6px">
    <select id="stylesel" style="flex:1" title="スタイルを選ぶと、モデル・LoRA・画風タグ・追加ネガがその場で切り替わる。本文プロンプトはそのまま残るので、同じシーンを別の絵柄で焼き比べられる"><option value="">（スタイル未選択）</option></select>
    <button class="chip" id="stylesave" title="今のモデル・LoRA・画風タグ・追加ネガを名前を付けて保存">保存</button>
    <button class="chip" id="styledel" title="選択中のスタイルを削除">削除</button>
  </div>
  <label>モデル</label><select id="model"></select>
  <div id="mbanner" title="このモデルの公式作例。クリックで拡大"><img id="mprev" alt=""><div class="nm" id="mname"></div></div>
  <label title="描きたい内容だけ英語タグで書く（例: 1girl, kimono, night sky）。「masterpiece」等の品質タグと定番ネガティブはモデルごとに自動で付く">プロンプト（本文だけ。品質タグは自動）</label><textarea id="prompt">1girl, solo, cafe barista, green apron, holding coffee cup, gentle smile, backlighting, sunlight through window, light particles, cafe interior, upper body</textarea>
  <label title="絵柄・塗りを固定するタグ（プロンプト末尾に自動で付く）。Illustrious系は画風タグが無いと絵柄が毎回暴れるので、当たりの画風はここで固定する">画風タグ（絵柄の固定・任意）</label><input id="styletags" placeholder="例: watercolor (medium), soft lighting, pastel colors">
  <label title="出したくない要素を書く。低品質・手の崩れ等の定番ネガティブは自動で付くので、それ以外の追加分だけ">追加ネガティブ（任意）</label><input id="negative">
  <label title="SDXLが得意な解像度（約100万画素）。これ以外のサイズは破綻しやすい。大きくしたい時はhiresをON">サイズ</label>
  <div class="seg" style="display:flex" id="sizeseg">
    <button data-s="832x1216" class="on"><span class="ric r23"></span>2:3 縦</button>
    <button data-s="1216x832"><span class="ric r32"></span>3:2 横</button>
    <button data-s="1344x768"><span class="ric r169"></span>16:9</button>
    <button data-s="1024x1024"><span class="ric r11"></span>1:1</button>
  </div>
  <label title="同じ設定でシード違いを何枚焼くか。多めに焼いて選ぶのが基本。8枚以上でAI自動採点が付く">枚数</label>
  <div style="display:flex;gap:8px;align-items:stretch">
    <div class="seg" style="flex:1;display:flex" id="cntseg">
      <button data-c="1" style="flex:1">1</button>
      <button data-c="2" style="flex:1">2</button>
      <button data-c="3" style="flex:1">3</button>
      <button data-c="4" style="flex:1" class="on">4</button>
      <button data-c="10" style="flex:1">10</button>
    </div>
    <input id="count" type="number" value="4" min="1" max="500" style="width:70px" title="自由入力">
  </div>
  <label title="一度焼いた絵をイラスト用AIで約2倍に拡大→もう一度描き直して仕上げる清書モード。見せる用はON推奨・時間は約2倍" style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:14px;font-size:13px;color:var(--tx)">
    高画質2段階（hires fix・所要約2倍）<input type="checkbox" id="hires" class="tgl">
  </label>
  <label title="顔を自動検出して、その部分だけ高解像度で描き直す。引きの構図で顔が小さい絵ほど効く。風景など顔なしの絵はOFFで高速化" style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:10px;font-size:13px;color:var(--tx)">
    顔レタッチ（検出→描き直し・推奨ON）<input type="checkbox" id="facefix" class="tgl" checked>
  </label>
  <div id="lorabox" style="margin-top:8px"></div>
  <details class="adv">
    <summary title="普段は触らなくてOK。モデルごとの推奨値が既定で入っている">高級設定（ステップ・CFG・シード・ジョブ名）</summary>
    <div class="inner">
      <div class="row">
        <div title="ノイズから絵を練り上げる回数。多いほど丁寧だが遅い。26前後が定番・8〜12はラフ検討用"><label>ステップ</label><input id="steps" type="number" value="26"></div>
        <div title="プロンプトへの従順さ。低い=自由に描く／高い=指示に忠実だが絵が硬くなる。5〜6が定番（6超は非推奨）"><label>CFG</label><input id="cfg" type="number" value="6.0" step="0.5"></div>
        <div title="乱数の種。同じシード＋同じ設定＝同じ絵が再現できる。-1なら毎回ランダム。当たり絵の微修正に使う"><label>シード(-1=乱数)</label><input id="seed" type="number" value="-1"></div>
      </div>
      <label title="出力フォルダの名前。レビュー画面のバッチ選択にこの名前で並ぶ">ジョブ名</label><input id="job" value="dash" title="出力フォルダの名前。レビュー画面のバッチ選択にこの名前で並ぶ">
    </div>
  </details>
  <details class="adv">
    <summary title="設定を総当たりで焼いて比較表にする実験装置。新モデルの実力測定と、当たり設定（CFG・ステップ）探しに使う。シードは自動で1個に固定される">XYZスイープ（新モデルの品定め・設定総当たり）</summary>
    <div class="inner">
      <div class="row" style="grid-template-columns:110px 1fr">
        <div title="比較表の横方向に振る項目。CFG=プロンプトへの従順さ／ステップ=描き込み回数／モデル=絵描きAI本体"><label>X軸（横）</label><select id="swx"><option value="cfg">CFG</option><option value="steps">ステップ</option><option value="model">モデル</option></select></div>
        <div title="試したい値をカンマ区切りで並べる。例: CFGなら「4, 5.5, 7」／ステップなら「12, 20, 28」／モデルなら登録名（空欄=手持ち全モデル）"><label>X軸の値（カンマ区切り。モデル軸で空=手持ち全部）</label><input id="swxv" value="4, 5.5, 7"></div>
      </div>
      <div class="row" style="grid-template-columns:110px 1fr">
        <div title="比較表の縦方向に振る項目。X軸と別のものを選ぶ。定番は「Y=モデル×X=CFG」（新モデルの品定め）"><label>Y軸（縦）</label><select id="swy"><option value="model">モデル</option><option value="cfg">CFG</option><option value="steps">ステップ</option></select></div>
        <div title="試したい値をカンマ区切りで並べる（空欄=手持ち全モデル）。セル数=X×Yで上限30"><label>Y軸の値（同上）</label><input id="swyv" placeholder="空欄=手持ち全モデル"></div>
      </div>
      <button class="go" id="swgo" title="本文・画風タグ・サイズは上の設定を使う。1セル1枚×セル数ぶんキューに積む">スイープを始める</button>
      <div class="hint">できあがりは比較表ページで見る（焼けたセルから順に埋まる・リロードで更新）。いい設定が見つかったら表の「これを当たりに登録」でモデルの既定値＋メモに保存</div>
      <div id="swlast" class="hint"></div>
    </div>
  </details>
  <button class="go" id="go">焼く</button>
  <div id="state">待機中</div>
</div>
<div class="card">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px">
    <span style="font-size:12px;color:var(--mut)">日付で見る:</span>
    <div id="genchips" style="display:flex;gap:6px;flex-wrap:wrap"></div>
    <span class="hint" style="margin:0">クリックでレビュー画面が開く（その日の全部・採用絞り込みもそこで）</span>
  </div>
  <div id="gal-gen"></div>
</div>
</div>

<div id="tab-rev" class="wrap" style="display:none;grid-template-columns:1fr">
<div class="card revbar">
  <div style="display:flex;gap:18px;align-items:flex-end;flex-wrap:wrap">
    <b style="font-size:14px;padding-bottom:7px">生成レビュー</b>
    <div><label>日付（クリックでその日ぜんぶ）</label><div id="daychips" style="display:flex;gap:6px;flex-wrap:wrap"></div></div>
    <div><label>バッチ（焼いた単位）</label><select id="batch" style="width:auto"><option value="">全バッチ</option></select></div>
    <div><label>表示</label><div class="seg" id="viewseg">
      <button data-v="all" class="on">すべて</button>
      <button data-v="unjudged">未判定</button>
      <button data-v="picks">採用だけ</button>
    </div></div>
    <div><label>並び順</label><div class="seg" id="sortseg">
      <button data-v="new" class="on">新しい順</button>
      <button data-v="score">点数順</button>
      <button data-v="picked">採用を先頭</button>
    </div></div>
    <div><label>サムネの大きさ</label><input type="range" id="rowh" min="120" max="400" step="20" style="width:120px"></div>
  </div>
  <span class="hint">画像クリックで拡大レビュー: ←→=移動｜<b>P=採用（もう一度で取り消し）</b>｜<b>X=ボツ</b>｜U=保留（判定で自動的に次へ。ボツは_trashへ退避・即削除しない）</span>
</div>
<div class="card"><div id="gal-rev"></div></div>
</div>

<div id="tab-mdl" class="wrap" style="grid-template-columns:420px 1fr;display:none">
<div class="card">
  <b style="font-size:14px">手持ちモデル</b>
  <div id="mlist" style="margin-top:10px"></div>
  <div class="hint">削除するとファイルも消える（登録だけのモデルは台帳から外すだけ）。追加は右のCivitaiカタログから。ログイン必須モデルは環境変数CIVITAI_TOKENが必要。</div>
</div>
<div class="card">
  <b style="font-size:14px">Civitaiショーケース（Checkpoint）</b>
  <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
    <input id="cq" placeholder="検索語（空なら人気順ショーケース）" style="flex:1;min-width:180px">
    <select id="ctype" style="width:auto">
      <option value="Checkpoint">モデル</option>
      <option value="LORA">LoRA（味変パーツ）</option>
    </select>
    <select id="cbase" style="width:auto">
      <option value="Illustrious">Illustrious系（FeMix/Nova用）</option>
      <option value="NoobAI">NoobAI系（noobai用）</option>
      <option value="SDXL 1.0">SDXL系（大元・汎用）</option>
      <option value="Pony">Pony系（別派閥・相性注意）</option>
      <option value="">全ベース（他系統も混ざる）</option>
    </select>
    <select id="csort" style="width:auto">
      <option value="Most Downloaded">人気順</option>
      <option value="Newest">新着順</option>
      <option value="Highest Rated">高評価順</option>
    </select>
    <button class="go" style="width:100px;margin:0" id="csearch">表示</button>
  </div>
  <div class="hint" style="line-height:1.7">系統＝土台モデルの家系図。<b>モデルもLoRAも、系統が合っているほどよく効く</b>。<br>
  <b>SDXL</b>（大元。汎用でLoRA資産が最多）→ <b>Illustrious</b>（SDXLをアニメ特化に鍛え直した現主流。FeMix/Novaはここ）→ <b>NoobAI</b>（Illustriousをさらに大量学習した新鋭）。<br>
  SDXL用LoRAはIllustrious/NoobAIでも大体効く（子孫だから）。<b>Pony</b>はSDXLから分かれた別派閥＝構造は同じでも効きは相性次第。<br>
  <b>注意</b>—相性の悪い系統: <b>SD 1.5</b>（旧世代。読めても効きは博打＝入れたら同シードA/Bで実測確認）／<b>Flux・SD3</b>（別アーキテクチャ・不可）。DL前にページの「Base Model」欄を確認</div>
  <div class="cgrid" id="cgrid"></div>
  <div class="hint">各カードの帯＝そのモデルの作例4枚（クリックで拡大）。「追加」でDLキューに入る（1体6GB級）。商用利用の可否は各ページで最終確認。</div>
</div>
</div>

<div id="lb">
  <div id="lbmain">
    <img id="lbimg" alt="">
    <div id="lbbar">
      <button id="lbprev"><svg class="ic" viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"/></svg></button>
      <span id="lbcnt"></span>
      <button id="lbnext"><svg class="ic" viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"/></svg></button>
      <button class="keep" id="lbkeep">採用 (P)</button>
      <button id="lbskip">保留 (U)</button>
      <button class="trash" id="lbtrash">ボツ (X)</button>
      <button id="lbclose">閉じる (Esc)</button>
    </div>
  </div>
  <div id="lbside">
    <div id="lbstat"></div>
    <div id="lbrefs" style="display:flex;flex-direction:column;gap:6px"></div>
    <div id="lbmeta"></div>
    <button class="go" id="lbapply" style="display:none;margin-top:0">この設定を生成フォームへ</button>
    <div id="lbvarrow" style="display:none;gap:6px;align-items:center">
      <select id="lbvarstr" style="width:auto" title="兄弟同士の違いの強さ（img2imgの描き直し度）。弱いほど元の絵にそっくり">
        <option value="0.35">差: 弱 0.35</option>
        <option value="0.45" selected>差: 中 0.45</option>
        <option value="0.6">差: 強 0.6</option>
      </select>
      <button class="go" id="lbvar" style="margin:0;flex:1" title="惜しい絵の救済（微調整ガチャ）。この画像を出発点に、構図・色を保ったまま細部だけ描き直した「ほぼ同じだけど少し違う」兄弟を量産する（img2img方式）。速度優先でhires無し＝当たりが出たら「この設定を生成フォームへ」→hiresで清書">兄弟を8枚焼く</button>
    </div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
// ---- サムネ行高スライダー ----
const rowh0=localStorage.getItem('rowh')||'220';
document.documentElement.style.setProperty('--rowh',rowh0+'px');
// ---- ライトボックス（単画像モード＋レビューコックピット） ----
let LB={key:null,idx:0,meta:null};
const LBBTNS=['lbprev','lbnext','lbkeep','lbskip','lbtrash'];
function zoom(u){
  LB={key:null,idx:0,meta:null};
  $('lbimg').src=u;$('lbcnt').textContent='';
  LBBTNS.forEach(id=>$(id).style.display='none');
  $('lbside').classList.add('off');
  $('lb').classList.add('on');
}
function openLB(key,idx){
  LB={key,idx,meta:null};
  LBBTNS.forEach(id=>$(id).style.display='');
  $('lbside').classList.remove('off');
  $('lb').classList.add('on');
  showLB();
}
async function showLB(){
  const g=GAL[LB.key];
  if(!g.list.length){closeLB();return;}
  if(LB.idx>=g.list.length)LB.idx=g.list.length-1;
  const i=g.list[LB.idx];
  $('lbimg').src='/img/'+i.path;
  $('lbcnt').textContent=`${LB.idx+1}/${g.list.length}`;
  $('lbstat').innerHTML=(i.picked?'<span class="badge" style="position:static">採用済み</span> ':'')
    +(i.score!=null?`<span class="badge sc" style="position:static">${Math.round(i.score*100)}点</span>`:'');
  $('lbkeep').textContent=i.picked?'採用を取り消し (P)':'採用 (P)';
  $('lbmeta').textContent='メタデータ取得中…';$('lbrefs').innerHTML='';$('lbapply').style.display='none';$('lbvarrow').style.display='none';
  const path=i.path;
  const m=await (await fetch('/api/meta?path='+encodeURIComponent(path))).json();
  if(!LB.key||GAL[LB.key].list[LB.idx]?.path!==path)return; // もう次の画像に進んでいる
  if(!m.found){LB.meta=null;$('lbmeta').textContent='生成パラメータなし（外部画像など）';return}
  LB.meta=m;
  // 使ったモデル（大きめ作例）・LoRAを表示
  let refs='';
  const reg=REG[m.model];
  if(m.model)refs+=`<div class="refcol">${reg&&reg.thumb?`<img src="${reg.thumb}">`:''}<span><b>モデル</b>${reg?reg.label:m.model}</span></div>`;
  (m.lora||[]).forEach(s=>{
    const n=s.split(':')[0];const l=LORAS.find(x=>x.name===n);
    refs+=`<div class="refrow">${l&&l.thumb?`<img src="${l.thumb}">`:''}<span><b>LoRA</b>${s}</span></div>`;
  });
  $('lbrefs').innerHTML=refs;
  $('lbmeta').textContent=`モデル: ${m.model??'?'}
シード: ${m.seed??'?'}｜steps ${m.steps??'?'}｜CFG ${m.cfg??'?'}
サイズ: ${m.size??'?'}${m.hires?'｜hires':''}${m.lora&&m.lora.length?'｜LoRA '+m.lora.join(' '):''}

▼プロンプト本文
${m.body||m.prompt||''}${m.extraNegative?'\\n\\n▼追加ネガティブ\\n'+m.extraNegative:''}`;
  $('lbapply').style.display='';
  if(m.seed&&REG[m.model])$('lbvarrow').style.display='flex';
}
function closeLB(){$('lb').classList.remove('on');if(LB.key)renderGal(LB.key);}
function lbAdvance(){if(LB.idx<GAL[LB.key].list.length-1){LB.idx++;showLB();}else closeLB();}
async function lbJudge(action){
  const g=GAL[LB.key];const i=g.list[LB.idx];
  if(action==='keep'&&i.picked)action='unkeep'; // Pもう一度=取り消し
  await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:i.path,action})});
  if(action==='keep'){i.picked=true;lbAdvance();}
  else if(action==='unkeep'){
    i.picked=false;
    if(LB.key==='rev'&&VIEW==='picks'){g.list.splice(LB.idx,1);g.total--;g.fetched--;g.list.length?showLB():closeLB();}
    else showLB();
  }
  else{g.list.splice(LB.idx,1);g.total--;g.fetched--;g.list.length?showLB():closeLB();}
}
$('lbvar').onclick=async()=>{
  const g=GAL[LB.key];const i=g&&g.list[LB.idx];if(!i)return;
  const r=await (await fetch('/api/variation',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:i.path,strength:+$('lbvarstr').value,count:8})})).json();
  alert(r.error?('失敗: '+r.error):`兄弟${r.count}枚をキューに入れた！\\n焼き上がりはレビューの新バッチ（var-〜）に並ぶよ（1枚目は元の再現）`);
};
$('lbprev').onclick=()=>{if(LB.idx>0){LB.idx--;showLB();}};
$('lbnext').onclick=lbAdvance;
$('lbskip').onclick=lbAdvance;
$('lbkeep').onclick=()=>lbJudge('keep');
$('lbtrash').onclick=()=>lbJudge('trash');
$('lbclose').onclick=closeLB;
$('lb').onclick=e=>{if(e.target.id==='lb'||e.target.id==='lbmain')closeLB();};
document.addEventListener('keydown',e=>{
  if(!$('lb').classList.contains('on'))return;
  if(e.key==='Escape'){closeLB();return;}
  if(!LB.key)return;
  const k=e.key.toLowerCase();
  if(e.key==='ArrowRight'){e.preventDefault();lbAdvance();}
  else if(e.key==='ArrowLeft'){e.preventDefault();if(LB.idx>0){LB.idx--;showLB();}}
  else if(k==='p')lbJudge('keep');
  else if(k==='x')lbJudge('trash');
  else if(k==='u')lbAdvance();
});
const tabs={gen:$('tab-gen'),rev:$('tab-rev'),mdl:$('tab-mdl')};
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  Object.entries(tabs).forEach(([k,el])=>el.style.display=k===b.dataset.t?'grid':'none');
  if(b.dataset.t==='mdl'){loadModels();if(!civitaiLoaded){civitaiLoaded=true;civitai();}}
  if(b.dataset.t==='rev'){loadJobs();loadReview();}
});
let REG={};
function updatePrev(){
  const r=REG[$('model').value];
  // モデルごとの当たり設定を反映する（models.jsonのsteps/cfgが正本）。
  // ここが無いと「当たりに登録→次回から自動適用」がGUIで成立しない
  if(r){
    if(r.steps) $('steps').value=r.steps;
    if(r.cfg) $('cfg').value=r.cfg;
  }
  const el=$('mbanner');
  if(r&&r.thumb){
    $('mprev').src=r.thumb;
    $('mname').textContent=r.label||$('model').value;
    el.style.display='block';
    el.onclick=()=>zoom(r.thumb);
  }else el.style.display='none';
}
// ---- セグメントボタン（サイズ・枚数・表示・並び順） ----
let SIZE='832x1216',VIEW='all',SORT='new';
function segWire(id,fn){
  $(id).querySelectorAll('button').forEach(b=>b.onclick=()=>{
    $(id).querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));
    fn(b.dataset);
  });
}
segWire('sizeseg',d=>SIZE=d.s);
segWire('cntseg',d=>$('count').value=d.c);
segWire('viewseg',d=>{VIEW=d.v;loadReview();});
segWire('sortseg',d=>{SORT=d.v;loadReview();});
$('count').oninput=()=>$('cntseg').querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.c===$('count').value));
fetch('/api/registry').then(r=>r.json()).then(reg=>{
  REG=reg;
  $('model').innerHTML=Object.entries(reg).map(([k,v])=>`<option value="${k}">${v.label}</option>`).join('');
  updatePrev();
});
$('model').onchange=updatePrev;
function baseTag(b){
  if(!b)return'';
  const bad=/1\\.5|flux|sd ?3/i.test(b);
  return ` <span style="font-size:10px;padding:1px 6px;border-radius:99px;background:${bad?'#3a1d1d':'#1d2a3a'};color:${bad?'var(--ng)':'var(--acc)'}">${b}</span>`;
}
// ---- スタイル（当たりプリセット） ----
let STYLES={};
async function loadStyles(){
  STYLES=await (await fetch('/api/styles')).json();
  const cur=$('stylesel').value;
  $('stylesel').innerHTML='<option value="">（スタイル未選択）</option>'
    +Object.keys(STYLES).map(n=>`<option value="${n}">${n}</option>`).join('');
  if(STYLES[cur]!==undefined)$('stylesel').value=cur;
}
loadStyles();
$('stylesel').onchange=()=>{
  const s=STYLES[$('stylesel').value];if(!s)return;
  if(s.model&&REG[s.model]){$('model').value=s.model;updatePrev();}
  $('styletags').value=s.styleTags||'';
  $('negative').value=s.negative||'';
  document.querySelectorAll('.lora-ck').forEach(ck=>{
    const spec=(s.loras||[]).find(x=>x.split(':')[0]===ck.dataset.name);
    ck.checked=!!spec;
    const w=spec&&spec.split(':')[1];
    if(w)document.querySelector(`.lora-w[data-name="${ck.dataset.name}"]`).value=w;
  });
};
$('stylesave').onclick=async()=>{
  const name=prompt('スタイル名（例: けふのうた夜風）',$('stylesel').value||'');
  if(!name)return;
  const loras=[...document.querySelectorAll('.lora-ck:checked')].map(ck=>
    `${ck.dataset.name}:${document.querySelector(`.lora-w[data-name="${ck.dataset.name}"]`).value}`);
  await fetch('/api/styles/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name,model:$('model').value,loras,styleTags:$('styletags').value,negative:$('negative').value})});
  await loadStyles();$('stylesel').value=name;
};
$('styledel').onclick=async()=>{
  const n=$('stylesel').value;if(!n)return;
  if(!confirm(`スタイル「${n}」を削除する？（画像は消えない・プリセットだけ）`))return;
  await fetch('/api/styles/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:n})});
  await loadStyles();
};
let LORAS=[];
async function loadLoras(){
  const ls=await (await fetch('/api/loras')).json();
  LORAS=ls;
  $('lorabox').innerHTML=ls.length?`<div style="font-size:11px;color:var(--mut);margin-bottom:4px;line-height:1.6">
      <b style="color:var(--tx)">LoRA＝絵柄の味変パーツ</b>（モデルに後がけするフィルター）。チェックした方向に絵柄が寄る。<br>
      数字=効き具合（0.8が基本・0.5で控えめ・1.0で強め）。複数掛けもOK。画像クリックで公式作例を拡大。<br>
      青バッジ=対応する系統（土台モデルの家系。詳しくはモデルタブ参照）。系統が合うほどよく効く</div>`+
    ls.map(l=>`<label style="display:flex;align-items:center;gap:8px;font-size:12px;margin-top:6px">
      <input type="checkbox" class="lora-ck" data-name="${l.name}" style="width:auto">
      ${l.thumb?`<img class="facecrop" src="${l.thumb}" style="width:40px;height:40px;object-fit:cover;border-radius:6px;cursor:zoom-in;flex-shrink:0" onclick="event.preventDefault();zoom('${l.thumb}')">`:''}
      <span style="flex:1;min-width:0;line-height:1.4">${l.name} <span style="color:var(--mut)">(${l.sizeMb}MB)</span>${baseTag(l.base)}
      ${l.desc?`<br><span style="font-size:11px;color:var(--mut)">${l.desc}</span>`:''}</span>
      <input type="number" class="lora-w" data-name="${l.name}" value="0.8" step="0.1" min="-3" max="3" style="width:60px;flex-shrink:0">
    </label>`).join(''):'';
}
loadLoras();
$('swgo').onclick=async()=>{
  const px=$('swx').value,py=$('swy').value;
  if(px===py){alert('X軸とY軸は別の項目にしてね');return;}
  const parse=v=>v.split(',').map(s=>s.trim()).filter(Boolean);
  const [w,h]=SIZE.split('x').map(Number);
  const st=$('styletags').value.trim();
  const body={prompt:$('prompt').value.trim()+(st?', '+st:''),negative:$('negative').value,
    x:{param:px,values:parse($('swxv').value)},y:{param:py,values:parse($('swyv').value)},
    model:$('model').value,width:w,height:h,facefix:$('facefix').checked,job:$('job').value};
  const j=await (await fetch('/api/sweep',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  if(j.error){alert(j.error);return;}
  $('swlast').innerHTML=`スイープ ${j.id} 投入（${j.cells}セル・待ち行列${j.queued}件）→ <a href="/sweep/${j.id}" target="_blank" style="color:var(--acc)">比較表を開く</a>`;
};
$('go').onclick=async()=>{
  const [w,h]=SIZE.split('x').map(Number);
  const loras=[...document.querySelectorAll('.lora-ck:checked')].map(ck=>{
    const wv=document.querySelector(`.lora-w[data-name="${ck.dataset.name}"]`).value;
    return `${ck.dataset.name}:${wv}`;
  });
  const st=$('styletags').value.trim();
  const body={model:$('model').value,prompt:$('prompt').value.trim()+(st?', '+st:''),negative:$('negative').value,
    width:w,height:h,count:+$('count').value,seed:+$('seed').value,
    steps:+$('steps').value,cfg:+$('cfg').value,job:$('job').value,
    hires:$('hires').checked,facefix:$('facefix').checked,loras};
  const j=await (await fetch('/api/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  $('state').textContent=`投入: ${j.id}（待ち行列 ${j.queued}件）`;
};
// ---- ギャラリー（justified＋バッチ見出し＋もっと見る） ----
const GAL={gen:{el:'gal-gen',list:[],total:0,fetched:0,limit:40},
           rev:{el:'gal-rev',list:[],total:0,fetched:0,limit:200}};
function fmtJob(j){
  const m=j.match(/^(\\d{4})(\\d{2})(\\d{2})_(\\d{2})(\\d{2})_?(.*)$/);
  return m?`${+m[2]}/${+m[3]} ${m[4]}:${m[5]}${m[6]?'　'+m[6]:''}`:j;
}
const SVG_STAR='<svg class="ic" viewBox="0 0 24 24"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>';
const SVG_STAR_F=SVG_STAR.replace('"ic"','"ic icf"');
const SVG_TRASH='<svg class="ic" viewBox="0 0 24 24"><path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>';
function itemHtml(i,key,idx){
  const ar=(i.w&&i.h)?i.w/i.h:0.7;
  const sc=i.score!=null?`<span class="badge sc" title="AIの自動採点（構図・画質の美的スコア）。参考程度、最後は目で選ぶ">${Math.round(i.score*100)}点</span>`:'';
  return `<div class="it" data-k="${key}" data-i="${idx}" style="flex-grow:${(ar*100).toFixed(1)};flex-basis:calc(${ar.toFixed(3)}*var(--rowh,220px))">
    ${i.picked?'<span class="badge">採用</span>':''}${sc}
    <img loading="lazy" src="/img/${i.path}">
    <div class="ops">
      <button class="keep" data-act="${i.picked?'unkeep':'keep'}" title="${i.picked?'採用を取り消し（P）':'採用（P）'}">${i.picked?SVG_STAR_F:SVG_STAR}</button>
      <button class="trash" data-act="trash" title="ボツ（X）">${SVG_TRASH}</button>
    </div></div>`;
}
function renderGal(key){
  const g=GAL[key];
  const secs=[];let cur=null;
  g.list.forEach((i,idx)=>{
    if(!cur||cur.job!==i.job){cur={job:i.job,items:[]};secs.push(cur);}
    cur.items.push(itemHtml(i,key,idx));
  });
  let html=secs.map(s=>`<div class="jsec">${fmtJob(s.job)}<span>${s.items.length}枚</span></div>
    <div class="gal">${s.items.join('')}<div class="fill"></div></div>`).join('');
  if(g.total>g.fetched)
    html+=`<button class="more" data-k="${key}">もっと見る（${g.fetched}/${g.total}枚 読み込み済み）</button>`;
  $(g.el).innerHTML=html||'<span class="hint">画像なし</span>';
}
document.addEventListener('click',e=>{
  const act=e.target.closest('button[data-act]');
  if(act){
    const it=act.closest('.it');
    rv(GAL[it.dataset.k].list[+it.dataset.i].path,act.dataset.act,it.dataset.k);
    e.stopPropagation();return;
  }
  const it=e.target.closest('.it[data-k]');
  if(it){openLB(it.dataset.k,+it.dataset.i);return;}
  const more=e.target.closest('button.more');
  if(more){
    GAL[more.dataset.k].limit+=200;
    more.dataset.k==='rev'?loadReview():refreshGen(true);
  }
});
async function rv(path,action,key){
  await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path,action})});
  const g=GAL[key];
  const idx=g.list.findIndex(i=>i.path===path);
  if(idx>=0){
    if(action==='keep')g.list[idx].picked=true;
    else if(action==='unkeep'){
      g.list[idx].picked=false;
      if(key==='rev'&&VIEW==='picks'){g.list.splice(idx,1);g.total--;g.fetched--;}
    }
    else{g.list.splice(idx,1);g.total--;g.fetched--;}
  }
  renderGal(key);
}
async function loadJobs(){
  const j=await (await fetch('/api/jobs')).json();
  const cur=$('batch').value;
  // 日付ごとにまとめる（日付を選ぶとその日の全バッチ表示）
  const days=[];const map={};
  j.jobs.forEach(x=>{
    const d=/^\\d{8}_/.test(x.job)?x.job.slice(0,8):'other';
    if(!map[d]){map[d]={jobs:[],n:0};days.push(d);}
    map[d].jobs.push(x);map[d].n+=x.count;
  });
  function fmtDay(d){return d==='other'?'その他':`${+d.slice(4,6)}/${+d.slice(6,8)}`;}
  $('batch').innerHTML='<option value="">全バッチ</option>'
    +days.map(d=>`<optgroup label="${fmtDay(d)}（${map[d].n}枚）">`
      +(d!=='other'?`<option value="date:${d}">${fmtDay(d)} この日ぜんぶ（${map[d].n}枚）</option>`:'')
      +map[d].jobs.map(x=>`<option value="${x.job}">${fmtJob(x.job)}（${x.count}枚）</option>`).join('')
      +'</optgroup>').join('');
  if([...$('batch').options].some(o=>o.value===cur))$('batch').value=cur;
  // 日付チップ（クリックでその日ぜんぶ表示）。レビューと生成タブの両方に置く
  const chipsHtml=`<button class="chip" data-day="">全部</button>`
    +days.filter(d=>d!=='other').map(d=>`<button class="chip" data-day="date:${d}">${fmtDay(d)}<span style="opacity:.6">・${map[d].n}</span></button>`).join('');
  $('daychips').innerHTML=chipsHtml;
  $('genchips').innerHTML=chipsHtml;
  markChips();
}
function markChips(){
  const v=$('batch').value;
  const day=v===''?'':(v.startsWith('date:')?v:(/^\\d{8}_/.test(v)?'date:'+v.slice(0,8):null));
  document.querySelectorAll('#daychips .chip,#genchips .chip').forEach(c=>c.classList.toggle('on',c.dataset.day===day));
}
document.addEventListener('click',e=>{
  const chip=e.target.closest('#daychips .chip,#genchips .chip');
  if(!chip)return;
  $('batch').value=chip.dataset.day;
  if(chip.parentElement.id==='genchips') // 生成タブから押したらレビュー画面へ移動
    document.querySelector('.tabs button[data-t="rev"]').click();
  else loadReview();
  markChips();
});
async function loadReview(){
  // 日付・バッチ・採用だけ、で絞ったときは上限なしで全部出す
  const filtered=$('batch').value!==''||VIEW==='picks';
  const p=new URLSearchParams({view:VIEW==='picks'?'picks':'all',limit:filtered?0:GAL.rev.limit});
  if($('batch').value)p.set('job',$('batch').value);
  const r=await (await fetch('/api/images?'+p)).json();
  let imgs=r.items;
  if(VIEW==='unjudged')imgs=imgs.filter(i=>!i.picked);
  if(SORT==='score')imgs=imgs.filter(i=>i.score!=null).sort((a,b)=>b.score-a.score);
  if(SORT==='picked')imgs=[...imgs].sort((a,b)=>(b.picked?1:0)-(a.picked?1:0));
  GAL.rev.list=imgs;GAL.rev.total=r.total;GAL.rev.fetched=r.items.length;
  renderGal('rev');
}
$('batch').onchange=()=>{loadReview();markChips();};
$('rowh').value=rowh0;
$('rowh').oninput=e=>{
  document.documentElement.style.setProperty('--rowh',e.target.value+'px');
  localStorage.setItem('rowh',e.target.value);
};
$('lbapply').onclick=()=>{
  const m=LB.meta;if(!m)return;
  if(m.model&&REG[m.model]){$('model').value=m.model;updatePrev();}
  $('prompt').value=m.body||m.prompt||'';
  $('negative').value=m.extraNegative||'';
  if(m.seed)$('seed').value=m.seed;
  if(m.steps)$('steps').value=m.steps;
  if(m.cfg)$('cfg').value=m.cfg;
  const sbtn=[...$('sizeseg').querySelectorAll('button')].find(b=>b.dataset.s===m.size);
  if(sbtn)sbtn.click();
  $('hires').checked=!!m.hires;
  document.querySelectorAll('.lora-ck').forEach(ck=>{
    const spec=(m.lora||[]).find(s=>s.split(':')[0]===ck.dataset.name);
    ck.checked=!!spec;
    const w=spec&&spec.split(':')[1];
    if(w)document.querySelector(`.lora-w[data-name="${ck.dataset.name}"]`).value=w;
  });
  closeLB();
  document.querySelector('.tabs button[data-t="gen"]').click();
  window.scrollTo(0,0);
};
async function loadModels(){
  const ms=await (await fetch('/api/models')).json();
  const ls=await (await fetch('/api/loras')).json();
  $('mlist').innerHTML=ms.map(m=>`<div class="mrow">
    ${m.thumb?`<img class="facecrop" src="${m.thumb}" style="width:52px;height:52px;object-fit:cover;border-radius:8px;cursor:zoom-in;flex-shrink:0" onclick="zoom('${m.thumb}')">`:''}
    <div><div class="nm">${m.label}</div>
    <div class="sub">-m ${m.name}｜${m.sizeGb?m.sizeGb+'GB':m.source}${m.vpred?'｜V-pred':''}${m.exists?'':'｜<span style="color:var(--ng)">ファイル無し</span>'}</div>
    ${m.popularity?`<div class="sub" style="color:#ffd54a">${m.popularity}</div>`:''}
    ${m.notes?`<div class="sub" style="color:var(--ok)">${m.notes}</div>`:''}</div>
    <button onclick="delModel('${m.name}')">削除</button></div>`).join('')
    +(ls.length?'<div style="font-size:12px;font-weight:bold;margin:14px 0 6px">LoRA置き場（味変パーツ）</div>'+
      ls.map(l=>`<div class="mrow">
      ${l.thumb?`<img class="facecrop" src="${l.thumb}" style="width:52px;height:52px;object-fit:cover;border-radius:8px;cursor:zoom-in;flex-shrink:0" onclick="zoom('${l.thumb}')">`:''}
      <div><div class="nm">${l.name}${baseTag(l.base)}</div>
      <div class="sub">${l.sizeMb}MB｜${l.desc?l.desc+'｜':''}生成タブでチェックして使う</div></div>
      <button onclick="delLora('${l.name}')">削除</button></div>`).join(''):'');
}
async function delLora(name){
  if(!confirm(`LoRA「${name}」を削除する？`))return;
  await fetch('/api/loras/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
  loadModels();loadLoras();
}
async function delModel(name){
  if(!confirm(`モデル「${name}」を削除する？（ファイルも消える・戻すには再DL）`))return;
  const r=await (await fetch('/api/models/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})})).json();
  alert(r.freedGb?`${r.freedGb}GB 返却された`:'台帳から外した');
  loadModels();
}
async function civitai(){
  $('cgrid').innerHTML='<span class="hint">Civitaiから取り寄せ中…</span>';
  const p=new URLSearchParams({query:$('cq').value,base:$('cbase').value,sort:$('csort').value,type:$('ctype').value});
  const items=await (await fetch('/api/civitai?'+p)).json();
  if(items.error){$('cgrid').innerHTML=`<span class="hint">${items.error}</span>`;return}
  if(!items.length){$('cgrid').innerHTML='<span class="hint">見つからなかった。条件を変えてみて</span>';return}
  $('cgrid').innerHTML=items.map(m=>`<div class="cit">
    <div class="strip">${(m.samples||[]).map(u=>`<img loading="lazy" src="${u}" onclick="zoom('${u}')">`).join('')}</div>
    <div class="b"><div><div class="n">${m.name}</div>
    <div class="s">${m.base??''} ${m.version??''}｜DL ${m.downloads?.toLocaleString()??'?'}｜高評価 ${m.thumbs?.toLocaleString()??'?'}｜<a href="${m.page}" target="_blank" style="color:var(--acc)">Civitaiで見る</a></div></div>
    <button onclick='addModel(${JSON.stringify(m.name)},${m.versionId},${m.downloads??0},${m.thumbs??0},$('ctype').value)'>追加（DLキューへ）</button></div></div>`).join('');
}
$('csearch').onclick=civitai;
let civitaiLoaded=false;
async function addModel(label,versionId,dl,thumbs,mtype){
  const name=prompt('登録名（英小文字とハイフン）',label.toLowerCase().replace(/[^a-z0-9]+/g,'-').slice(0,20));
  if(!name)return;
  const pop=`DL ${dl.toLocaleString()}・高評価${thumbs.toLocaleString()}（Civitai）`;
  const kind=mtype==='LORA'?'lora':'model';
  const j=await (await fetch('/api/models/add',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,label,versionId:String(versionId),popularity:pop,kind})})).json();
  alert(`DLキューに入れた: ${j.name}（待ち ${j.queued}件）。進み具合は「生成」タブの実況欄で見えるよ`);
}
let genSig='';
async function refreshGen(force){
  const r=await (await fetch('/api/images?limit='+GAL.gen.limit)).json();
  const sig=JSON.stringify(r.items.map(i=>[i.path,i.picked]));
  if(!force&&sig===genSig)return; // 変化がない時は再描画しない（スクロール位置を守る）
  genSig=sig;
  GAL.gen.list=r.items;GAL.gen.total=r.total;GAL.gen.fetched=r.items.length;
  renderGal('gen');
}
async function poll(){
  try{
    const s=await (await fetch('/api/state')).json();
    let t=s.running?`実行中: ${s.running.id} (${s.running.summary})`:'待機中';
    if(s.queue.length)t+=`\\n待ち: ${s.queue.join(' / ')}`;
    if(s.log)t+='\\n---\\n'+s.log;
    $('state').textContent=t;
    if(tabs.gen.style.display!=='none'&&!$('lb').classList.contains('on'))
      await refreshGen();
  }catch(e){}
  setTimeout(poll,4000);
}
poll();
loadJobs();
</script></body></html>"""


if __name__ == "__main__":
    # ポートが埋まっている環境向けに環境変数で変更できる: IMG_FORGE_PORT=4000 python dashboard.py
    port = int(os.environ.get("IMG_FORGE_PORT", "3942"))
    print(f"img-forge dashboard → http://localhost:{port} （停止はCtrl+C）")
    app.run(host="127.0.0.1", port=port, debug=False)
