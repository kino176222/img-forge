#!/usr/bin/env python3
"""img-forge: フォルダ内の画像をアニメ美的採点（deepghs・CPU）→ scores.json

例: python score.py output/20260802_2300_overnight
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import score_folder

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python score.py <画像フォルダ>")
    target = Path(sys.argv[1]).expanduser()
    if not target.is_dir():
        sys.exit(f"フォルダが見つからない: {target}")
    score_folder(target)
