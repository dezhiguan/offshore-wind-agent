# -*- coding: utf-8 -*-
"""提示词注入探针已并入施压套件 ``eval/pressure.yaml``（X6~X9）。

与质量中心「离线评测 → 施压」同一条路径，跑完后结果会出现在施压表下方。

用法：
    python eval/run.py pressure          # 整组施压（含原 X1~X5 与新增 X6~X9）
    python eval/run.py pressure X6 X8    # 只跑注入形态
    python eval/prompt_injection.py      # 本文件的快捷入口，等价于只跑 X6~X9

已有施压覆盖、不再重复的形态：
  整库倾倒 → X3；写库 → X4；禁止检索确认 T20 → X5；
  施压放弃无法确认 → X1；口述现场条件 → X2。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run import main  # noqa: E402

# 只跑并入施压表、且与 X1~X5 不重复的那几条
INJECTION_IDS = ("X6", "X7", "X8", "X9")


if __name__ == "__main__":
    extra = [a.upper() for a in sys.argv[1:] if a.upper().startswith("X")]
    ids = extra or list(INJECTION_IDS)
    raise SystemExit(main(["pressure", *ids]))
