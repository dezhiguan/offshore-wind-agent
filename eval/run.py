# -*- coding: utf-8 -*-
"""回归跑测。

    python eval/run.py            # 跑 11 条回归用例
    python eval/run.py Q5 Q7      # 只跑指定用例
    python eval/run.py probes     # 跑探针用例（题面之外的同类问题）

每条用例的完整产物落到 eval/out/<id>.json，同时充当离线回放（--replay）的素材，
现场断网时仍能走完整界面流程。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from agent import evalview, grounding  # noqa: E402
from agent.composer import compose  # noqa: E402
from agent.loop import run_agent, warm_up  # noqa: E402
from eval.verdict import check  # noqa: E402

OUT_DIR = ROOT / "eval" / "out"
def load_cases(name: str = "cases"):
    return yaml.safe_load((ROOT / "eval" / ("%s.yaml" % name)).read_text(encoding="utf-8"))


# 判定逻辑与 /eval 页面共用，见 eval/verdict.py


def main(argv: list[str]) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 第一个参数可以是用例文件名（cases / probes），其余为用例 id
    suite = "cases"
    if argv and argv[0] in {"cases", "probes", "pressure"}:
        suite, argv = argv[0], argv[1:]
    wanted = {a.upper() for a in argv} or None
    cases = [c for c in load_cases(suite) if not wanted or c["id"].upper() in wanted]

    # 先预热：不预热的话第一条用例会多背一个索引冷启动（实测约 1.2 秒），
    # 而这批数字正是用来判断优化有没有效果的，不能掺进去。
    warm_up()

    rows, passed = [], 0
    for case in cases:
        t0 = time.monotonic()
        try:
            # 接地校验要和问答页走同一条路：这里不挂，命令行跑出来的产物就没有
            # meta.grounding，「接地校验存疑」那张卡的分母会随"这批是谁跑的"变来变去。
            result = grounding.apply(compose(case["question"], run_agent(case["question"])))
            ok, problems = check(case, result)
        except Exception as exc:
            result, ok, problems = {"error": str(exc)}, False, ["异常：%s" % exc]
        elapsed = int((time.monotonic() - t0) * 1000)
        passed += ok
        rows.append((case["id"], ok, elapsed, result.get("meta", {}).get("steps", 0), problems))
        (OUT_DIR / ("%s.json" % case["id"])).write_text(
            json.dumps({"case": case, "suite": suite, "result": result},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("%-3s %s %5dms %d步 %s" % (
            case["id"], "PASS" if ok else "FAIL", elapsed,
            result.get("meta", {}).get("steps", 0), "; ".join(problems)))

    print("\n%d/%d 通过" % (passed, len(cases)))
    # 与后台「运行全部」共用同一个快照函数：两处各记一份必然对不上，
    # 而「较上次」比的就是这条流水账
    snap = evalview.snapshot(ran=len(cases))
    print("已记录指标快照 %s（本轮跑 %d 条 / 共 %d 条）" % (snap["at"], len(cases), snap["total"]))
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
