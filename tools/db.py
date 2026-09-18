# -*- coding: utf-8 -*-
"""只读数据库访问层。

题目要求数据库作为只读资料使用（机试题目说明 第五节）。这里不依赖单一手段，
而是叠四层防护，任何一层单独失效都不会导致写入：

  L1  连接层：URI 以 mode=ro 打开，SQLite 引擎层面拒绝写
  L2  会话层：PRAGMA query_only = ON
  L3  授权层：set_authorizer 白名单，只放行 SELECT / READ / FUNCTION 类操作，
      并且**限定可读对象**：只有两张业务表放行
  L4  语句层：单语句 + SELECT/WITH 开头 + 关键字黑名单（同时负责给出可读的报错）

L4 放在最后，是因为正则是最弱的一环——它的主要价值是产生一句模型看得懂的
错误信息以便重写，而不是充当安全边界。

L3 的读对象限制是 2026-09-18 对抗性测试补的：此前四层拦的全是**写**，没有一层
限制**读什么**。实测 `SELECT name, sql FROM sqlite_master` 与 UNION 拼接均放行，
换个中性问法模型就会把建表语句原样吐出来——挡住它的是模型的判断，不是工具的边界。
本项目 schema 本就写在提示词里，直接危害有限，但这是一条通用的元数据外泄通道。
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "海上风电维检.db"

DEFAULT_LIMIT = 200
QUERY_TIMEOUT_SECONDS = 2.0

# L4 黑名单：命中即拒，附带中文原因
_FORBIDDEN = {
    "INSERT": "写入", "UPDATE": "更新", "DELETE": "删除", "DROP": "删表",
    "ALTER": "改表", "CREATE": "建表", "REPLACE": "替换写入", "TRUNCATE": "清空",
    "ATTACH": "挂载外部数据库", "DETACH": "卸载数据库", "VACUUM": "整理库文件",
    "REINDEX": "重建索引", "PRAGMA": "修改会话设置",
}

# 可读对象白名单：只有这两张业务表。SQLite 的内部表（sqlite_master 等）
# 以及未来误建的任何表都不在其列。
ALLOWED_TABLES = {"alarm_records", "maintenance_records"}

# L4 用：命中即拒并给一句可读的原因。正则挡不住变形写法（那是 L3 的活），
# 它的价值是让模型看懂自己错在哪、据此重写，而不是充当边界。
_INTERNAL_OBJECT = re.compile(
    r"\bsqlite_(?:master|schema|temp_master|temp_schema|sequence|stat\d+)\b|\bpragma_\w+", re.I)

# L3 授权白名单
_ALLOWED_ACTIONS = {
    getattr(sqlite3, name)
    for name in ("SQLITE_SELECT", "SQLITE_READ", "SQLITE_FUNCTION", "SQLITE_RECURSIVE")
    if hasattr(sqlite3, name)
}


class SqlGuardError(Exception):
    """语句未通过守卫。消息会原样回灌给模型，用于重写重试。"""


def _authorizer(action: int, arg1: str | None = None, *_args) -> int:
    """动作白名单 + 读对象白名单。

    SQLITE_READ 的 arg1 是表名、arg2 是列名。只放行两张业务表，
    因此 `sqlite_master`、`pragma_*` 这类元数据读在引擎层就被拒，
    无论它藏在子查询、UNION 还是 CTE 里——正则看不见的地方这一层看得见。
    """
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_READ and arg1 and arg1.lower() not in ALLOWED_TABLES:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _strip_comments(sql: str) -> str:
    """去掉注释，避免 `-- ` 或 `/* */` 把关键字藏起来绕过 L4。"""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return sql


def check_sql(sql: str) -> str:
    """L4 语句层校验。返回可执行的 SQL（可能补了 LIMIT），不通过则抛 SqlGuardError。"""
    if not sql or not sql.strip():
        raise SqlGuardError("SQL 为空。")

    body = _strip_comments(sql).strip().rstrip(";").strip()
    if not body:
        raise SqlGuardError("SQL 去掉注释后为空。")

    # 单语句：去掉结尾分号后不应再有分号
    if ";" in body:
        raise SqlGuardError("只允许执行单条语句，请不要用分号拼接多条 SQL。")

    upper = body.upper()
    if not re.match(r"^\s*(SELECT|WITH)\b", upper):
        raise SqlGuardError("只允许 SELECT 查询（可以用 WITH ... SELECT），本次语句不是查询。")

    for kw, cn in _FORBIDDEN.items():
        if re.search(r"\b%s\b" % kw, upper):
            raise SqlGuardError(
                "数据库为只读题目资料，禁止 %s 操作（检测到关键字 %s）。" % (cn, kw)
            )

    hit = _INTERNAL_OBJECT.search(body)
    if hit:
        raise SqlGuardError(
            "只能查询业务表 %s，不能访问数据库内部元数据（检测到 %s）。"
            % ("、".join(sorted(ALLOWED_TABLES)), hit.group(0))
        )

    # 无 LIMIT 时补一个，避免一次拉回整表
    if not re.search(r"\bLIMIT\b", upper):
        body = "%s LIMIT %d" % (body, DEFAULT_LIMIT)

    return body


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise SqlGuardError("数据库文件不存在：%s" % DB_PATH)
    # L1 连接层只读
    conn = sqlite3.connect("file:%s?mode=ro" % DB_PATH.as_posix(), uri=True)
    conn.row_factory = sqlite3.Row
    # L2 会话层只读
    conn.execute("PRAGMA query_only = ON")
    # L3 授权层白名单（放在 PRAGMA 之后，否则上面这句自己会被拒）
    conn.set_authorizer(_authorizer)
    return conn


def query_db(sql: str) -> dict[str, Any]:
    """执行只读查询。

    永远返回 dict，不向上抛异常——错误以 {"error": ...} 形式回灌给模型重写。
    """
    try:
        executed = check_sql(sql)
    except SqlGuardError as exc:
        return {"ok": False, "error": str(exc), "sql": sql}

    conn = None
    try:
        conn = _connect()
        deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 2000)

        cursor = conn.execute(executed)
        raw = cursor.fetchall()
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = [dict(r) for r in raw]
        return {
            "ok": True,
            "sql": executed,
            "columns": columns,
            "row_count": len(rows),
            "rows": rows,
            "note": "查询无匹配记录。" if not rows else None,
        }
    except sqlite3.OperationalError as exc:
        msg = str(exc)
        if "interrupted" in msg.lower():
            msg = "查询超时（上限 %.0f 秒），请缩小范围或加筛选条件。" % QUERY_TIMEOUT_SECONDS
        return {"ok": False, "error": "SQL 执行失败：%s" % msg, "sql": executed}
    except sqlite3.DatabaseError as exc:
        return {"ok": False, "error": "SQL 执行失败：%s" % exc, "sql": executed}
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------- schema 提示词

SCHEMA_PROMPT = """\
数据库为只读 SQLite，仅两张业务表。

表 alarm_records —— 风机运行告警记录，每行一条告警
  alarm_id      INTEGER  记录编号，主键
  turbine_id    TEXT     风机编号，T01~T09
  turbine_model TEXT     风机型号，同一风机保持一致
  turbine_status TEXT    该条告警对应的运行状态：RUNNING 运行 / STOPPED 停机 / MAINTENANCE 检修 / LIMITED 限功率
  fault_code    TEXT     故障代码，**裸码**如 '24002'（注意：手册标题是 '24002_SC_变流器心跳'）
  fault_name    TEXT     故障名称，完整形如 '24002_SC_变流器心跳'
  severity      TEXT     INFO 提示 / WARNING 警告 / MAJOR 重大 / CRITICAL 严重
  occurred_at   TEXT     告警发生时间，'YYYY-MM-DD HH:MM:SS'
  alarm_status  TEXT     ACTIVE 未解除 / CLEARED 已解除

表 maintenance_records —— 维检工单，每行一张工单
  work_order_id       TEXT     工单编号，主键，形如 'WO-260701'
  turbine_id          TEXT     对应风机编号
  fault_code          TEXT     对应故障代码，裸码
  priority            TEXT     NORMAL 普通 / HIGH 高 / EMERGENCY 紧急
  status              TEXT     OPEN 待安排 / PLANNED 已计划 / IN_PROGRESS 处理中 / COMPLETED 已完成 / CANCELLED 已取消
  created_at          TEXT     工单创建时间，'YYYY-MM-DD HH:MM:SS'
  resolution_note     TEXT     处理措施与关闭记录的综合说明，可为空
  observation_minutes INTEGER  处理后观察时长（分钟），未进入观察阶段可为空
  required_part       TEXT     所需备件名称，不需要备件时为空
  part_available      INTEGER  1 备件可用 / 0 备件不可用 / NULL 不需要备件或未记录

关联与约定
  - 关联工单与告警必须 **同时** 用 turbine_id 和 fault_code，只用其一会串到别的风机或别的故障。
  - 所有记录时间位于 2026-07-01 00:00:00 ~ 2026-07-20 23:59:59。
  - 时间为文本格式，可直接做字符串比较；题目中的时间范围一律按 **闭区间** 理解。
  - COMPLETED 只是工单状态，不能证明关闭过程合规（规程第 6.5 条）。

示例
  -- 最新一条告警
  SELECT turbine_model, turbine_status, occurred_at FROM alarm_records
  WHERE turbine_id = 'T01' ORDER BY occurred_at DESC LIMIT 1;

  -- 时间闭区间内的告警
  SELECT occurred_at, fault_code, alarm_status FROM alarm_records
  WHERE turbine_id = 'T04'
    AND occurred_at BETWEEN '2026-07-10 00:00:00' AND '2026-07-15 23:59:59'
  ORDER BY occurred_at;

  -- 双键关联工单
  SELECT * FROM maintenance_records WHERE turbine_id = 'T05' AND fault_code = '24005';
"""
