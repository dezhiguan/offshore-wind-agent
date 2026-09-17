# -*- coding: utf-8 -*-
"""Markdown 文档检索层。

题面红线：不得把整份文档放进 LLM 上下文，必须做检索。这里的检索单位是
**一个标题小节**——故障手册按故障码（###），安全规程按「第 x.x 条」（###）——
所以命中结果天然就是「章节或条款」，可以直接展示给用户当依据。

打分 = BM25 + 精确命中加权。语料只有约 16 KB，BM25 足够且完全可解释；
不引入向量库是刻意取舍，见 设计说明。

一个必须处理的坑：数据库里 fault_code 是裸码 '24002'，而手册标题是
'24002_SC_变流器心跳'。不建立映射的话，agent 拿着库里查到的码去检索文档会
静默落空——不报错、答案照出，最难发现。
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import warnings

# jieba 0.42.1 的正则字符串未转义反斜杠，Python 3.12+ 编译时会报 SyntaxWarning。
# 与本项目无关，且上游未修；在导入前静音，避免交付后控制台刷屏。
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"jieba.*")
import jieba

# jieba 首次分词会往 stderr 打四行加载日志。功能上无害，但交付出去在 IDE 控制台里
# 会盖住启动地址，所以静音。
jieba.setLogLevel(logging.WARNING)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DOCS = {
    "fault_manual": DATA_DIR / "故障处理手册.md",
    "safety_regulation": DATA_DIR / "海上风电机组检修作业与安全管理规程.md",
}
DOC_LABELS = {
    "fault_manual": "故障处理手册.md",
    "safety_regulation": "海上风电机组检修作业与安全管理规程.md",
}
# 允许模型用中文文件名或关键词指定文档
DOC_ALIASES = {
    "故障处理手册.md": "fault_manual", "故障处理手册": "fault_manual", "手册": "fault_manual",
    "海上风电机组检修作业与安全管理规程.md": "safety_regulation",
    "海上风电机组检修作业与安全管理规程": "safety_regulation",
    "安全管理规程": "safety_regulation", "规程": "safety_regulation",
}

_PUNCT = re.compile(r"^[\s\W_]+$", re.U)
_FAULT_CODE = re.compile(r"(?<!\d)(\d{5})(?!\d)")
_CLAUSE_NO = re.compile(r"第\s*(\d+(?:\.\d+)?)\s*条")

K1, B = 1.5, 0.75
CODE_BOOST, CLAUSE_BOOST, TITLE_BOOST = 5.0, 5.0, 3.0


@dataclass
class Chunk:
    doc: str                      # fault_manual / safety_regulation
    section_id: str               # '24002_SC_变流器心跳' 或 '4.1'
    title: str                    # 原始标题行
    chapter: str                  # 所属章
    text: str                     # 小节全文（含 #### 子标题）
    codes: set[str] = field(default_factory=set)
    clause_no: str | None = None
    tokens: list[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        return "%s › %s" % (self.chapter, self.title) if self.chapter else self.title


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in jieba.lcut(text) if t.strip() and not _PUNCT.match(t)]


def _split_sections(raw: str, doc: str) -> list[Chunk]:
    """按 ### 切块，把所属 ## 章标题带上。"""
    chunks: list[Chunk] = []
    chapter = ""
    cur: Chunk | None = None
    buf: list[str] = []

    def flush():
        if cur is not None:
            cur.text = "\n".join(buf).strip()
            cur.tokens = tokenize(cur.title + "\n" + cur.text)
            chunks.append(cur)

    for line in raw.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            flush(); cur, buf = None, []
            chapter = line[3:].strip()
        elif line.startswith("### "):
            flush(); buf = []
            title = line[4:].strip()
            clause = _CLAUSE_NO.search(title)
            cur = Chunk(
                doc=doc,
                section_id=clause.group(1) if clause else title,
                title=title,
                chapter=chapter,
                text="",
                codes=set(_FAULT_CODE.findall(title)),
                clause_no=clause.group(1) if clause else None,
            )
            buf.append(line)
        elif cur is not None:
            buf.append(line)
    flush()
    return chunks


class Index:
    """全库 BM25 索引。语料极小，进程启动时一次性建好即可。"""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        for doc, path in DOCS.items():
            self.chunks.extend(_split_sections(path.read_text(encoding="utf-8"), doc))

        self.df: Counter[str] = Counter()
        for c in self.chunks:
            self.df.update(set(c.tokens))
        self.N = len(self.chunks)
        self.avgdl = sum(len(c.tokens) for c in self.chunks) / max(self.N, 1)
        self.tf = [Counter(c.tokens) for c in self.chunks]

        # 裸码 -> 手册小节，解决 '24002' 与 '24002_SC_变流器心跳' 的口径差
        self.code_map: dict[str, Chunk] = {}
        for c in self.chunks:
            for code in c.codes:
                self.code_map[code] = c

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def _bm25(self, i: int, terms: list[str]) -> float:
        tf, dl = self.tf[i], len(self.chunks[i].tokens)
        score = 0.0
        for t in terms:
            f = tf.get(t, 0)
            if f:
                score += self._idf(t) * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / self.avgdl))
        return score

    def search(self, query: str, doc: str | None = None, top_k: int = 4) -> list[dict[str, Any]]:
        doc_key = DOC_ALIASES.get(doc, doc) if doc else None
        if doc_key is not None and doc_key not in DOCS:
            doc_key = None  # 指定了识别不了的文档就全库搜，不要直接返空

        terms = tokenize(query)
        want_codes = set(_FAULT_CODE.findall(query))
        want_clauses = {m for m in _CLAUSE_NO.findall(query)}
        q_lower = query.lower()

        scored = []
        for i, c in enumerate(self.chunks):
            if doc_key and c.doc != doc_key:
                continue
            score = self._bm25(i, terms)
            if want_codes & c.codes:
                score += CODE_BOOST
            if c.clause_no and c.clause_no in want_clauses:
                score += CLAUSE_BOOST
            if c.title.lower() in q_lower or (len(c.title) > 6 and c.title.lower() in q_lower):
                score += TITLE_BOOST
            if score > 0:
                scored.append((score, c))

        scored.sort(key=lambda x: (-x[0], x[1].section_id))
        return [_hit(c, s) for s, c in scored[:top_k]]

    def get_section(self, doc: str, section_id: str) -> dict[str, Any]:
        doc_key = DOC_ALIASES.get(doc, doc)
        sid = section_id.strip()
        # 允许用裸码取手册小节
        if sid in self.code_map and doc_key in (None, "fault_manual"):
            return _full(self.code_map[sid])
        clause = _CLAUSE_NO.search(sid)
        if clause:
            sid = clause.group(1)
        for c in self.chunks:
            if c.doc == doc_key and (c.section_id == sid or c.title == sid):
                return _full(c)
        # 退一步：跨文档按 section_id 找
        for c in self.chunks:
            if c.section_id == sid or c.title == sid:
                return _full(c)
        return {
            "ok": False,
            "error": "未找到章节 '%s'（文档 %s）。可先用 search_docs 确认可用的 section_id。" % (section_id, doc),
        }


def _snippet(text: str, limit: int = 160) -> str:
    body = "\n".join(l for l in text.splitlines()[1:] if l.strip() and not l.startswith("#"))
    return body[:limit] + ("…" if len(body) > limit else "")


def _hit(c: Chunk, score: float) -> dict[str, Any]:
    return {
        "doc": DOC_LABELS[c.doc],
        "doc_key": c.doc,
        "section_id": c.section_id,
        "title": c.title,
        "path": c.path,
        "score": round(score, 3),
        "snippet": _snippet(c.text),
    }


def _full(c: Chunk) -> dict[str, Any]:
    return {
        "ok": True,
        "doc": DOC_LABELS[c.doc],
        "doc_key": c.doc,
        "section_id": c.section_id,
        "title": c.title,
        "path": c.path,
        "text": c.text,
    }


_INDEX: Index | None = None


def get_index() -> Index:
    global _INDEX
    if _INDEX is None:
        _INDEX = Index()
    return _INDEX


def search_docs(query: str, doc: str | None = None, top_k: int = 4) -> dict[str, Any]:
    hits = get_index().search(query, doc=doc, top_k=top_k)
    return {
        "ok": True,
        "query": query,
        "hits": hits,
        "note": "未检索到相关章节，可换用故障代码或条款号重试。" if not hits else None,
    }


def get_doc_section(doc: str, section_id: str) -> dict[str, Any]:
    return get_index().get_section(doc, section_id)
