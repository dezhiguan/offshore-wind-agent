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
from functools import lru_cache
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

# 停用词：只收虚词，不收任何领域词。
#
# 不加这张表时，BM25 的 idf 在小语料上会**倒挂**：全库只有 34 个小节，「才」只出现在
# 1 节里（idf 3.15），而「告警」出现在 20 节（idf 0.95）—— 虚词的权重反而高过术语。
# 实测「弄完之后要盯多久才算数」的 top1 是「24006 低穿激活」，唯一贡献词是「才」。
# 这是缺陷不是调优，而且这张表不随语料增长：加多少份手册，中文虚词还是这几十个。
#
# 「要求」「记录」「状态」「条件」「处理」「情形」这类词有领域含义（第 3.3 条 处理要求、
# 第 6.1 条 必备记录），一律不进表。
STOPWORDS = frozenset("""
的 了 吗 呢 吧 是 不 也 就 都 还 又 才 很 太 在 有 和 与 及 对 把 被 给 让 从 到 个 这 那
什么 怎么 哪些 哪 如果 需要 应该 可以 能 要 会 没 没有 之后 之前 自己 是不是 说明 情况
时候 进行 一下 我们 你 我 它 其 以及 或 等 上 下 里 中 做 好 多久 算数 一个 一样 直接
""".split())

# 低置信阈值：top1 得分低于此值时，检索结果大概率没捞到该捞的那节。
# 这是**告警**不是拦截 —— 只写进结果与链路，不改变返回内容，避免误伤真实答案。
LOW_SCORE = 2.0

# top-1 比 top-2 高出这个倍数，就认为"命中哪一节"已经没有悬念，可以直接带回全文，
# 省掉模型再发一次 get_doc_section 的往返（实测一轮往返 3~5 秒）。
# 2.5 不是拍的：实测「24002 变流器心跳 常见原因」的 top1/top2 = 13.61/4.10 = 3.3 倍，
# 而含糊的问法（「弄完之后要盯多久」）前两名基本同分。定在 2.5 把两类分开。
INLINE_DOMINANCE = 2.5


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


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """分词并去虚词。建索引与查询走同一条口径，否则 df 与 tf 对不上。"""
    out = []
    for t in jieba.lcut(text):
        if not t.strip() or _PUNCT.match(t):
            continue
        t = t.lower()
        if not keep_stopwords and t in STOPWORDS:
            continue
        out.append(t)
    return out


@lru_cache(maxsize=1)
def catalog() -> str:
    """两份文档的目录：只有章节标题，没有任何正文。

    注进系统提示词，和 SCHEMA_PROMPT 是同一个道理——题面红线禁的是把整个文档
    放进上下文，目录不在此列，而没有目录模型就只能凭印象猜有哪些条款。实测代价
    是实打实的：问「第 2.5 条和第 8.3 条怎么规定的」（两条都不存在），模型连查
    9 步撞上步数上限才敢说不存在；而同一次回答里列举第六章条款时又漏掉了第 6.4 条
    「最低观察时间」——恰恰是关单判定最常命中的那条，却用的是「第六章只规定……」
    这种完整枚举的口吻。

    刻意不走 Index：建索引要分词（jieba 首次加载约 2.4 秒），而这里只需要扫标题行。
    """
    lines: list[str] = []
    for doc, path in DOCS.items():
        chapter = ""
        grouped: dict[str, list[str]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("### "):
                title = line[4:].strip()
                clause = _CLAUSE_NO.search(title)
                grouped.setdefault(chapter, []).append(
                    "%s %s" % (clause.group(1), title[clause.end():].strip()) if clause else title)
            elif line.startswith("## "):
                chapter = line[3:].strip()
        if not grouped:
            continue
        # 条数由代码数好写进去：模型自己数列表会错（实测列了 9 个却说「只收录 8 个」），
        # 和"禁止心算次数"是同一条纪律——能确定性算出来的，就不要让它数。
        total = sum(len(v) for v in grouped.values())
        lines.append("《%s》（共 %d 章、%d 节）：" % (path.stem, len(grouped), total))
        for chap, items in grouped.items():
            prefix = "  %s（%d 节）：" % (chap, len(items)) if chap else "  "
            lines.append(prefix + " / ".join(items))
    return "\n".join(lines)


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

    def diagnose(self, query: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
        """标注这次检索的可信程度。**判据是标定过的，不是拍的。**

        起因：漏召回原本完全静默 —— 链路显示「命中 4 节」，绿的，而该命中的那节
        排在第 9、第 12。检索质量退化没有任何人会发现。

        标定过程（45 条真实链路检索词，28 条捞到 / 17 条漏了）证伪了三个候选判据：
            top_score    捞到 3.59~21.27，漏了 0.00~10.58  —— 完全重叠
            词项覆盖率   两边中位数都是 1.00              —— 不可分
            未识别词数   捞到中位 1，漏了中位 2            —— 不可分
        按 unknown_terms 报警会标掉 73% 的查询（其中 25 条其实是对的），
        全黄等于全不黄；而且仍漏报 6 条。

        结论：**单条查询自身分不出「漏没漏」** —— 能否命中取决于语料里有没有别的
        小节把它挤下去，不是查询的属性。所以这里只报无歧义的情形（无命中 / 得分为 0），
        其余一律降级为信息字段，不下判断。成体系的召回退化要靠探针集回归
        （tests/test_retriever_recall.py），那才是真正让漏召回可观测的地方。
        """
        terms = tokenize(query)
        unknown = sorted({t for t in terms if self.df.get(t, 0) == 0})
        top = hits[0]["score"] if hits else 0.0
        reasons = []
        if not hits:
            reasons.append("无命中")
        elif top <= 0:
            reasons.append("最高分为 0")
        return {
            "status": "low_confidence" if reasons else "ok",
            "top_score": round(top, 3),
            # 信息字段：解释「为什么可能没捞到」，但不作为报警判据（标定证明不可分）
            "unknown_terms": unknown,
            "reasons": reasons,
            "hint": "换用故障代码、条款号或规程原文用词重试" if reasons else None,
        }

    def inline_target(self, query: str, hits: list[dict[str, Any]]) -> tuple[Chunk, str] | None:
        """判断 top-1 是否确凿到可以在检索结果里直接带回全文。

        **只带 top-1，且只在无悬念时带**。手册一共只有 9 节 3845 字符，top-4 里
        最大的四节合起来就是整本的 59% —— 那不是检索，是把大半本手册搬进上下文，
        题面红线正是冲着这个来的。而单独一节的全文，本来就是 get_doc_section
        会放进去的同一段原文：合并掉那一轮，注入内容不增反减（少一份摘要）。

        两个判据，满足其一：
          · 精确锚点 —— 问句里写了故障码或条款号，而 top-1 正是那一节；
          · 显著领先 —— top-1 得分是 top-2 的 INLINE_DOMINANCE 倍以上。
        召回本身就不可信时（diagnose 报 low_confidence）一律不带：那种情况下把
        全文推过去，等于把一次漏召回放大成一整段"看着像依据"的原文。
        """
        if not hits or hits[0].get("score", 0) <= LOW_SCORE:
            return None
        top = hits[0]
        chunk = next((c for c in self.chunks
                      if c.doc == top.get("doc_key") and c.section_id == top.get("section_id")),
                     None)
        if chunk is None:
            return None
        if set(_FAULT_CODE.findall(query)) & chunk.codes:
            return chunk, "问句里的故障代码精确命中本节"
        if chunk.clause_no and chunk.clause_no in set(_CLAUSE_NO.findall(query)):
            return chunk, "问句里的条款号精确命中本节"
        runner_up = hits[1]["score"] if len(hits) > 1 else 0.0
        if runner_up <= 0 or top["score"] >= INLINE_DOMINANCE * runner_up:
            return chunk, "得分显著领先第二名（%.1f 倍）" % (
                top["score"] / runner_up if runner_up > 0 else float("inf"))
        return None

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
    """检索章节。命中无悬念时，top-1 直接带回全文，省掉一次 get_doc_section 往返。

    带回的是**一节**原文，与 get_doc_section 放进上下文的是同一段；其余命中仍然
    只给摘要。判据与边界见 Index.inline_target。
    """
    index = get_index()
    hits = index.search(query, doc=doc, top_k=top_k)
    recall_check = index.diagnose(query, hits)
    inlined = None
    if recall_check["status"] == "ok":
        target = index.inline_target(query, hits)
        if target:
            chunk, reason = target
            hits = [dict(hits[0], text=chunk.text)] + hits[1:]
            inlined = {"doc": DOC_LABELS[chunk.doc], "doc_key": chunk.doc,
                       "section_id": chunk.section_id, "title": chunk.title,
                       "path": chunk.path, "reason": reason}
    return {
        "ok": True,
        "query": query,
        "hits": hits,
        "recall_check": recall_check,
        # 非空表示 hits[0] 里已经是全文，不必再取一次
        "inlined": inlined,
        "note": "未检索到相关章节，可换用故障代码或条款号重试。" if not hits else None,
    }


def get_doc_section(doc: str, section_id: str) -> dict[str, Any]:
    return get_index().get_section(doc, section_id)
