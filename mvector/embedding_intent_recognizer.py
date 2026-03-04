"""
EmbeddingIntentRecognizer — 纯推理引擎
───────────────────────────────────────
职责范围：模型加载、向量计算、余弦匹配。
无业务逻辑、无日志、无异常捕获，所有错误向上抛出由 Service 层处理。

支持两种初始化模式：
  - 立即加载：EmbeddingIntentRecognizer(model_name=..., intent_meta=..., lazy=False)
  - 延迟加载：EmbeddingIntentRecognizer(..., lazy=True)  →  之后调用 load()

核心编码方式（Qwen3-Embedding 官方）：
  - last-token pooling + padding_side='left'
  - 查询端使用 Instruction-Aware 编码: "Instruct: {task}\\nQuery:{text}"
  - Prototype (文档端) 不加指令前缀
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel

# ──────────────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class IntentResult:
    """单个意图命中结果。"""
    label: str        # 意图标签（细粒度子标签，如 "许可起飞"）
    score: float      # 余弦相似度得分
    span: str         # 命中 span 描述（用于可追溯性）
    category: str     # 意图分类（如 "塔台"、"进近"、"区域管制"）
    group: str        # 意图大类（如 "起飞"）
    action: str       # 动作极性（如 APPROVE / CANCEL / ABORT）


# ──────────────────────────────────────────────────────────────────────────────
# 意图元数据结构
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class IntentMeta:
    """单个意图的元数据（加载自 JSON）。"""
    label: str
    group: str
    category: str
    action: str
    prototypes: list[str]


# ──────────────────────────────────────────────────────────────────────────────
# 意图字典加载（从 JSON 文件读取，便于后期迁移到数据库）
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_INTENT_DICT_PATH: str = "dataset/intent_dict.json"


def _load_intent_dict_from_json(path: str | Path) -> list[IntentMeta]:
    """
    从生产级 JSON 文件加载意图字典（v2 格式）。

    JSON 格式要求::

        {
          "metadata": { "version": "...", ... },
          "intents": [
            {
              "label": "许可起飞",
              "group": "起飞",
              "category": "塔台",
              "action": "APPROVE",
              "prototypes": [...]
            },
            ...
          ]
        }

    Returns
    -------
    list[IntentMeta]
        意图元数据列表，保留 label/group/category/action 全部字段。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        IntentMeta(
            label=entry["label"],
            group=entry.get("group", entry["label"]),
            category=entry.get("category", ""),
            action=entry.get("action", ""),
            prototypes=entry["prototypes"],
        )
        for entry in data["intents"]
    ]


DEFAULT_INTENT_META: list[IntentMeta] = _load_intent_dict_from_json(
    DEFAULT_INTENT_DICT_PATH
)


# ──────────────────────────────────────────────────────────────────────────────
# Qwen3-Embedding 官方 last-token pooling
# ──────────────────────────────────────────────────────────────────────────────
def _last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Qwen3-Embedding 官方 last-token pooling。
    配合 padding_side='left'，取每个序列最后一个有效 token 的隐藏状态。
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 推理引擎
# ──────────────────────────────────────────────────────────────────────────────

# 意图分类专用指令（仅作用于 query 端编码）
_TASK_INSTRUCTION = (
    "Classify the intent of an aviation or air traffic control communication"
)


# 子句切分正则：按中英文标点、连词拆分多意图长句
_CLAUSE_SPLIT_RE = re.compile(
    r'[，,；;。.！!？?]|\s+(?:并|然后|同时|接着|之后|以及|和|且)\s*'
    r'|\s*(?:and|then|also|while|after)\s+',
    re.IGNORECASE,
)


class EmbeddingIntentRecognizer:
    """
    基于 Qwen3-Embedding 的多意图识别推理引擎。

    Parameters
    ----------
    model_name : str
        HuggingFace 模型路径或名称。
    intent_meta : list[IntentMeta] | None
        意图元数据列表，None 则使用 DEFAULT_INTENT_META。
    threshold : float
        余弦相似度默认阈值，默认 0.55。
    lazy : bool
        True = 延迟加载，需显式调用 load()；False = 立即加载，默认 False。
    """

    def __init__(
        self,
        model_name: str = "models/hf/Qwen3-Embedding-4B",
        intent_meta: list[IntentMeta] | None = None,
        threshold: float = 0.55,
        lazy: bool = False,
    ) -> None:
        self._model_name = model_name
        self._pending_meta = intent_meta or DEFAULT_INTENT_META
        self.threshold = threshold

        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = None
        self._model = None
        self._intent_labels: list[str] = []
        self._intent_metas: list[IntentMeta] = []  # 完整元数据

        # per-prototype 存储：保留每个原型的独立向量以提高判别力
        self._proto_embeddings: torch.Tensor | None = None  # (total_protos, dim)
        self._proto_to_intent: list[int] = []                # proto_idx → intent_idx
        self._intent_masks: list[list[int]] = []             # intent_idx → [proto_indices]
        self._num_intents: int = 0

        if not lazy:
            self.load()

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """加载模型并构建 prototype 向量（幂等：重复调用无副作用）。"""
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            padding_side="left",       # Qwen3-Embedding 要求左填充
        )
        self._model = AutoModel.from_pretrained(
            self._model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self._build_prototypes(self._pending_meta)

    @property
    def is_ready(self) -> bool:
        """模型是否已完成加载。"""
        return self._model is not None

    # ── 公开推理接口 ──────────────────────────────────────────────────────────

    def predict(
        self,
        text: str,
        threshold: float | None = None,
    ) -> list[IntentResult]:
        """
        双路多意图识别。

        路 A：整句 instruction-aware 编码（兜底）
        路 B：子句切分编码（多意图长句检测），按标点/连词自然分句

        Parameters
        ----------
        text : str
            待识别文本（支持中英文混合）。
        threshold : float, optional
            本次调用的阈值，不传则使用构造时默认值。

        Returns
        -------
        list[IntentResult]
            按得分降序排列，无命中则返回空列表。
        """
        if not self.is_ready:
            raise RuntimeError("推理引擎尚未初始化，请先调用 load()")

        thr = threshold if threshold is not None else self.threshold
        best: dict[str, tuple[float, str]] = {}

        # 路 A：整句 instruction-aware last-token-pool 编码
        sent_emb = self._encode_query([text])
        self._match(sent_emb, thr, f"[整句] {text}", best)

        # 路 B：子句切分（按标点/连词自然分句）
        clauses = self._split_clauses(text)
        if len(clauses) > 1:
            # 批量编码所有子句
            clause_embs = self._encode_query(clauses)
            for i, clause in enumerate(clauses):
                self._match(
                    clause_embs[i:i+1], thr,
                    f"[子句] {clause}", best,
                )

        return sorted(
            [
                IntentResult(
                    label=label,
                    score=round(score, 6),
                    span=desc,
                    category=meta.category if meta else "",
                    group=meta.group if meta else "",
                    action=meta.action if meta else "",
                )
                for label, (score, desc, meta) in best.items()
            ],
            key=lambda x: x.score,
            reverse=True,
        )

    def update_intents(self, intent_meta: list[IntentMeta]) -> None:
        """
        热更新意图字典，重新计算 prototype 向量。
        模型未加载时更新缓存，待 load() 时生效。
        """
        if self.is_ready:
            self._build_prototypes(intent_meta)
        else:
            self._pending_meta = intent_meta

    @property
    def intent_labels(self) -> list[str]:
        """当前已加载的意图标签列表。"""
        return list(self._intent_labels)

    @property
    def intent_metas(self) -> list[IntentMeta]:
        """当前已加载的完整意图元数据列表。"""
        return list(self._intent_metas)

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    def _build_prototypes(self, metas: list[IntentMeta]) -> None:
        """预计算每个意图下所有 prototype 的独立向量。"""
        labels: list[str] = []
        all_proto_embs: list[torch.Tensor] = []
        proto_to_intent: list[int] = []

        for intent_idx, meta in enumerate(metas):
            # 原型用文档端编码（不加指令前缀）
            emb = self._encode_document(meta.prototypes)  # (P, dim)
            labels.append(meta.label)
            all_proto_embs.append(emb)
            proto_to_intent.extend([intent_idx] * emb.shape[0])

        self._intent_labels = labels
        self._intent_metas = metas
        self._num_intents = len(labels)
        self._proto_embeddings = torch.cat(all_proto_embs, dim=0)  # (total, dim)
        self._proto_to_intent = proto_to_intent
        self._intent_masks = self._build_intent_masks(proto_to_intent, len(labels))

    @staticmethod
    def _build_intent_masks(
        proto_to_intent: list[int], num_intents: int
    ) -> list[list[int]]:
        """预计算每个意图对应的 prototype 索引列表，避免 _match 中重复扫描。"""
        masks: list[list[int]] = [[] for _ in range(num_intents)]
        for pi, intent_idx in enumerate(proto_to_intent):
            masks[intent_idx].append(pi)
        return masks

    @staticmethod
    def _format_query(text: str) -> str:
        """构建 Qwen3-Embedding 的 instruction-aware query 格式。"""
        return f"Instruct: {_TASK_INSTRUCTION}\nQuery:{text}"

    @torch.no_grad()
    def _encode_query(self, texts: list[str]) -> torch.Tensor:
        """
        Query 端编码：加指令前缀 + last-token pooling。
        返回归一化向量 (B, dim)。
        """
        formatted = [self._format_query(t) for t in texts]
        return self._encode_raw(formatted)

    @torch.no_grad()
    def _encode_document(self, texts: list[str]) -> torch.Tensor:
        """
        Document (prototype) 端编码：不加指令前缀 + last-token pooling。
        返回归一化向量 (B, dim)。
        """
        return self._encode_raw(texts)

    @torch.no_grad()
    def _encode_raw(self, texts: list[str]) -> torch.Tensor:
        """
        底层编码：last-token pooling + L2 归一化。
        """
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)
        outputs = self._model(**inputs)
        emb = _last_token_pool(
            outputs.last_hidden_state, inputs["attention_mask"]
        ).float()
        return F.normalize(emb, p=2, dim=1)


    @staticmethod
    def _split_clauses(text: str) -> list[str]:
        """
        按标点和连词将长句切分为多个子句。
        例如 "engine fault detected 并修正航向" → ["engine fault detected", "修正航向"]
        """
        parts = _CLAUSE_SPLIT_RE.split(text)
        return [p.strip() for p in parts if p and p.strip() and len(p.strip()) >= 2]

    def _match(
        self,
        vec: torch.Tensor,
        threshold: float,
        span_label: str,
        best: dict[str, tuple[float, str]],
    ) -> None:
        """
        Per-prototype max-sim 匹配。
        将查询向量与所有 prototype 向量比较，每个意图取最高分。
        使用预计算的 _intent_masks 避免重复扫描。
        """
        # vec: (1, dim), _proto_embeddings: (total_protos, dim)
        sims = torch.matmul(vec, self._proto_embeddings.T).squeeze(0)  # (total,)

        # 对每个意图取该意图下所有 prototype 中的最大相似度
        for intent_idx, mask in enumerate(self._intent_masks):
            if not mask:
                continue
            max_score = float(sims[mask].max())
            if max_score > threshold:
                label = self._intent_labels[intent_idx]
                meta = self._intent_metas[intent_idx] if self._intent_metas else None
                if label not in best or max_score > best[label][0]:
                    best[label] = (max_score, span_label, meta)
