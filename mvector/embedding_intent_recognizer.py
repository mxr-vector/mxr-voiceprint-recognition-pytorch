"""
EmbeddingIntentRecognizer — 纯推理引擎
───────────────────────────────────────
职责范围：模型加载、向量计算、余弦匹配。
无业务逻辑、无日志、无异常捕获，所有错误向上抛出由 Service 层处理。

支持两种初始化模式：
  - 立即加载：EmbeddingIntentRecognizer(model_name=..., intent_dict=..., lazy=False)
  - 延迟加载：EmbeddingIntentRecognizer(..., lazy=True)  →  之后调用 load()
"""

from __future__ import annotations

import json
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
    label: str      # 意图标签
    score: float    # 余弦相似度得分
    span: str       # 命中 span 描述（用于可追溯性）

    def __iter__(self):
        """支持 for label, score, desc in results 位置解包。"""
        return iter((self.label, self.score, self.span))


# ──────────────────────────────────────────────────────────────────────────────
# 意图字典加载（从 JSON 文件读取，便于后期迁移到数据库）
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_INTENT_DICT_PATH: str = "dataset/intent_dict.json"


def _load_intent_dict_from_json(path: str | Path) -> dict[str, list[str]]:
    """
    从生产级 JSON 文件加载意图字典。

    JSON 格式要求::

        {
          "metadata": { "version": "...", ... },
          "intents": [
            { "label": "起飞", "category": "塔台", "prototypes": [...] },
            ...
          ]
        }

    Returns
    -------
    dict[str, list[str]]
        {意图标签: 原型短语列表}，与推理引擎所需格式一致。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        entry["label"]: entry["prototypes"]
        for entry in data["intents"]
    }


DEFAULT_INTENT_DICT: dict[str, list[str]] = _load_intent_dict_from_json(
    DEFAULT_INTENT_DICT_PATH
)


# ──────────────────────────────────────────────────────────────────────────────
# 推理引擎
# ──────────────────────────────────────────────────────────────────────────────
class EmbeddingIntentRecognizer:
    """
    基于嵌入模型的多意图识别推理引擎。

    Parameters
    ----------
    model_name : str
        HuggingFace 模型路径或名称。
    intent_dict : dict[str, list[str]] | None
        意图字典，None 则使用 DEFAULT_INTENT_DICT。
    threshold : float
        余弦相似度默认阈值，默认 0.55。
    window_sizes : list[int] | None
        token 滑动窗口大小列表，默认 [3, 5, 8, 12]。
    lazy : bool
        True = 延迟加载，需显式调用 load()；False = 立即加载，默认 False。
    """

    def __init__(
        self,
        model_name: str = "models/hf/Qwen3-Embedding-4B",
        intent_dict: dict[str, list[str]] | None = None,
        threshold: float = 0.55,
        window_sizes: list[int] | None = None,
        lazy: bool = False,
    ) -> None:
        self._model_name = model_name
        self._pending_intent_dict = intent_dict or DEFAULT_INTENT_DICT
        self.threshold = threshold
        self.window_sizes = window_sizes or [3, 5, 8, 12]

        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = None
        self._model = None
        self._intent_labels: list[str] = []
        self._intent_embeddings: torch.Tensor | None = None

        if not lazy:
            self.load()

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """加载模型并构建 prototype 向量（幂等：重复调用无副作用）。"""
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(
            self._model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self._build_prototypes(self._pending_intent_dict)

    @property
    def is_ready(self) -> bool:
        """模型是否已完成加载。"""
        return self._model is not None

    # ── 公开推理接口 ──────────────────────────────────────────────────────────

    def predict(
        self,
        text: str,
        threshold: float | None = None,
        window_sizes: list[int] | None = None,
    ) -> list[IntentResult]:
        """
        双路多意图识别。

        Parameters
        ----------
        text : str
            待识别文本（支持中英文混合）。
        threshold : float, optional
            本次调用的阈值，不传则使用构造时默认值。
        window_sizes : list[int], optional
            本次调用的滑窗尺寸，不传则使用构造时默认值。

        Returns
        -------
        list[IntentResult]
            按得分降序排列，无命中则返回空列表。
        """
        if not self.is_ready:
            raise RuntimeError("推理引擎尚未初始化，请先调用 load()")

        thr = threshold if threshold is not None else self.threshold
        wins = window_sizes or self.window_sizes
        best: dict[str, tuple[float, str]] = {}

        # 路 A：整句 mean pooling 兜底
        sent_emb = self._encode([text])
        self._match(sent_emb, thr, f"[整句] {text}", best)

        # 路 B：token 级滑动窗口
        hidden, tokens = self._encode_token_level(text)
        L = hidden.shape[0]
        for w in wins:
            if w > L:
                continue
            for start in range(L - w + 1):
                span_emb = hidden[start: start + w].mean(dim=0, keepdim=True)
                span_emb = F.normalize(span_emb, p=2, dim=1)
                span_str = self._tokenizer.convert_tokens_to_string(
                    tokens[start: start + w]
                ).strip()
                self._match(span_emb, thr, f"[span={w}t] {span_str}", best)

        return sorted(
            [
                IntentResult(label=label, score=round(score, 6), span=desc)
                for label, (score, desc) in best.items()
            ],
            key=lambda x: x.score,
            reverse=True,
        )

    def update_intents(self, intent_dict: dict[str, list[str]]) -> None:
        """
        热更新意图字典，重新计算 prototype 向量。
        模型未加载时更新缓存，待 load() 时生效。
        """
        if self.is_ready:
            self._build_prototypes(intent_dict)
        else:
            self._pending_intent_dict = intent_dict

    @property
    def intent_labels(self) -> list[str]:
        """当前已加载的意图标签列表。"""
        return list(self._intent_labels)

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    def _build_prototypes(self, intent_dict: dict[str, list[str]]) -> None:
        """预计算每个意图的 prototype 均值向量。"""
        labels: list[str] = []
        emb_list: list[torch.Tensor] = []
        for intent, prototypes in intent_dict.items():
            emb = self._encode(prototypes).mean(dim=0, keepdim=True)
            labels.append(intent)
            emb_list.append(emb)
        self._intent_labels = labels
        self._intent_embeddings = torch.cat(emb_list, dim=0)  # (N, dim)

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> torch.Tensor:
        """整句 mean pooling，返回归一化向量 (B, dim)。"""
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self._device)
        outputs = self._model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (B, seq_len, dim)
        emb = hidden.mean(dim=1).float()
        return F.normalize(emb, p=2, dim=1)

    @torch.no_grad()
    def _encode_token_level(self, text: str) -> tuple[torch.Tensor, list[str]]:
        """返回 token 级上下文 embedding (seq_len, dim) 及 token 字符串列表。"""
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self._device)
        outputs = self._model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0].float()
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        return hidden, tokens

    def _match(
        self,
        vec: torch.Tensor,
        threshold: float,
        span_label: str,
        best: dict[str, tuple[float, str]],
    ) -> None:
        """将向量与所有意图 prototype 比较，更新 best 字典（保留最高分）。"""
        sims = torch.matmul(vec, self._intent_embeddings.T).squeeze(0)
        for i, score in enumerate(sims):
            s = float(score)
            if s > threshold:
                label = self._intent_labels[i]
                if label not in best or s > best[label][0]:
                    best[label] = (s, span_label)
