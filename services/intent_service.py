"""
IntentService — 意图识别业务层
────────────────────────────────
职责：
  - 管理 EmbeddingIntentRecognizer 的生命周期（懒加载、线程安全初始化）
  - 参数从 core.config.args 读取，支持外部覆盖
  - 统一异常处理与业务日志
  - 提供热更新意图字典接口

对外：
  singleIntentService  ← 全局单例，供 routers 层直接导入
"""

from __future__ import annotations

from typing import Optional

from core.config import args as main_args
from core.logger import logger
from mvector.embedding_intent_recognizer import (
    EmbeddingIntentRecognizer,
    DEFAULT_INTENT_DICT,
    IntentResult,
)
from services.base import AsyncServiceBase, run_sync


class __IntentService(AsyncServiceBase):
    """意图识别服务（私有类，外部通过 singleIntentService 单例访问）。"""

    def __init__(self) -> None:
        super().__init__()
        self._recognizer: Optional[EmbeddingIntentRecognizer] = None

        # 从全局 args 读取配置，字段不存在时使用安全默认值
        self._model_name: str = getattr(
            main_args, "intent_model_path", "models/hf/Qwen3-Embedding-4B"
        )
        self._threshold: float = getattr(main_args, "intent_threshold", 0.55)
        self._window_sizes: list[int] = getattr(
            main_args, "intent_window_sizes", [3, 5, 8, 12]
        )

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    def _is_ready(self) -> bool:
        return self._recognizer is not None and self._recognizer.is_ready

    def _create_recognizer(self) -> None:
        """（同步）实际创建推理引擎，仅在锁内调用。"""
        logger.info(f"[IntentService] 加载嵌入模型: {self._model_name}")
        self._recognizer = EmbeddingIntentRecognizer(
            model_name=self._model_name,
            intent_dict=DEFAULT_INTENT_DICT,
            threshold=self._threshold,
            window_sizes=self._window_sizes,
            lazy=False,
        )
        logger.info(
            f"[IntentService] 模型加载完成，"
            f"共 {len(self._recognizer.intent_labels)} 个意图"
        )

    def load(self) -> None:
        """
        主动初始化推理引擎（线程安全，幂等）。
        可在 FastAPI lifespan startup 事件中调用，提前完成模型加载。
        此方法为同步方法，适合在非 async 上下文（如启动钩子）中调用。
        """
        try:
            self._sync_lazy_init(self._is_ready, self._create_recognizer)
        except Exception as e:
            logger.error(f"[IntentService] 模型加载失败: {e}")
            raise

    async def _ensure_loaded(self) -> EmbeddingIntentRecognizer:
        """
        懒加载：首次调用前完成模型初始化。

        注意：模型加载涉及 CUDA 初始化和 device_map，必须在主线程中执行，
        因此使用 _sync_lazy_init 而非 _async_lazy_init。
        模型只加载一次，后续调用直接返回缓存实例。
        """
        try:
            self._sync_lazy_init(self._is_ready, self._create_recognizer)
        except Exception as e:
            logger.error(f"[IntentService] 模型加载失败: {e}")
            raise
        return self._recognizer  # type: ignore[return-value]

    @property
    def is_ready(self) -> bool:
        """服务是否已就绪。"""
        return self._is_ready()

    # ── 业务接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        text: str,
        threshold: Optional[float] = None,
        window_sizes: Optional[list[int]] = None,
    ) -> list[IntentResult]:
        """
        对文本进行多意图识别。

        Parameters
        ----------
        text : str
            待识别文本（支持中英文混合）。
        threshold : float, optional
            本次调用阈值，不传则使用服务默认值。
        window_sizes : list[int], optional
            本次滑窗尺寸，不传则使用服务默认值。

        Returns
        -------
        list[IntentResult]
            按得分降序排列，无命中则返回空列表。
        """
        recognizer = await self._ensure_loaded()
        logger.debug(
            f"[IntentService] 识别请求: text='{text[:50]}...'"
            if len(text) > 50
            else f"[IntentService] 识别请求: text='{text}'"
        )
        try:
            # _rw_lock 防止 reload_intents 在推理途中替换 prototype 向量
            async with self._get_async_lock("rw"):
                results = await run_sync(
                    recognizer.predict,
                    text=text,
                    threshold=threshold,
                    window_sizes=window_sizes,
                )
            logger.debug(f"[IntentService] 识别完成，命中 {len(results)} 个意图")
            return results
        except Exception as e:
            logger.error(f"[IntentService] 识别异常: {e}")
            raise

    async def reload_intents(
        self, intent_dict: dict[str, list[str]]
    ) -> int:
        """
        热更新意图字典（重新计算 prototype 向量）。

        Parameters
        ----------
        intent_dict : dict[str, list[str]]
            新的意图字典，完全替换原有意图。

        Returns
        -------
        int
            更新后的意图数量。
        """
        recognizer = await self._ensure_loaded()
        logger.info(f"[IntentService] 热更新意图字典: {len(intent_dict)} 个意图")
        try:
            # _rw_lock 确保更新与推理互斥，防止读取到半更新的 prototype 向量
            async with self._get_async_lock("rw"):
                await run_sync(recognizer.update_intents, intent_dict)
            count = len(recognizer.intent_labels)
            logger.info(f"[IntentService] 意图字典更新完成，共 {count} 个意图")
            return count
        except Exception as e:
            logger.error(f"[IntentService] 意图字典更新失败: {e}")
            raise

    async def get_intent_labels(self) -> list[str]:
        """
        查询当前已加载的意图标签列表（不触发模型加载）。
        """
        if not self.is_ready:
            return []
        return self._recognizer.intent_labels  # type: ignore[union-attr]


# ── 全局单例 ──────────────────────────────────────────────────────────────────
singleIntentService = __IntentService()
