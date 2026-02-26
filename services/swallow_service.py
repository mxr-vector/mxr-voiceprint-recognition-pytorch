from typing import Optional
from core.config import args as main_args
import argparse
from mvector.swallow_predictor import SwallowPredictor
from yeaudio.audio import AudioSegment
from services.base import AsyncServiceBase, run_sync


class __SwallowPredictorService(AsyncServiceBase):
    """
    吞音检测服务
    """

    def __init__(self, args: Optional[argparse.Namespace] = None):
        super().__init__()
        self.args = main_args if args is None else args

        # 全局共用一份 SwallowPredictor，不再按语言加载
        self._predictor: Optional[SwallowPredictor] = None

    def _is_model_ready(self) -> bool:
        return self._predictor is not None

    def _create_predictor(self) -> None:
        kwargs = {
            "risk_threshold": self.args.risk_threshold,
            "severe_threshold": self.args.severe_threshold,
            "use_gpu": self.args.use_gpu,
            "use_admm": False,
        }
        if getattr(self.args, "acoustic_model_path", None):
            kwargs["acoustic_model_path"] = self.args.acoustic_model_path
        if self.args.forced_aligner_model_path:
            kwargs["forced_aligner_model_path"] = self.args.forced_aligner_model_path
        self._predictor = SwallowPredictor(**kwargs)

    async def _get_swallow_predictor(self) -> SwallowPredictor:
        """
        获取（或懒加载）SwallowPredictor，线程安全。
        """
        await self._async_lazy_init(
            lock_name="init_swallow",
            check_fn=self._is_model_ready,
            init_fn=self._create_predictor,
        )
        return self._predictor

    async def analyze(
        self,
        lang: str,
        reference_text: str,
        audio_segment: AudioSegment,
    ) -> dict:
        """
        :param lang: 语言类型（传递给 Qwen3 强制对齐器）
        :param reference_text: 参考文本
        :param audio_segment: 音频数据
        :return: 分析结果
        """
        predictor = await self._get_swallow_predictor()
        return await run_sync(
            predictor.analyze,
            language=lang,
            reference_text=reference_text,
            audio_segment=audio_segment,
        )


# ---- 单例 ----
singleSwallowService = __SwallowPredictorService()
