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

        # 按语言缓存 SwallowPredictor，避免每次请求都重新加载模型
        self._predictors: dict[str, SwallowPredictor] = {}

    def _is_lang_ready(self, lang: str) -> bool:
        return lang in self._predictors

    def _create_predictor(self, lang: str) -> None:
        kwargs = {
            "language": lang,
            "risk_threshold": self.args.risk_threshold,
            "severe_threshold": self.args.severe_threshold,
            "use_gpu": self.args.use_gpu,
            "use_admm": False,
        }
        if self.args.ctc_token_model_path:
            kwargs["token_model_path"] = self.args.ctc_token_model_path
        if self.args.ctc_phoneme_model_path:
            kwargs["phoneme_model_path"] = self.args.ctc_phoneme_model_path
        if self.args.forced_aligner_model_path:
            kwargs["forced_aligner_model_path"] = self.args.forced_aligner_model_path
        self._predictors[lang] = SwallowPredictor(**kwargs)

    async def _get_swallow_predictor(self, lang: str = "zh-cn") -> SwallowPredictor:
        """
        获取（或懒加载）指定语言的 SwallowPredictor，线程安全。
        :param lang: 语言类型，影响模型选择
        :return: SwallowPredictor 对象
        """
        await self._async_lazy_init(
            lock_name=f"init_{lang}",
            check_fn=lambda: self._is_lang_ready(lang),
            init_fn=lambda: self._create_predictor(lang),
        )
        return self._predictors[lang]

    async def analyze(
        self,
        lang: str,
        reference_text: str,
        audio_segment: AudioSegment,
    ) -> dict:
        """
        :param lang: 语言类型
        :param reference_text: 参考文本
        :param audio_segment: 音频数据
        :return: 分析结果
        :rtype: dict
        """
        predictor = await self._get_swallow_predictor(lang)
        return await run_sync(
            predictor.analyze,
            reference_text=reference_text,
            audio_segment=audio_segment,
        )


# ---- 单例 ----
singleSwallowService = __SwallowPredictorService()
