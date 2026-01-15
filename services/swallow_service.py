from typing import Optional
from core.config import args as main_args
import argparse
from mvector.swallow_predictor import SwallowPredictor
from yeaudio.audio import AudioSegment


class __SwallowPredictorService:
    """
    吞音检测服务
    """

    def __init__(self, args: Optional[argparse.Namespace] = None):
        self.args = main_args if args is None else args

    def __get_swallow_predictor(self, lang: str = "zh-cn") -> SwallowPredictor:
        """
        获取模型
        :param language: 语言类型,影响模型选择
        :param token_model_path: 词模型路径,针对中日韩文字做token级别分割检测
        :param phoneme_model_path: 音素模型路径,针对英文做音素级别分割检测
        :param risk_threshold: 模型输出的阈值，低于该阈值的结果会被标记为高风险
        :param severe_threshold: 模型输出的阈值，低于该阈值结果会被标记为疑似
        :param use_gpu: 是否使用GPU
        :param use_admm: 是否使用ADMM,目前未实现
        :return: SwallowPredictor 对象
        """

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
        return SwallowPredictor(**kwargs)

    async def analyze(
        self,
        lang: str,
        reference_text: str,
        audio_segment: AudioSegment,
    ) -> dict:
        """
        :param wav_data: 上传的文件
        :param reference_text: 参考文本
        :return: 分析结果
        :rtype: dict
        """
        res = self.__get_swallow_predictor(lang).analyze(
            reference_text=reference_text,
            audio_segment=audio_segment,
        )
        return res


# ---- 单例 ----
singleSwallowService = __SwallowPredictorService()
