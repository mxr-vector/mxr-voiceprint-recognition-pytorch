from mvector.predict import MVectorPredictor
from typing import Union
from mvector.utils.utils import add_arguments, print_arguments
import argparse
import functools
from typing import Optional
from yeaudio.audio import AudioSegment
from fastapi import HTTPException


def _build_parser():
    """
    构建参数解析器
    """
    parser = argparse.ArgumentParser(description="Voiceprint Recognition CLI")
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("configs", str, "configs/cam++.yml", "配置文件")
    # add_arg("audio_path", str, "dataset/test_long.wav", "预测音频路径")
    add_arg("audio_db_path", str, "audio_db/", "音频库路径")
    add_arg("speaker_num", int, None, "说话人数量")
    add_arg("threshold", float, 0.6, "相似度阈值")
    add_arg("record_seconds", int, 10, "录音长度")
    add_arg("model_path", str, "models/CAMPPlus_Fbank/best_model/", "模型文件路径")
    add_arg("search_audio_db", bool, True, help="是否在音频库中搜索对应的说话人")
    add_arg("use_gpu", bool, False, help="是否使用GPU预测")
    print_arguments(args=parser.parse_args())
    return parser


class __VoiceprintService:
    """声纹识别服务"""

    def __init__(self, args: Optional[argparse.Namespace] = None):
        self.args = _build_parser().parse_args() if args is None else args

        if self.args.search_audio_db:
            assert self.args.audio_db_path, "需要指定音频库路径"

        self.predictor = MVectorPredictor(
            configs=self.args.configs,
            model_path=self.args.model_path,
            threshold=self.args.threshold,
            audio_db_path=self.args.audio_db_path,
            use_gpu=self.args.use_gpu,
        )

    async def predict(self, audio_data: object) -> Union[dict, str]:
        """预测

        :param audio_data: 音频数据
        :return: 音频特征
        """
        embedding = self.predictor.predict(audio_data)
        return embedding.tolist()

    async def contrast(
        self, audio_data1: object, audio_data2: object
    ) -> Union[dict, str]:
        """对比两个音频的相似度
        :param audio_data1: 音频数据1
        :param audio_data2: 音频数据2
        :return: 相似度
        """
        similarity = self.predictor.contrast(audio_data1, audio_data2)
        threshold = self.args.threshold
        return float(similarity), float(threshold)

    async def register(self, user_id: str, audio_data: object) -> Union[dict, str]:
        """注册用户音频

        :param user_id: 用户ID
        :param audio_data: 音频数据
        :return: 注册结果
        """
        audio_segment = AudioSegment.from_file(audio_data)
        duration = audio_segment.duration
        max_duration = self.args.record_seconds
        if duration > max_duration:
            raise HTTPException(
                status_code=400,
                detail=f"音频长度超出限制: {duration:.2f}s，最大允许 {max_duration}s",
            )
        result = self.predictor.register(audio_segment, user_id)
        return result

    async def recognition(self, audio_data: object) -> Union[tuple, str]:
        """识别用户音频

        :param audio_data: 音频数据
        :return: 识别结果
        """
        user_id, score = self.predictor.recognition(audio_data)
        return user_id, score

    async def speaker_diarization(
        self, audio_data: object, speaker_num: int = None, search_audio_db: bool = True
    ) -> Union[list, str]:
        """说话人日志识别

        :param audio_path: 音频路径
        :param speaker_num: 说话人数量
        :param search_audio_db: 是否在音频库中搜索对应的说话人
        :return: 说话人日志识别结果
        """
        results = self.predictor.speaker_diarization(
            audio_data, speaker_num=speaker_num, search_audio_db=search_audio_db
        )
        return results

    async def get_users(self) -> Union[list, str]:
        """获取所有用户
        :return: 所有用户列表
        """
        users = self.predictor.get_users()
        return set(users)

    async def remove_user(self, user_id: str) -> Union[dict, str]:
        """删除用户音频

        :param user_id: 用户ID
        :return: 删除结果
        """
        result = self.predictor.remove_user(user_id)
        return result


# ---- 单例 ----
singleVoiceprintService = __VoiceprintService()
