from mvector.predict import MVectorPredictor
from typing import Union
import argparse
from typing import Optional
from yeaudio.audio import AudioSegment
from core.config import args as main_args


class __VoiceprintService:
    """声纹识别服务"""

    def __init__(self, args: Optional[argparse.Namespace] = None):
        self.args = main_args if args is None else args

        if self.args.search_audio_db:
            assert self.args.audio_db_path, "需要指定音频库路径"

    def __get_predictor(self) -> MVectorPredictor:
        return MVectorPredictor(
            configs=self.args.configs,
            model_path=self.args.model_path,
            threshold=self.args.threshold,
            audio_db_path=self.args.audio_db_path,
            use_gpu=self.args.use_gpu,
        )

    async def predict(self, audio_segment: object) -> Union[dict, str]:
        """预测

        :param audio_data: 音频数据
        :return: 音频特征
        """
        embedding = self.__get_predictor().predict(audio_segment)
        return embedding.tolist()

    async def contrast(
        self, audio_segment1: object, audio_segment2: object
    ) -> Union[dict, str]:
        """对比两个音频的相似度
        :param audio_data1: 音频数据1
        :param audio_data2: 音频数据2
        :return: 相似度
        """
        similarity = self.__get_predictor().contrast(audio_segment1, audio_segment2)
        threshold = self.args.threshold
        return float(similarity), float(threshold)

    async def register(
        self, storage_id: str, audio_segment: AudioSegment
    ) -> Union[dict, str]:
        """注册用户音频

        :param user_id: 用户ID
        :param audio_data: 音频数据
        :return: 注册结果
        """
        is_save, storage_id, audio_path = self.__get_predictor().register(
            audio_segment, storage_id
        )
        return is_save, storage_id, audio_path

    async def recognition(self, audio_segment: AudioSegment) -> Union[tuple, str]:
        """识别用户音频

        :param audio_data: 音频数据
        :return: 识别结果
        """
        storage_id, score = self.__get_predictor().recognition(audio_segment)
        return storage_id, score

    async def speaker_diarization(
        self,
        audio_segment: AudioSegment,
        speaker_num: int = None,
        search_audio_db: bool = True,
    ) -> Union[list]:
        """说话人日志识别

        :param audio_path: 音频路径
        :param speaker_num: 说话人数量
        :param search_audio_db: 是否在音频库中搜索对应的说话人
        :return: 说话人日志识别结果
        """
        results = self.__get_predictor().speaker_diarization(
            audio_segment, speaker_num=speaker_num, search_audio_db=search_audio_db
        )
        return results

    async def get_users(self) -> Union[list, str]:
        """获取所有用户
        :return: 所有用户列表
        """
        users = self.__get_predictor().get_users()
        return users

    async def clear_user(self, storage_id: str) -> Union[dict, str]:
        """清空用户音频

        :param storage_id: 用户ID
        :return: 删除结果
        """
        result = self.__get_predictor().clear_user(storage_id)
        return result

    async def delete_audio(self, storage_id: str, audio_path: str) -> Union[dict, str]:
        """删除用户音频

        :return: 删除结果
        """
        result = self.__get_predictor().delete_audio(storage_id, audio_path)
        return result


# ---- 单例 ----
singleVoiceprintService = __VoiceprintService()
