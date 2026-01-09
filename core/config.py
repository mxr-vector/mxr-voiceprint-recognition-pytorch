import functools
import argparse
from mvector.utils.utils import add_arguments, print_arguments
import torch


def __build_parser():
    """
    构建参数解析器
    """
    parser = argparse.ArgumentParser(description="Voiceprint Recognition CLI")
    add_arg = functools.partial(add_arguments, argparser=parser)
    # 声纹检测参数
    add_arg("configs", str, "configs/cam++.yml", "配置文件")
    # add_arg("audio_path", str, "dataset/test_long.wav", "预测音频路径")
    add_arg("audio_db_path", str, "audio_db/", "音频库路径")
    add_arg("speaker_num", int, None, "说话人数量")
    add_arg("threshold", float, 0.6, "相似度阈值")
    add_arg("record_seconds", int, 10, "录音时长")
    # 说话人分离模型 如cam++, ERes2NetV2，ERes2Net
    add_arg(
        "speaker_embedding_model_path",
        str,
        "models/speaker/CAMPPlus_Fbank/best_model/",
        "说话人分离模型文件路径",
    )
    add_arg("search_audio_db", bool, True, help="是否在音频库中搜索对应的说话人")

    # 吞音检测参数
    # 语音表征CTC模型 如wav2vec2，HuBERT, M-CTC-T
    """
    字符集检测模型，用于加载 wav2vec2 中文模型，
    该模型可替换为任意字符级语言模型如 中 日 韩 
    """
    add_arg(
        "ctc_token_model_path",
        str,
        "./models/ctc/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        "语音表征CTC 字符检测模型文件路径",
    )
    """
    音素级检测模型，用于加载 wav2vec2 英文模型xl 模型，
    用户输入必须映射为 语言规范编码，如 en-us、zh-cn、ja-jp、ko-kr
    该目前使用的是espeak-ng音素检测模型，额外需要检查模型语言支持 
    espeak --voices
    """
    add_arg(
        "ctc_phoneme_model_path",
        str,
        "./models/ctc/facebook/wav2vec2-large-960h-lv60-self",
        "语音表征CTC 音素检测模型文件路径",
    )
    add_arg("risk_threshold", float, 0.5, " 高风险阈值")
    add_arg("severe_threshold",float, 0.7, " 疑似吞音风险阈值")
    # 通用参数
    add_arg("use_gpu", bool, torch.cuda.is_available(), help="是否使用GPU预测")
    add_arg("web_secret_key", str, "voiceprint-open-api-token", "接口请求秘钥")
    add_arg("host", str, "0.0.0.0", "服务启动IP地址")
    add_arg("port", int, 8000, "服务启动端口")
    add_arg("base_url", str, "/voiceprint/api/v1", "接口基础路径")
    return parser


args = __build_parser().parse_args()
