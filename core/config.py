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
    add_arg("configs", str, "configs/cam++.yml", "配置文件")
    # add_arg("audio_path", str, "dataset/test_long.wav", "预测音频路径")
    add_arg("audio_db_path", str, "audio_db/", "音频库路径")
    add_arg("speaker_num", int, None, "说话人数量")
    add_arg("threshold", float, 0.6, "相似度阈值")
    add_arg("record_seconds", int, 10, "录音时长")
    add_arg("model_path", str, "models/CAMPPlus_Fbank/best_model/", "模型文件路径")
    add_arg("search_audio_db", bool, True, help="是否在音频库中搜索对应的说话人")
    add_arg("use_gpu", bool, torch.cuda.is_available(), help="是否使用GPU预测")
    add_arg("web_secret_key", str, "voiceprint-open-api-token", "接口请求秘钥")
    add_arg("host", str, "0.0.0.0", "服务启动IP地址")
    add_arg("port", int, 8000, "服务启动端口")
    add_arg("base_url", str, "/voiceprint/api/v1", "接口基础路径")
    return parser
args = __build_parser().parse_args()
