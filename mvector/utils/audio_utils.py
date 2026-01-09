from fastapi import HTTPException,UploadFile
from yeaudio.audio import AudioSegment
from core.config import args
from io import BufferedReader
from tempfile import SpooledTemporaryFile
import numpy as np
MAX_AUDIO_SEC = 3600 
MAX_FILE_SIZE = 220  # 1小时 一小时16k,单声道的wav大约为219.73mb

ALLOWED_AUDIO_TYPES = [
    "audio/wav",
    "audio/mp3",
    "audio/flac",
    "audio/wave",
    "audio/x-wav",
]


def load_audio_segment(
    audio_data,
    sample_rate: int = 16_000,
    is_voiceprint: bool = False,
) -> AudioSegment:
    """
    校验上传的音频文件 格式 大小 时长
    :param audio_data: 上传的音频文件
    :param is_voiceprint: 是否为声纹录制音频
    """
    """加载音频
    :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy，AudioSegment对象。如果是字节的话，必须是完整的字节文件
    :param sample_rate: 如果传入的事numpy数据，需要指定采样率
    :return: 识别的文本结果和解码的得分数
    """
    # 加载音频文件，并进行预处理
    try:
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, (BufferedReader, SpooledTemporaryFile)):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)
        elif isinstance(audio_data, AudioSegment):
            audio_segment = audio_data
        elif isinstance(audio_data, UploadFile):
            audio_bytes = audio_data.file.read()
            "校验文件格式"
            if audio_data.content_type not in ALLOWED_AUDIO_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件格式错误: {audio_data.content_type}，仅支持 {ALLOWED_AUDIO_TYPES}",
                )
            "校验文件大小"
            file_size_mb = len(audio_bytes) / 1024 / 1024  # MB
            if file_size_mb > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件大小超出限制: {file_size_mb:.2f}MB，最大允许 {MAX_FILE_SIZE / (1024 * 1024)}MB",
                )

            audio_segment = AudioSegment.from_bytes(audio_bytes)
        else:
            raise Exception(f"不支持该数据类型，当前数据类型为：{type(audio_data)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件解析错误: {str(e)}")
    assert (
        audio_segment.duration >= args.configs.dataset_conf.dataset.min_duration
    ), f"音频太短，最小应该为{args.configs.dataset_conf.dataset.min_duration}s，当前音频为{audio_segment.duration}s"

    # ---------- 2. 重采样 + 单声道 ----------
    if audio_segment.sample_rate != args.configs.dataset_conf.dataset.sample_rate:
        audio_segment.resample(args.configs.dataset_conf.dataset.sample_rate)
    "校验音频时长"
    duration = audio_segment.duration
    if duration > MAX_AUDIO_SEC:
        raise HTTPException(
            status_code=400,
            detail=f"音频时长超出限制: {duration:.2f}s, 仅支持 {MAX_AUDIO_SEC}s",
        )

    "若为声纹音频，则校验音频时长"
    if is_voiceprint:
        max_duration = args.record_seconds
        if duration > max_duration:
            raise HTTPException(
                status_code=400,
                detail=f"音频时长超出限制: {duration:.2f}s, 最大允许 {max_duration}s",
            )
    # decibel normalization
    if args.configs.dataset_conf.dataset.use_dB_normalization:
        audio_segment.normalize(target_db=args.configs.dataset_conf.dataset.target_dB)
    return audio_segment
