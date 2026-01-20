from fastapi import HTTPException
from starlette.datastructures import UploadFile
from yeaudio.audio import AudioSegment
from core.config import args
from io import BufferedReader
from tempfile import SpooledTemporaryFile
import numpy as np
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io


# 1小时
MAX_AUDIO_SEC = 3600
#  一小时16k,单声道的wav大约为219.73mb
MAX_FILE_SIZE = 220
# 采样率：所有音频在处理前会被重采样到该采样率（Hz）
SAMPLE_RATE = 16_000

ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",  # mp3
    "audio/mp3",
    "audio/flac",
    "audio/m4a",
    "audio/mp4",  # m4a
    "audio/ogg",
}


def load_audio_segment(
    audio_data,
    sample_rate: int = SAMPLE_RATE,
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
    if audio_segment.sample_rate != sample_rate:
        audio_segment.resample(sample_rate)
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


def show_melspec_gui(audio_array: np.ndarray):
    """
    可视化音频的梅尔谱（用于调试或展示）。
    参数：
      - audio_array: 一维 numpy 数组，已采样到 SAMPLE_RATE
    行为：
      - 使用 librosa 计算 mel spectrogram 并显示图像（matplotlib）。
    注意：
      - 在无显示环境下调用会报错，可通过 is_show 控制是否展示。
    """
    import matplotlib.pyplot as plt

    S = librosa.feature.melspectrogram(
        y=audio_array, sr=SAMPLE_RATE, n_mels=128, fmax=SAMPLE_RATE >> 1
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure().set_figwidth(12)
    librosa.display.specshow(
        S_dB, x_axis="time", y_axis="mel", sr=SAMPLE_RATE, fmax=SAMPLE_RATE >> 1
    )
    plt.colorbar()
    plt.show()


def show_melspec_to_bytes(audio_data) -> bytes:
    """
    可视化音频的梅尔谱（用于调试或展示）。
    参数：
      - audio_data: 音频数据
    行为：
      - 使用 librosa 计算 mel spectrogram 并保存为 webp 格式。
    返回：
      - 一个字节数组，包含 mel spectrogram 的图像数据。
    """
    audio_segment = load_audio_segment(audio_data)
    wav = audio_segment.samples.astype(np.float32) / 32768.0

    S = librosa.feature.melspectrogram(
        y=wav,
        sr=SAMPLE_RATE,
        n_mels=128,
        fmax=SAMPLE_RATE >> 1,
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        S_dB,
        sr=SAMPLE_RATE,
        x_axis="time",
        y_axis="mel",
        fmax=SAMPLE_RATE >> 1,
        ax=ax,
    )
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # 设置中文字体
    font_name = __get_chinese_font()
    matplotlib.rcParams["font.sans-serif"] = [font_name]
    matplotlib.rcParams["axes.unicode_minus"] = False
    ax.set(title="梅尔时频图")

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="webp", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def __get_chinese_font():
    """
    自动选择可用中文字体，兼容 Windows 和 Linux
    """
    # 中文字体优先列表
    preferred_fonts = [
        "SimHei",  # Windows 黑体
        "Microsoft YaHei",  # Windows 微软雅黑
        "Noto Sans CJK SC",  # Linux / Mac 通用思源黑体
        "WenQuanYi Micro Hei",  # Linux 常用字体
    ]

    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    for font in preferred_fonts:
        if font in available_fonts:
            return font

    # 如果没有找到中文字体，返回默认字体
    return matplotlib.rcParams["font.sans-serif"][0]
