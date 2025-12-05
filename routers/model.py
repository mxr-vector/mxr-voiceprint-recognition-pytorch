from typing import Union
from fastapi import APIRouter, UploadFile, HTTPException, File, Form, Body, Query, Path
from services.voiceprint_service import singleVoiceprintService
from core.response import R
from yeaudio.audio import AudioSegment
from main import build_parser

# 创建路由
router = APIRouter(prefix="/model", tags=["OpenAPI - 声纹识别开放接口"])


# 获取音频特征
@router.post("/feature")
async def getEmbedding(
    audio_data: UploadFile = File(..., description="音频文件"),
) -> Union[R]:
    audio_segment = validate_audio_file(audio_data, is_voiceprint=True)
    embedding = await singleVoiceprintService.predict(audio_segment)
    return R.success(embedding)


# 获取两个音频相似度
@router.post("/similarity")
async def getSimilarity(
    audio_data1: UploadFile = File(..., description="音频文件1"),
    audio_data2: UploadFile = File(..., description="音频文件2"),
) -> Union[R]:
    audio_segment1 = validate_audio_file(audio_data1, is_voiceprint=True)
    audio_segment2 = validate_audio_file(audio_data2, is_voiceprint=True)
    similarity, threshold = await singleVoiceprintService.contrast(
        audio_segment1, audio_segment2
    )
    return R.success({"similarity": similarity, "threshold": threshold})


# 注册用户音频
@router.post("/register")
async def registerAudio(
    user_id: str = Form(..., description="声纹id"),
    audio_data: UploadFile = File(..., description="音频文件"),
) -> Union[R]:
    audio_segment = validate_audio_file(audio_data, is_voiceprint=True)
    result = await singleVoiceprintService.register(user_id, audio_segment)
    return R.success("注册成功") if result else R.fail("注册失败")


# 识别用户音频
@router.post("/recognition")
async def recognitionAudio(
    audio_data: UploadFile = File(..., description="音频文件")
) -> Union[R]:
    audio_segment = validate_audio_file(audio_data)
    user_id, score = await singleVoiceprintService.recognition(audio_segment)
    return R.success({"user_id": user_id, "score": score})


# 说话人日志识别
@router.post("/speaker_diarization")
async def speaker_diarization(
    speaker_num: int = Form(None, description="说话人数量"),
    audio_data: UploadFile = File(..., description="音频文件"),
) -> Union[R]:
    audio_segment = validate_audio_file(audio_data)
    results = await singleVoiceprintService.speaker_diarization(
        audio_segment, speaker_num
    )
    return R.success(results)


# 获取所有用户
@router.get("/users")
async def getUsers() -> Union[R]:
    users = await singleVoiceprintService.get_users()
    return R.success(users)


# 删除用户音频
@router.delete("/delete")
async def deleteAudio(user_id: str = Query(..., description="声纹id")) -> Union[R]:
    result = await singleVoiceprintService.remove_user(user_id)
    return R.success("删除成功") if result else R.fail("删除失败")


ALLOWED_AUDIO_TYPES = ["audio/wav", "audio/mp3", "audio/flac", "audio/wave"]


def validate_audio_file(
    audio_data: UploadFile, is_voiceprint: bool = False
) -> AudioSegment:
    """
    校验上传的音频文件 格式 大小 时长
    :param audio_data: 上传的音频文件
    :param is_voiceprint: 是否为声纹录制音频
    """
    args = build_parser().parse_args()
    "校验文件格式"
    if audio_data.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"文件格式错误: {audio_data.content_type}，仅支持 {ALLOWED_AUDIO_TYPES}",
        )
    "校验文件大小"
    MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB
    audio_data.file.seek(0, 2)  # 移动到文件末尾
    file_size = audio_data.file.tell()
    audio_data.file.seek(0)  # 重置文件指针到开头
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超出限制: {file_size / (1024 * 1024):.2f}MB，最大允许 {MAX_FILE_SIZE / (1024 * 1024)}MB",
        )
    audio_segment = AudioSegment.from_file(audio_data.file)
    "若为录制音频，则校验音频时长"
    if is_voiceprint:
        duration = audio_segment.duration
        max_duration = args.record_seconds
        if duration > max_duration:
            raise HTTPException(
                status_code=400,
                detail=f"音频时长超出限制: {duration:.2f}s，最大允许 {max_duration}s",
            )
    return audio_segment
