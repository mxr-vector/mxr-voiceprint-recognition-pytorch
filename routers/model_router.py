from typing import Union
from fastapi import APIRouter, UploadFile, File, Form, Body, Query, Path
from services.voiceprint_service import singleVoiceprintService
from core.response import R
from fastapi.responses import JSONResponse

# 创建路由
router = APIRouter(prefix="/model", tags=["OpenAPI - 声纹识别开放接口"])


# 获取音频特征
@router.post("/feature")
async def getEmbedding(audio_data: UploadFile = File(...)) -> Union[R]:
    embedding = await singleVoiceprintService.predict(audio_data.file)
    return R.success(embedding)


# 获取两个音频相似度
@router.post("/similarity")
async def getSimilarity(
    audio_data1: UploadFile = File(...), audio_data2: UploadFile = File(...)
) -> Union[R]:
    similarity, threshold = await singleVoiceprintService.contrast(
        audio_data1.file, audio_data2.file
    )
    return R.success({"similarity": similarity, "threshold": threshold})


# 注册用户音频
@router.post("/register")
async def registerAudio(
    user_id: str = Form(...), audio_data: UploadFile = File(...)
) -> Union[R]:
    result = await singleVoiceprintService.register(user_id, audio_data.file)
    return R.success("注册成功") if result else R.fail("注册失败")


# 识别用户音频
@router.post("/recognize")
async def recognizeAudio(audio_data: UploadFile = File(...)) -> Union[R]:
    user_id, score = await singleVoiceprintService.recognition(audio_data.file)
    return R.success({"user_id": user_id, "score": score})


# 说话人日志识别
@router.post("/speaker_diarization")
async def speaker_diarization(
    speaker_num: int = Form(None), audio_data: UploadFile = File(...)
) -> Union[R]:
    results = await singleVoiceprintService.speaker_diarization(
        audio_data.file, speaker_num
    )
    return R.success(results)


# 获取所有用户
@router.get("/users")
async def getUsers() -> Union[R]:
    users = await singleVoiceprintService.get_users()
    return R.success(users)


# 删除用户音频
@router.delete("/delete")
async def deleteAudio(user_id: str) -> Union[R]:
    result = await singleVoiceprintService.remove_user(user_id)
    return R.success("删除成功") if result else R.fail("删除失败")
