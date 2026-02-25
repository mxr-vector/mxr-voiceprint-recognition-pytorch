from typing import Union
from fastapi import APIRouter, UploadFile, File, Form, Body, Query, Path
from fastapi.responses import FileResponse, StreamingResponse
import io
from services import singleVoiceprintService, singleSwallowService
from core.response import R
from mvector.utils.audio_utils import load_audio_segment, show_melspec_to_bytes

# 创建路由
router = APIRouter(prefix="/model", tags=["OpenAPI - 音频表征识别开放接口"])


@router.post("/feature",summary="获取音频特征")
async def getEmbedding(
    audio_data: UploadFile = File(..., description="音频文件"),
) -> Union[R]:
    audio_segment = load_audio_segment(audio_data, is_voiceprint=True)
    embedding = await singleVoiceprintService.predict(audio_segment)
    return R.success(embedding)

@router.post("/similarity",summary="获取两个音频相似度")
async def getSimilarity(
    audio_data1: UploadFile = File(..., description="音频文件1"),
    audio_data2: UploadFile = File(..., description="音频文件2"),
) -> Union[R]:
    audio_segment1 = load_audio_segment(audio_data1, is_voiceprint=True)
    audio_segment2 = load_audio_segment(audio_data2, is_voiceprint=True)
    similarity, threshold = await singleVoiceprintService.contrast(
        audio_segment1, audio_segment2
    )
    return R.success({"similarity": similarity, "threshold": threshold})

@router.post("/register",summary="注册用户音频")
async def registerAudio(
    storage_id: str = Form(..., description="声纹id"),
    audio_data: UploadFile = File(..., description="音频文件"),
) -> Union[R]:
    audio_segment = load_audio_segment(audio_data, is_voiceprint=True)
    is_save, storage_id, audio_path = await singleVoiceprintService.register(
        storage_id, audio_segment
    )
    return (
        R.success({"storage_id": storage_id, "audio_path": audio_path})
        if is_save
        else R.fail("注册失败")
    )

@router.post("/recognition",summary="识别用户音频")
async def recognitionAudio(
    audio_data: UploadFile = File(..., description="音频文件")
) -> Union[R]:
    audio_segment = load_audio_segment(audio_data)
    storage_id, score = await singleVoiceprintService.recognition(audio_segment)
    return R.success({"storage_id": storage_id, "score": score})

@router.post("/speaker_diarization",summary="说话人日志识别", response_model_exclude_none=False)
async def speaker_diarization(
    speaker_num: int = Form(None, description="说话人数量"),
    audio_data: UploadFile = File(..., description="音频文件"),
) -> Union[R]:
    audio_segment = load_audio_segment(audio_data)
    results = await singleVoiceprintService.speaker_diarization(
        audio_segment, speaker_num
    )
    return R.success(results)

@router.get("/users",summary="获取所有用户")
async def getUsers() -> Union[R]:
    users = await singleVoiceprintService.get_users()
    return R.success(users)

@router.delete("/clear",summary="清空用户音频")
async def clearAudio(
    storage_id: str = Query(..., description="用户目录id")
) -> Union[R]:
    result = await singleVoiceprintService.clear_user(storage_id)
    return R.success("删除成功") if result else R.fail("删除失败")

@router.delete("/delete",summary="删除用户声纹")
async def deleteAudio(
    storage_id: str = Query(..., description="用户目录id"),
    audio_path: str = Query(..., description="音频路径"),
) -> Union[R]:
    result = await singleVoiceprintService.delete_audio(storage_id, audio_path)
    return R.success("删除成功") if result else R.fail("删除失败")


@router.get("/preview",summary="声纹检测预览接口")
async def preview(
    file_url: str = Query(..., description="声纹文件相对地址")
) -> Union[FileResponse]:
    return FileResponse(
        path=file_url, media_type="audio/wav", headers={"Accept-Ranges": "bytes"}
    )

@router.post("/swallow",summary="吞音检测接口")
async def swallow(
    lang: str = Form("zh-cn", description="语种"),
    reference_text: str = Form(None, description="参考文本"),
    audio_data: UploadFile = File(..., description="音频文件"),
) -> Union[R]:
    audio_segment = load_audio_segment(audio_data)
    result = await singleSwallowService.analyze(
        lang=lang,
        audio_segment=audio_segment,
        reference_text=reference_text,
    )
    return R.success(result)

@router.post("/swallow/preview",summary="显示mel谱")
async def swallow_preview(
    audio_data: UploadFile = File(..., description="音频文件")
) -> Union[FileResponse]:
    img_bytes = show_melspec_to_bytes(audio_data)
    return StreamingResponse(
        io.BytesIO(img_bytes),
        media_type="image/webp",
        headers={"Accept-Ranges": "bytes"},
    )
