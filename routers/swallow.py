from fastapi import APIRouter, UploadFile, File, Form, Body, Query, Path
from fastapi.responses import FileResponse, StreamingResponse
from mvector.utils.audio_utils import load_audio_segment, show_melspec_to_bytes
from services import singleSwallowService
from core.response import R
import io

# 创建路由
router = APIRouter(prefix="/swallow", tags=["OpenAPI - 吞音检测开放接口"])


@router.post("/analyze",summary="吞音检测接口")
async def swallow(
    lang: str = Form("chinese", description="语种（chinese/english/japanese/korean）"),
    reference_text: str = Form(None, description="参考文本"),
    audio_data: UploadFile = File(..., description="音频文件"),
) -> R:
    audio_segment = load_audio_segment(audio_data)
    result = await singleSwallowService.analyze(
        lang=lang,
        audio_segment=audio_segment,
        reference_text=reference_text,
    )
    return R.success(result)


@router.post("/preview",summary="显示mel谱")
async def swallow_preview(
    audio_data: UploadFile = File(..., description="音频文件")
) -> FileResponse:
    img_bytes = show_melspec_to_bytes(audio_data)
    return StreamingResponse(
        io.BytesIO(img_bytes),
        media_type="image/webp",
        headers={"Accept-Ranges": "bytes"},
    )
