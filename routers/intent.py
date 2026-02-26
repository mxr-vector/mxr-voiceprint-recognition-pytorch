from typing import Union
from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from services import singleIntentService
from core.response import R

# 创建路由
router = APIRouter(prefix="/intent", tags=["OpenAPI - 意图识别开放接口"])


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic 数据模型
# ──────────────────────────────────────────────────────────────────────────────
class IntentRecognitionRequest(BaseModel):
    text: str = Field(..., description="待识别的语音指令文本（支持中英文混合）")
    threshold: float = Field(
        default=0.55, ge=0.0, le=1.0,
        description="余弦相似度阈值，默认 0.55"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "text": "engine fault detected 并修正航向",
            "threshold": 0.55,
        }
    }}


class IntentItem(BaseModel):
    label: str  = Field(..., description="意图标签")
    score: float = Field(..., description="余弦相似度得分")
    span: str   = Field(..., description="命中的 span 描述（可追溯性）")


class IntentRecognitionData(BaseModel):
    text: str              = Field(..., description="输入文本")
    total: int             = Field(..., description="命中意图总数")
    intents: list[IntentItem] = Field(..., description="识别结果（按得分降序）")


class ReloadIntentsRequest(BaseModel):
    intent_dict: dict[str, list[str]] = Field(
        ...,
        description="新的意图字典，key=意图标签，value=prototype 句子列表",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "intent_dict": {
                "起飞": ["准备起飞", "执行 takeoff"],
                "降落": ["准备降落", "gear down for landing"],
            }
        }
    }}


# ──────────────────────────────────────────────────────────────────────────────
# 接口定义
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/recognition",
    summary="语音指令意图识别",
    response_model=R,
)
async def intent_recognition(req: IntentRecognitionRequest) -> R:
    """
    对输入的语音指令文本进行多意图识别。

    - 支持单意图和多意图（长句）
    - 支持中英文混合输入
    - 双路识别：整句语义 + 子句切分
    """
    results = await singleIntentService.recognize(
        text=req.text,
        threshold=req.threshold,
    )
    data = IntentRecognitionData(
        text=req.text,
        total=len(results),
        intents=[
            IntentItem(label=r.label, score=r.score, span=r.span)
            for r in results
        ],
    )
    return R.success(data.model_dump())


@router.post(
    "/reload",
    summary="热更新意图字典",
    response_model=R,
)
async def reload_intents(req: ReloadIntentsRequest) -> R:
    """
    动态替换意图字典并重新计算 prototype 向量，无需重启服务。
    """
    count = await singleIntentService.reload_intents(req.intent_dict)
    return R.success({"intent_count": count}, msg=f"意图字典已更新，共 {count} 个意图")


@router.get(
    "/intents",
    summary="查询当前意图标签列表",
    response_model=R,
)
async def get_intents() -> R:
    """
    返回当前已加载的所有意图标签，不触发模型加载。
    """
    labels = await singleIntentService.get_intent_labels()
    return R.success({"labels": labels, "total": len(labels)})
