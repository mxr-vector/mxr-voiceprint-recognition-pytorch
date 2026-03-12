from fastapi import APIRouter
from pydantic import BaseModel, Field

from services import singleIntentService
from mvector.embedding_intent_recognizer import IntentMeta
from core.config import args
from core.response import R

# 创建路由
router = APIRouter(prefix="/intent", tags=["OpenAPI - 意图识别开放接口"])


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic 数据模型
# ──────────────────────────────────────────────────────────────────────────────
class IntentRecognitionRequest(BaseModel):
    text: str = Field(..., description="待识别的语音指令文本（支持中英文混合）")
    threshold: float = Field(
        default=None, ge=0.0, le=1.0, description="余弦相似度阈值，不传则使用配置默认值"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "engine fault detected 并修正航向",
                "threshold": 0.55,
            }
        }
    }


class IntentItem(BaseModel):
    """识别命中的单个意图结果。"""
    label: str = Field(..., description="意图标签（细粒度子标签）")
    group: str = Field(..., description="意图大类")
    category: str = Field(..., description="意图分类（塔台/进近/区域管制等）")
    action: str = Field(..., description="动作极性（APPROVE/CANCEL/ABORT 等）")
    score: float = Field(..., description="余弦相似度得分")
    span: str = Field(..., description="命中的 span 描述（可追溯性）")


class IntentRecognitionData(BaseModel):
    text: str = Field(..., description="输入文本")
    total: int = Field(..., description="命中意图总数")
    intents: list[IntentItem] = Field(..., description="识别结果（按得分降序）")


class IntentEntry(BaseModel):
    """意图条目 —— 查询 / 热更新 共用模型。"""
    label: str = Field(..., description="意图标签（细粒度子标签）")
    group: str = Field(..., description="意图大类")
    category: str = Field(..., description="意图分类")
    action: str = Field(default="", description="动作极性")
    prototypes: list[str] = Field(..., description="prototype 句子列表")


class ReloadIntentsRequest(BaseModel):
    intents: list[IntentEntry] = Field(
        ..., description="意图元数据列表，完全替换原有意图"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "intents": [
                    {
                        "label": "准备约会", # 子类
                        "group": "出行", # 顶级大类
                        "category": "我", # 主体
                        "action": "APPROVE", # 动作极性
                        "prototypes": ["准备约会","希望约会", "Getting ready for a date"], # 模板句
                    },
                    {
                        "label": "取消约会",
                        "group": "出行",
                        "category": "我",
                        "action": "CANCEL",
                        "prototypes": ["取消约会", "cancel the date"],
                    },
                ]
            }
        }
    }


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
    - 返回细粒度子标签 + 动作极性
    """
    results = await singleIntentService.recognize(
        text=req.text,
        threshold=req.threshold,
    )
    # 只返回得分最高的前 5 个意图
    top_results = results[:5]
    data = IntentRecognitionData(
        text=req.text,
        total=len(top_results),
        intents=[
            IntentItem(
                label=r.label,
                group=r.group,
                category=r.category,
                action=r.action,
                score=r.score,
                span=r.span,
            )
            for r in top_results
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
    提交完整的 IntentMeta 列表（含 label/group/category/action/prototypes）。
    """
    intent_meta = [
        IntentMeta(
            label=e.label,
            group=e.group,
            category=e.category,
            action=e.action,
            prototypes=e.prototypes,
        )
        for e in req.intents
    ]
    count = await singleIntentService.reload_intents(intent_meta)
    return R.success({"intent_count": count}, msg=f"意图字典已更新，共 {count} 个意图")


@router.get(
    "/",
    summary="查询当前意图字典（全量）",
    response_model=R,
)
async def get_intents() -> R:
    """
    返回当前已加载的完整意图字典，包含 label/group/category/action/prototypes，
    与热更新接口格式一致，便于浏览和修改后提交。
    """
    metas = await singleIntentService.get_intent_metas()
    intents = [
        IntentEntry(
            label=m.label,
            group=m.group,
            category=m.category,
            action=m.action,
            prototypes=m.prototypes,
        )
        for m in metas
    ]
    return R.success({"intents": [i.model_dump() for i in intents], "total": len(intents)})
