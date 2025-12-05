from fastapi import FastAPI
from core.middleware_auth import TokenAuthMiddleware
from core.gobal_exception import register_exception
from core.auto_import import load_routers
from core.middleware_request_id import RequestIDMiddleware
from core.middleware_access_log import AccessLogMiddleware
from core.logger import logger
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import functools
import argparse
from mvector.utils.utils import add_arguments, print_arguments
import torch
def build_parser():
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
    return parser


# 配置允许跨域的域名
origins = ["http://localhost", "http://localhost:8000", "*"]

app = FastAPI()
# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")
# 注册中间件
app.add_middleware(RequestIDMiddleware)  # 请求ID
app.add_middleware(AccessLogMiddleware)  # 访问日志
app.add_middleware(TokenAuthMiddleware)  # 认证
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许的域名列表
    allow_credentials=True,  # 是否允许发送 Cookie
    allow_methods=["*"],  # 允许的请求方法，* 表示全部
    allow_headers=["*"],  # 允许的请求头
)


# 注册全局异常处理
register_exception(app)

# 路由注册
load_routers(app)


logger.info("声纹识别 Web服务器启动....")


# 启动（仅本地调试用）
if __name__ == "__main__":
    import uvicorn

    # uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, workers=1)
