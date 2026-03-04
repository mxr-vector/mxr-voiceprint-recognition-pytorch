from fastapi import FastAPI
from core.middleware_auth import TokenAuthMiddleware
from core.gobal_exception import register_exception
from core.auto_import import load_routers
from core.middleware_request_id import RequestIDMiddleware
from core.middleware_access_log import AccessLogMiddleware
from core.logger import logger
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from core.config import args


def create_app() -> FastAPI:
    """
    create_app 的 Docstring
    创建 FastAPI 对象
    :return: FastAPI 对象
    :rtype: FastAPI
    """
    # 配置允许跨域的域名
    origins = ["*"]

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    # 挂载静态文件
    app.mount("/static", StaticFiles(directory="static"), name="static")
    # app.mount("/audio_db", StaticFiles(directory="audio_db"), name="audio_db")
    # 注册中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,  # 允许的域名列表
        allow_credentials=True,  # 是否允许发送 Cookie
        allow_methods=["*"],  # 允许的请求方法，* 表示全部
        allow_headers=["*"],  # 允许的请求头
    )
    app.add_middleware(RequestIDMiddleware)  # 请求ID
    app.add_middleware(TokenAuthMiddleware)  # 认证
    app.add_middleware(AccessLogMiddleware)  # 访问日志

    # 注册全局异常处理
    register_exception(app)

    # 路由注册
    load_routers(app)
    return app


app = create_app()

# 启动（仅本地调试用）
if __name__ == "__main__":
    import uvicorn
    import os

    # uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    cpu_count = os.cpu_count() or 1
    workers = cpu_count // 8 + 1
    uvicorn.run("infer:app", host=args.host, port=args.port, reload=False, workers=1)
    # uvicorn.run("main:app", host=args.host, port=args.port, reload=False, workers=1)
    logger.info("声纹识别 Web服务器启动....")
