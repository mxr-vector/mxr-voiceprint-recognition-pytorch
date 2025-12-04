import sys
from loguru import logger
from contextvars import ContextVar

request_id_ctx = ContextVar("request_id", default="-")

logger.remove()
logger.add(
    sys.stdout,
    format="{time} {level} {message} [{extra[request_id]}]",
    level="INFO",
)


# 拦截器：自动注入 request_id
def inject_request_id(record):
    record["extra"]["request_id"] = request_id_ctx.get()
    return record


logger = logger.patch(inject_request_id)
