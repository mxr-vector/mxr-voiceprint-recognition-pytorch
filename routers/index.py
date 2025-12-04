from fastapi.responses import FileResponse
from main import app

# 静态首页
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")
