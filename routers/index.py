from fastapi.responses import FileResponse
from main import app

@app.get("/",summary="静态首页", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")
@app.get("/favicon.ico",summary="图标", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")
