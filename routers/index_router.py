from fastapi.responses import FileResponse
from fastapi import FastAPI

app = FastAPI()
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")
