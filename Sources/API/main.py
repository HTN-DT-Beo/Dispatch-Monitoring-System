from fastapi import FastAPI, UploadFile, File
import uvicorn, shutil, subprocess, uuid, os, tempfile
from pathlib import Path
from fastapi.responses import FileResponse

app = FastAPI()

@app.post("/infer")
async def infer_video(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    tmp_dir = Path(tempfile.gettempdir()) / "dms_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    raw = tmp_dir / f"{uid}.mp4"
    out = tmp_dir / f"{uid}_out.mp4"

    with raw.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    subprocess.run(["python", "Sources/Pipeline/infer.py", raw, out], check=True)

    # Trả thẳng file video
    return FileResponse(
        path=out,
        media_type="video/mp4",
        filename=out.name,
    )
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)