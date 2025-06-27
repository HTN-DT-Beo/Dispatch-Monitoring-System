from fastapi import FastAPI, UploadFile, File
import uvicorn, shutil, subprocess, uuid, os

app = FastAPI()

@app.post("/infer")
async def infer_video(file: UploadFile=File(...)):
    uid = uuid.uuid4().hex
    raw = f"/tmp/{uid}.mp4"; out = f"/tmp/{uid}_out.mp4"
    with open(raw, "wb") as f: shutil.copyfileobj(file.file, f)
    subprocess.run(["python", "src/pipeline/infer.py", raw, out])
    return {"result": out.split('/')[-1]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
