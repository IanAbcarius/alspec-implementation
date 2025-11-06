# api/mock_server.py
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4
import time

app = FastAPI(title="Local Inference Mock Server")

STATE = {
    "active_model": "demo-base",
    "models": [
        {"model_name": "demo-base", "precision": "fp16", "device": "cuda"},
        {"model_name": "demo-graph", "precision": "fp16", "device": "cuda"},
    ],
    "jobs": {}
}

# ----- models -----
class LoadModelReq(BaseModel):
    model_name: str
    model_path: Optional[str] = None
    precision: str = "fp16"
    device: str = "cuda"

class SwitchReq(BaseModel):
    model_name: str

class UnloadReq(BaseModel):
    model_name: str

@app.get("/models")
def list_models():
    return {"models": STATE["models"], "active_model": STATE["active_model"]}

@app.post("/models/load")
def load_model(req: LoadModelReq):
    if not any(m["model_name"] == req.model_name for m in STATE["models"]):
        STATE["models"].append({
            "model_name": req.model_name,
            "precision": req.precision,
            "device": req.device
        })
    STATE["active_model"] = req.model_name
    return {"ok": True, "active_model": STATE["active_model"]}

@app.post("/models/switch")
def switch_model(req: SwitchReq):
    if not any(m["model_name"] == req.model_name for m in STATE["models"]):
        return JSONResponse(status_code=400, content={"error": "model not found"})
    STATE["active_model"] = req.model_name
    return {"ok": True, "active_model": STATE["active_model"]}

@app.post("/models/unload")
def unload_model(req: UnloadReq):
    STATE["models"] = [m for m in STATE["models"] if m["model_name"] != req.model_name]
    if STATE["active_model"] == req.model_name:
        STATE["active_model"] = STATE["models"][0]["model_name"] if STATE["models"] else None
    return {"ok": True, "active_model": STATE["active_model"]}

# ----- inference -----
# Accept BOTH JSON and multipart:
# - JSON: {"model_name": "...", "input": "...", "batch_size": 1, "stream": false}
# - multipart: form fields + file "input"
@app.post("/inference")
async def inference(
    request: Request,
    model_name: Optional[str] = Form(None),
    batch_size: Optional[int] = Form(1),
    stream: Optional[bool] = Form(False),
    input: UploadFile = File(None),
):
    preview = None

    # If JSON request
    ct = request.headers.get("content-type", "")
    if "application/json" in ct:
        data = await request.json()
        model_name = data.get("model_name", model_name)
        batch_size = data.get("batch_size", batch_size)
        stream = data.get("stream", stream)
        input_text = data.get("input")
        preview = (str(input_text)[:64] if input_text is not None else "(no input)")
    else:
        # Multipart/form-data path
        if input is not None:
            content = await input.read()
            preview = content[:64].decode(errors="ignore")
        else:
            preview = "(no file)"
        # model_name may come via form; if not provided, fallback below

    if model_name is None:
        model_name = STATE["active_model"]

    job_id = str(uuid4())[:8]
    STATE["jobs"][job_id] = {"status": "queued", "model": model_name, "preview": preview}
    time.sleep(0.05)  # simulate brief work
    STATE["jobs"][job_id]["status"] = "succeeded"
    return {"job_id": job_id, "status": "succeeded", "result": {"echo": preview}, "model": model_name}

# ----- system -----
@app.get("/system/status")
def system_status():
    return {"ok": True, "active_model": STATE["active_model"], "models_count": len(STATE["models"])}

@app.get("/system/stats")
def system_stats():
    return {"uptime_s": 123, "jobs_total": len(STATE["jobs"])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.mock_server:app", host="127.0.0.1", port=8000, reload=True)
