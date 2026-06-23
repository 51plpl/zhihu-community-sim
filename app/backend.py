import json
import random
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Self-Space 知乎社区模拟")

# --- Generation endpoint (lazy import) ---

class GenerateRequest(BaseModel):
    question: str
    count: int = 5
    chunks: int = 3
    exclude_ids: list[str] = []

_gen_module = None

def get_generator():
    global _gen_module
    if _gen_module is None:
        import app.generate as _gen_module
    return _gen_module

@app.post("/api/generate")
def generate(req: GenerateRequest):
    gen = get_generator()
    answers = gen.generate_answer(
        question=req.question,
        identity_count=req.count,
        chunks_per_person=req.chunks,
        temperature=0.7,
        exclude_ids=req.exclude_ids if req.exclude_ids else None,
    )
    # Return persona IDs so frontend can track which were used
    result = []
    for a in answers:
        result.append(a)
    return {"answers": result}


# --- Static files ---

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app.backend:app", host="127.0.0.1", port=8080, reload=True)
