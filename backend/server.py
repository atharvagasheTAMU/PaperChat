import os
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
os.environ.setdefault("PAPERS_DIR", os.path.join(BACKEND_DIR, "Papers"))
os.environ.setdefault("PAPERS_CHROMA_DIR", os.path.join(BACKEND_DIR, "chroma_papers"))
os.environ.setdefault("PAPERS_VISUALS_DIR", os.path.join(BACKEND_DIR, "visuals"))

from Agents import Research_Paper_Summarizer as rps

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(rps.VISUALS_DIR, exist_ok=True)
app.mount("/visuals", StaticFiles(directory=rps.VISUALS_DIR), name="visuals")


class AskRequest(BaseModel):
    message: str
    history: list[dict] | None = None
    include_visuals: bool | None = None
    max_pages: int | None = None


class VisualRecord(BaseModel):
    url: str
    page: int
    captions: List[str]


class AskResponse(BaseModel):
    answer: str
    visuals: List[VisualRecord]
    show_visuals: bool


@app.get("/api/papers")
def list_papers():
    return {"papers": sorted(rps.get_paper_names())}


@app.post("/api/papers/upload")
async def upload_paper(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(rps.PAPERS_DIR, exist_ok=True)
    target_path = os.path.join(rps.PAPERS_DIR, filename)
    with open(target_path, "wb") as handle:
        content = await file.read()
        handle.write(content)

    rps.reload_indexes()
    return {"papers": sorted(rps.get_paper_names())}


@app.delete("/api/papers/{paper_name}")
def delete_paper(paper_name: str):
    filename = os.path.basename(paper_name)
    target_path = os.path.join(rps.PAPERS_DIR, filename)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="Paper not found.")
    os.remove(target_path)
    rps.reload_indexes()
    return {"papers": sorted(rps.get_paper_names())}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    answer = rps.answer_question(req.message, history=req.history)
    visuals: List[VisualRecord] = []

    include_visuals = (
        req.include_visuals
        if req.include_visuals is not None
        else os.getenv("DEFAULT_INCLUDE_VISUALS", "true").strip().lower()
        in {"1", "true", "yes", "y"}
    )
    max_pages = (
        req.max_pages
        if req.max_pages is not None
        else int(os.getenv("DEFAULT_MAX_PAGES", "3"))
    )
    show_visuals = include_visuals and rps.should_include_visuals(
        req.message, history=req.history
    )
    if show_visuals:
        records = rps.extract_visuals_records_for_query(
            req.message, max_pages=max_pages, history=req.history
        )
        for record in records:
            filename = os.path.basename(record["path"])
            visuals.append(
                VisualRecord(
                    url=f"/visuals/{filename}",
                    page=record["page"],
                    captions=record["captions"],
                )
            )

    return AskResponse(answer=answer, visuals=visuals, show_visuals=show_visuals)

