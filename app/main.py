import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from .rag import PolicyRAG

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("policy_qa")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rag"] = PolicyRAG()   # the index is built before the service accepts requests
    yield


app = FastAPI(title="Kestrel Nova Policy Q&A", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def check_length(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        if len(v) > 1000:
            raise ValueError("question must be 1 to 1000 characters")
        return v


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    errs = exc.errors()
    if errs:
        field = ".".join(str(p) for p in errs[0].get("loc", []) if p != "body") or "body"
        msg = f"{field}: {errs[0].get('msg', 'invalid value').removeprefix('Value error, ')}"
    else:
        msg = "invalid request"
    return JSONResponse(status_code=422, content={"error": msg})


@app.exception_handler(Exception)
async def on_unhandled_error(request: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"error": "internal server error"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        r = state["rag"].answer(req.question)
    except Exception as e:
        log.exception("answer failed")
        return JSONResponse(status_code=500,
                            content={"error": f"could not generate an answer ({type(e).__name__})"})
    return {
        "question": req.question,   # exactly as received
        "answered": r.answered,
        "answer": r.answer,
        "citations": [{"doc_id": c.doc_id, "section": c.section, "quote": c.quote}
                      for c in r.citations],
    }
