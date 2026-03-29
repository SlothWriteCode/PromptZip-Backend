"""
PromptZip Backend — FastAPI + LLMLingua
Compress AI prompts using Microsoft's LLMLingua models.
Deploy free on Hugging Face Spaces.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import tiktoken

# ── Models loaded once at startup ─────────────────────────────────────────────
_models = {}


def get_tokenizer():
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return enc
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, clean up at shutdown."""
    print("Loading LLMLingua models (this may take a moment on first run)...")
    try:
        from llmlingua import PromptCompressor
        # LLMLingua-2 (fast, recommended)
        _models["fast"] = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
        print("✅ LLMLingua-2 loaded")
    except Exception as e:
        print(f"⚠️  LLMLingua-2 failed to load: {e}")

    try:
        from llmlingua import PromptCompressor
        # LLMLingua original
        _models["standard"] = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=False,
            device_map="cpu",
        )
        print("✅ LLMLingua (standard) loaded")
    except Exception as e:
        print(f"⚠️  LLMLingua standard failed to load: {e}")

    try:
        from llmlingua import LongContextualCompressor
        # LongLLMLingua — for long docs
        _models["long"] = LongContextualCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            device_map="cpu",
        )
        print("✅ LongLLMLingua loaded")
    except Exception as e:
        print(f"⚠️  LongLLMLingua failed to load: {e}")

    yield
    _models.clear()


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PromptZip API",
    description="AI-powered prompt compression using Microsoft LLMLingua",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class CompressRequest(BaseModel):
    prompt: str
    ratio: float = 0.5          # 0.5 = compress to 50% of original tokens
    target_token: int = -1      # optional hard token limit (-1 = use ratio)


class CompressResponse(BaseModel):
    compressed: str
    tokens_before: int
    tokens_after: int
    ratio_achieved: float
    model_used: str


# ── Token counting helper ──────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    """Approximate token count using tiktoken (cl100k_base = GPT-4 tokenizer)."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough estimate
        return max(1, len(text.split()) * 4 // 3)


def make_response(original: str, compressed: str, model_used: str) -> CompressResponse:
    tb = count_tokens(original)
    ta = count_tokens(compressed)
    return CompressResponse(
        compressed=compressed,
        tokens_before=tb,
        tokens_after=ta,
        ratio_achieved=round(ta / tb, 3) if tb > 0 else 1.0,
        model_used=model_used,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": list(_models.keys()),
    }


@app.post("/compress/fast", response_model=CompressResponse)
async def compress_fast(req: CompressRequest):
    """LLMLingua-2 — fastest and recommended for most prompts."""
    if "fast" not in _models:
        raise HTTPException(503, "LLMLingua-2 model not loaded")
    try:
        result = _models["fast"].compress_prompt(
            req.prompt,
            rate=req.ratio,
            force_tokens=["\n", ".", "!", "?", ","],
        )
        compressed = result.get("compressed_prompt", req.prompt)
        return make_response(req.prompt, compressed, "LLMLingua-2")
    except Exception as e:
        raise HTTPException(500, f"Compression failed: {str(e)}")


@app.post("/compress/standard", response_model=CompressResponse)
async def compress_standard(req: CompressRequest):
    """LLMLingua — original model, good quality."""
    if "standard" not in _models:
        raise HTTPException(503, "LLMLingua standard model not loaded")
    try:
        result = _models["standard"].compress_prompt(
            req.prompt,
            rate=req.ratio,
            force_tokens=["\n", ".", "!", "?", ","],
        )
        compressed = result.get("compressed_prompt", req.prompt)
        return make_response(req.prompt, compressed, "LLMLingua")
    except Exception as e:
        raise HTTPException(500, f"Compression failed: {str(e)}")


@app.post("/compress/long", response_model=CompressResponse)
async def compress_long(req: CompressRequest):
    """LongLLMLingua — optimised for long documents and context."""
    if "long" not in _models:
        raise HTTPException(503, "LongLLMLingua model not loaded")
    try:
        result = _models["long"].compress_prompt(
            req.prompt,
            rate=req.ratio,
            force_tokens=["\n", ".", "!", "?", ","],
        )
        compressed = result.get("compressed_prompt", req.prompt)
        return make_response(req.prompt, compressed, "LongLLMLingua")
    except Exception as e:
        raise HTTPException(500, f"Compression failed: {str(e)}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
