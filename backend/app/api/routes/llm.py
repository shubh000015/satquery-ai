"""Interim LLM ask endpoint — used by the workspace when no LoRA adapter is served."""

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.schemas.api import LlmAskRequest, LlmAskResponse
from app.tools.llm_client import ask_llm, llm_configured

router = APIRouter(prefix="/llm", tags=["llm"])


@router.get("/status")
def llm_status() -> dict:
    settings = get_settings()
    return {
        "configured": llm_configured(settings),
        "provider": settings.llm_provider,
        "model": settings.llm_model,
    }


@router.post("/ask", response_model=LlmAskResponse)
def ask(payload: LlmAskRequest) -> LlmAskResponse:
    settings = get_settings()
    if not llm_configured(settings):
        raise HTTPException(status_code=503, detail="SATQUERY_LLM_API_KEY is not set.")
    try:
        reply = ask_llm(
            settings,
            payload.query,
            task=payload.task,
            image_pngs_b64=[payload.image_b64] if payload.image_b64 else None,
            context=payload.context,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return LlmAskResponse(
        answer=reply.answer,
        model=reply.model,
        provider=settings.llm_provider,
        confidence=reply.confidence,
    )
