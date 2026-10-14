from typing import Literal

from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    thread_id: str = ""
    user_text: str = ""
    image_url: str = ""
    file_type: str = ""


class DeepResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(default="", max_length=200)
    mode: Literal["paper_deep", "general"] = "paper_deep"
    confirm_external_processing: bool = False


class AdaptiveResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(default="", max_length=200)
    strategy: Literal["auto", "fast", "deep"] = "auto"
    answer_depth: Literal["beginner", "concise", "advanced"] = "beginner"
    history_mode: Literal["fresh", "continue"] = "continue"
    explanation_pipeline: Literal["direct", "structured"] = "direct"
    enable_quality_review: bool | None = None
    confirm_external_processing: bool = False

class HistoryItem(BaseModel):
    thread_id: str = ""
    title: str = ""
    created_at: str = ""
    updated_at: str = ""

class HistoryResponse(BaseModel):
    sessions: list[HistoryItem]

class DeleteResponse(BaseModel):
    message: str
