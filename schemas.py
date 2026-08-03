from pydantic import BaseModel

class ChatRequest(BaseModel):
    thread_id: str = ""
    user_text: str = ""
    image_url: str = ""
    file_type: str = ""

class HistoryItem(BaseModel):
    thread_id: str = ""
    title: str = ""
    created_at: str = ""
    updated_at: str = ""

class HistoryResponse(BaseModel):
    sessions: list[HistoryItem]

class DeleteResponse(BaseModel):
    message: str