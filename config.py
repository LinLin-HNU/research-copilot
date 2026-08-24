from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from pathlib import Path

# 以 config.py 所在目录为基准找 .env，避免依赖当前工作目录
load_dotenv(Path(__file__).parent / '.env')

#========================配置通义千问模型 ===================
model=init_chat_model(
    "qwen3.7-flash-2026-07-15",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model_provider="openai",
    temperature=0.7,
    timeout=30,
    max_tokens=4096,
    max_retries=6,
)