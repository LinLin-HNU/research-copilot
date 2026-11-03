"""加载统一配置，借助统一工厂创建旧路径使用的模型对象"""
from dotenv import load_dotenv
from research_model_settings import build_chat_model
from pathlib import Path

# 以 config.py 所在目录为基准找 .env，避免依赖当前工作目录
load_dotenv(Path(__file__).parent / '.env')

# Provider-specific options are resolved centrally, never hard-coded here.
# Preserve the legacy answer ceiling; share provider, temperature and retries.
model = build_chat_model(max_output_tokens=8192, timeout_seconds=60)
