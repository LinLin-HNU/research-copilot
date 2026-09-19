from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from pathlib import Path

# 以 config.py 所在目录为基准找 .env，避免依赖当前工作目录
load_dotenv(Path(__file__).parent / '.env')

#========================配置通义千问模型 ===================
# 该模型默认开启思考（thinking）。在「长系统提示 + 论文证据」下，思考过程会持续
# 很久，叠加 timeout 会在正文开始前就切断流，且思考增量被 langchain 解析成
# content="" 的空 chunk，表现为界面一直「输出中」却没有任何正文。这里全局关闭思考
# （与 query_router 的处理一致）；本任务是基于给定证据的总结/问答，无需长推理。
model=init_chat_model(
    "qwen3.8-omni-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model_provider="openai",
    temperature=0.7,
    timeout=120,
    max_tokens=8192,
    max_retries=6,
    extra_body={"enable_thinking": False},
)