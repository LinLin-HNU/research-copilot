"""Render an existing local response for inspection; never calls a model."""
import argparse
import html
import json
from pathlib import Path


def render(response, trace):
    lines = []
    for line in response.get("answer", "").splitlines():
        escaped = html.escape(line)
        if line.startswith("### "):
            lines.append("<h2>" + html.escape(line[4:]) + "</h2>")
        elif line.startswith("- "):
            lines.append("<p>" + html.escape(line[2:]) + "</p>")
        elif line.strip():
            lines.append("<p>" + escaped + "</p>")
    model_events = [event for event in trace.get("events", []) if event.get("step") == "model"]
    metadata = {"request_id": response.get("request_id"), "status": response.get("status"),
                "model_calls": trace.get("model_calls"), "input_tokens": trace.get("input_tokens"),
                "output_tokens": trace.get("output_tokens"), "elapsed_ms": trace.get("elapsed_ms"),
                "history_mode": trace.get("request_context", {}).get("history_mode"),
                "model_requests": [event.get("request_meta") for event in model_events]}
    return ("<!doctype html><html lang='zh'><meta charset='utf-8'><title>模型切换后讲解测试</title>"
            "<style>body{font-family:system-ui;max-width:950px;margin:35px auto;padding:20px;line-height:1.9;"
            "background:#f5f7fb;color:#172b4d}article{background:white;padding:30px;border-radius:15px}"
            "h2{font-size:1.2rem;margin-top:28px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
            "<article><h1>Sundial：模型切换后的实际讲解</h1>"
            "<p>这是一次真实接口测试的已保存结果。下面是模型原始讲解，未人工改写；通路成功不等于全部事实已核验。"
            "末尾关于数学细节未提供的断言仍待核对。本次使用原会话历史，不能作为新旧模型的严格质量对照。</p>"
            + "".join(lines) + "<details><summary>管理员测试记录</summary><pre>"
            + html.escape(json.dumps(metadata, ensure_ascii=False, indent=2)) + "</pre></details></article></html>")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    response = json.loads(Path(args.response).read_text(encoding="utf-8"))
    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    path = Path(args.output)
    if path.exists():
        raise ValueError("Do not overwrite an existing report")
    path.write_text(render(response, trace), encoding="utf-8")
    print(json.dumps({"output": str(path.resolve()), "network_calls": 0}))


if __name__ == "__main__":
    main()
