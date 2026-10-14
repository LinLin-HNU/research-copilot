"""One explicitly approved 64-token probe, no paper, history or retries."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark.experiments.run_reading_pilot import save_json


def run(output, confirmed):
    if not confirmed:
        raise ValueError("Needs --confirm-paid for ONE short provider call")
    if not output.is_relative_to((ROOT / "benchmark/experiments/private").resolve()):
        raise ValueError("Output must stay private")
    from dotenv import load_dotenv
    from research_model_settings import resolve_research_model, build_chat_model
    from deep_research_runtime import AliyunResearchModel
    from langchain_core.messages import HumanMessage, SystemMessage
    load_dotenv(ROOT / ".env")
    settings = resolve_research_model()
    if settings.host != "api.deepseek.com" or settings.name != "deepseek-flash":
        raise ValueError("Only the approved DeepSeek model/host may be probed")
    output.mkdir(parents=True, exist_ok=False)
    record = {"model": settings.name, "provider_host": settings.host, "max_paid_calls": 1,
              "requested_output_tokens": 64, "automatic_retries": 0,
              "paper_sent": False, "history_sent": False, "status": "prepared", "http_dispatches": 0}
    save_json(output / "probe.json", record)
    model = build_chat_model(settings=settings, max_output_tokens=64, timeout_seconds=20)

    def capture(request):
        record["http_dispatches"] += 1
        if record["http_dispatches"] > 1:
            raise RuntimeError("Probe forbids a second HTTP dispatch")
        body = json.loads(request.content)
        record["wire_parameters"] = {k: body.get(k) for k in
                    ("model", "max_tokens", "max_completion_tokens", "temperature", "thinking")}
        if request.url.host != settings.host or body.get("max_tokens") != 64 or "max_completion_tokens" in body:
            record["status"] = "blocked_before_dispatch"
            save_json(output / "probe.json", record)
            raise RuntimeError("Probe request contract mismatch")
        record["status"] = "dispatch_pending"
        save_json(output / "probe.json", record)  # No headers/credentials are recorded.

    model.root_client._client.event_hooks["request"].append(capture)
    try:
        answer = model.invoke([
            SystemMessage(content="Follow the user instruction. Do not add an introduction."),
            HumanMessage(content="Print the integers from 1 to 1000 in order, separated by single spaces. Do not skip any integers. Keep going until 1000.")])
        usage = AliyunResearchModel._extract_usage(answer.usage_metadata or {}, answer.response_metadata or {})
        finish = (answer.response_metadata or {}).get("finish_reason")
        n = usage.get("output_tokens")
        verdict = ("limit_triggered" if isinstance(n, int) and n <= 64 and finish == "length"
                   else "exceeded" if isinstance(n, int) and n > 64
                   else "within_limit_not_triggered" if isinstance(n, int) else "usage_unknown")
        record.update(status="returned", verdict=verdict, finish_reason=finish,
                      usage=usage, response_text=answer.content)
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__,
                      **AliyunResearchModel.failure_metadata(exc))
        # Do not print an exception message that might contain private request data.
    finally:
        save_json(output / "probe.json", record)
        model.root_client.close()
    return {k: record.get(k) for k in ("status", "verdict", "http_dispatches", "wire_parameters", "finish_reason", "usage")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-paid", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.output).resolve(), args.confirm_paid), ensure_ascii=False))
