"""Load only frozen definitions, never frozen app startup or checkpoints."""
import ast
import math
import re
from types import SimpleNamespace


def definitions(source, names, namespace):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                # Frozen constants may refer to an earlier constant (e.g.
                # QUESTION_SECTION_TIERS uses SUMMARY_SECTION_TIERS). Permit
                # only literal containers/name lookup, never calls or imports.
                allowed = (ast.Constant, ast.Name, ast.Load, ast.Dict, ast.List, ast.Set, ast.Tuple)
                if not all(isinstance(part, allowed) for part in ast.walk(node.value)):
                    continue
                try:
                    value = eval(compile(ast.Expression(node.value), "<frozen-constant>", "eval"),
                                 {"__builtins__": {}}, namespace)
                except (NameError, TypeError):
                    continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = value
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = []
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                         "<frozen-v1-definition>", "exec"), namespace)
    if any(name not in namespace for name in names):
        raise ValueError("Frozen V1 definition missing")
    return namespace


def load(sources, chat, embed):
    import chromadb
    from chromadb.config import Settings
    from chromadb.errors import NotFoundError
    import json
    rag = definitions(sources["rag_store.py"], {"_normalize", "_cos", "_mmr_select", "PaperRAG"},
                      {"math": math, "chromadb": chromadb, "Settings": Settings,
                       "NotFoundError": NotFoundError, "_embed": embed})
    from langchain_core.messages import HumanMessage, SystemMessage
    router = definitions(sources["query_router.py"],
                 {n.name for n in ast.parse(sources["query_router.py"]).body if isinstance(n, ast.FunctionDef)},
                 {"model": chat, "json": json, "re": re, "HumanMessage": HumanMessage, "SystemMessage": SystemMessage})
    app = definitions(sources["app.py"], {"build_evidence", "build_summary_evidence", "build_question_evidence"}, {})
    prompts = definitions(sources["prompts.py"], {"get_system_prompt"}, {})
    middleware = definitions(sources["agent_setup.py"], {"request_prompt"},
                             {"get_system_prompt": prompts["get_system_prompt"]})
    return rag, router, app, middleware


def legacy_answer(sources, chat, embed, chroma, case):
    rag_ns, router, app, middleware = load(sources, chat, embed)
    rag = rag_ns["PaperRAG"](str(chroma))
    route = router["build_route"](case["question"])
    if chat.failed:
        raise RuntimeError("Legacy routing failed; do not silently continue spending")
    evidence = app["build_question_evidence"](rag, case["thread_id"], route) if route["needs_retrieval"] else ""
    guidance = (app["GUIDANCE_B"] if not route["needs_retrieval"] else
                app["ROUTE_GUIDANCE"].get(route["category"], app["ROUTE_GUIDANCE"]["A1"])
                if evidence else app["GUIDANCE_NO_EVIDENCE"])
    prompt = middleware["request_prompt"](SimpleNamespace(runtime=SimpleNamespace(context={"evidence": evidence, "guidance": guidance})))
    from langchain_core.messages import HumanMessage, SystemMessage
    answer = chat.invoke([SystemMessage(content=prompt), HumanMessage(content=case["question"])])
    finish = (answer.response_metadata or {}).get("finish_reason")
    return {"answer": answer.content, "status": "partial" if finish == "length" else "ok",
            "stop_reason": "output_token_limit_partial" if finish == "length" else "legacy_answer",
            "route": route, "evidence": evidence, "explanation": "Frozen V1, fresh context, no checkpoint/tool loop"}
