"""Persist one core-business QA verification response without recording secrets."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


QUESTIONS = {
    "vector": "根据已经入库的文档，张三在星河公司担任什么角色？",
    "graph": "张三和李四通过哪个项目产生关联？两人分别担任什么角色？",
    "followup": "他们共同负责的项目使用哪两种存储技术？",
    "proof": "请只列出你上一次回答中提到的两种存储技术名称。",
}


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=QUESTIONS)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--session-id", default="codex-verification-20260910")
    parser.add_argument("--output-dir", default=".runtime/business-verification")
    args = parser.parse_args()
    started = time.monotonic()
    request_id = f"{args.step}-{int(time.time())}"
    payload = json.dumps({"question": QUESTIONS[args.step], "session_id": args.session_id, "user_id": "codex-verification"}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/api/qa/ask", data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    result = {"request_id": request_id, "timestamp": int(time.time()), "endpoint": "/api/qa/ask", "step": args.step, "session_id": args.session_id}
    output_path = Path(args.output_dir) / f"{args.step}-qa-response.json"
    write_json_atomic(Path(args.output_dir) / f"{args.step}-qa-started.json", result)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=240) as response:
            raw = response.read().decode("utf-8")
            result["http_status"] = response.status
            result["elapsed_seconds"] = round(time.monotonic() - started, 3)
            result["json_parsed"] = True
            result["response"] = json.loads(raw)
    except urllib.error.HTTPError as exc:
        result.update(http_status=exc.code, elapsed_seconds=round(time.monotonic() - started, 3), json_parsed=False, error_summary="HTTP request failed")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        result.update(http_status=None, elapsed_seconds=round(time.monotonic() - started, 3), json_parsed=False, error_summary=type(exc).__name__)
    write_json_atomic(output_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
