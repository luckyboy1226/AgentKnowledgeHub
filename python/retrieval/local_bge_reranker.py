"""Offline-only local BGE sequence-classification rerank provider."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from retrieval.reranker import (
    RerankRequest,
    RerankScore,
    RerankerMalformedResponse,
    RerankerUnavailableError,
)


@dataclass(frozen=True)
class LocalBGEModelIdentity:
    architecture: str
    model_type: str
    weight_format: str


def validate_local_bge_model_path(model_path: str | Path) -> LocalBGEModelIdentity:
    """Validate only local metadata; never consult a model hub or cache."""
    path = Path(model_path)
    if not path.is_dir():
        raise ValueError("rerank_model_directory_missing")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError("rerank_model_config_missing")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("rerank_model_config_invalid") from exc
    architectures = config.get("architectures")
    architecture = str(architectures[0]) if isinstance(architectures, list) and architectures else ""
    if not architecture.endswith("ForSequenceClassification"):
        raise ValueError("rerank_model_architecture_invalid")
    if not ((path / "tokenizer.json").is_file() or (path / "sentencepiece.bpe.model").is_file()):
        raise ValueError("rerank_tokenizer_missing")
    if (path / "model.safetensors").is_file():
        weight_format = "safetensors"
    elif (path / "pytorch_model.bin").is_file():
        weight_format = "pytorch_bin"
    else:
        raise ValueError("rerank_model_weights_missing")
    return LocalBGEModelIdentity(
        architecture=architecture,
        model_type=str(config.get("model_type") or ""),
        weight_format=weight_format,
    )


def select_local_bge_device(device: str, torch_module: Any) -> str:
    requested = str(device).strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("rerank_device_invalid")
    cuda_available = bool(torch_module.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RerankerUnavailableError("cuda_unavailable")
    return "cuda" if requested == "cuda" or (requested == "auto" and cuda_available) else "cpu"


class _TransformersBGERuntime:
    def __init__(self, model_path: Path, device: str) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            resolved_device = select_local_bge_device(device, torch)
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), local_files_only=True, trust_remote_code=False,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path), local_files_only=True, trust_remote_code=False,
                use_safetensors=(model_path / "model.safetensors").is_file(),
            )
            if resolved_device == "cuda":
                model = model.half()
            self.model = model.to(resolved_device).eval()
            self.tokenizer = tokenizer
            self.torch = torch
            self.device_type = resolved_device
        except RerankerUnavailableError:
            raise
        except Exception as exc:
            raise RerankerUnavailableError("model_load_failed") from exc

    def score_pairs(self, query: str, documents: Sequence[str], *, batch_size: int, max_length: int) -> list[float]:
        scores: list[float] = []
        try:
            for start in range(0, len(documents), batch_size):
                pairs = [[query, document] for document in documents[start:start + batch_size]]
                inputs = self.tokenizer(
                    pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
                )
                inputs = {key: value.to(self.device_type) for key, value in inputs.items()}
                with self.torch.inference_mode():
                    logits = self.model(**inputs).logits.reshape(-1).float().cpu().tolist()
                scores.extend(float(value) for value in logits)
            return scores
        except Exception as exc:
            reason = "rerank_oom" if "out of memory" in str(exc).lower() else "inference_failed"
            raise RerankerUnavailableError(reason) from exc


class LocalBGERerankProvider:
    """Lazy, single-load local provider implementing ``ModelRerankProvider``."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        batch_size: int = 4,
        max_length: int = 512,
        runtime_factory: Callable[[Path, str], Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.identity = validate_local_bge_model_path(self.model_path)
        if str(device).lower() not in {"auto", "cpu", "cuda"}:
            raise ValueError("rerank_device_invalid")
        if batch_size < 1 or batch_size > 128 or max_length < 8 or max_length > 8192:
            raise ValueError("rerank_bounds_invalid")
        self.device = str(device).lower()
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self._runtime_factory = runtime_factory or _TransformersBGERuntime
        self._runtime: Any | None = None
        self._load_lock = threading.Lock()
        self.last_diagnostics: dict[str, Any] = {
            "provider_kind": "local_bge", "device_type": None, "candidate_count": 0,
            "batch_count": 0, "elapsed_ms": 0.0, "fallback_reason": None,
        }

    def _runtime_once(self) -> Any:
        if self._runtime is None:
            with self._load_lock:
                if self._runtime is None:
                    try:
                        self._runtime = self._runtime_factory(self.model_path, self.device)
                    except RerankerUnavailableError:
                        raise
                    except Exception as exc:
                        raise RerankerUnavailableError("model_load_failed") from exc
        return self._runtime

    def _score_sync(self, request: RerankRequest) -> tuple[RerankScore, ...]:
        started = time.perf_counter()
        candidate_ids = [str(document.candidate_id) for document in request.documents]
        if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
            raise RerankerMalformedResponse("candidate identity invalid")
        try:
            runtime = self._runtime_once()
            values = runtime.score_pairs(
                str(request.query), [str(document.text) for document in request.documents],
                batch_size=self.batch_size, max_length=self.max_length,
            )
            if len(values) != len(candidate_ids):
                raise RerankerMalformedResponse("score count mismatch")
            scores = [float(value) for value in values]
            if any(not math.isfinite(value) for value in scores):
                raise RerankerMalformedResponse("non-finite rerank score")
            self.last_diagnostics = {
                "provider_kind": "local_bge", "device_type": str(runtime.device_type),
                "candidate_count": len(candidate_ids),
                "batch_count": math.ceil(len(candidate_ids) / self.batch_size),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "fallback_reason": None,
            }
            return tuple(RerankScore(candidate_id, score) for candidate_id, score in zip(candidate_ids, scores))
        except (RerankerUnavailableError, RerankerMalformedResponse) as exc:
            self.last_diagnostics = {
                "provider_kind": "local_bge", "device_type": None,
                "candidate_count": len(candidate_ids),
                "batch_count": math.ceil(len(candidate_ids) / self.batch_size),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "fallback_reason": type(exc).__name__,
            }
            raise
        except Exception as exc:
            self.last_diagnostics = {
                "provider_kind": "local_bge", "device_type": None,
                "candidate_count": len(candidate_ids),
                "batch_count": math.ceil(len(candidate_ids) / self.batch_size),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "fallback_reason": "RerankerUnavailableError",
            }
            raise RerankerUnavailableError("inference_failed") from exc

    async def score(self, request: RerankRequest) -> Sequence[RerankScore]:
        return await asyncio.to_thread(self._score_sync, request)
