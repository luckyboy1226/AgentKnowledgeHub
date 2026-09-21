from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.settings import Settings
from retrieval.context_builder import ContextBuilderV2
from retrieval.local_bge_reranker import (
    LocalBGERerankProvider,
    select_local_bge_device,
    validate_local_bge_model_path,
)
from retrieval.parent_expander import ParentExpander
from retrieval.reranker import (
    ConfigurableModelReranker,
    RerankDocument,
    RerankRequest,
    RerankerMalformedResponse,
    RerankerUnavailableError,
)
from retrieval.reranker_factory import create_reranker
from tests.test_reranker import fused


def model_dir(tmp_path: Path) -> Path:
    path=tmp_path/"local-model"; path.mkdir(parents=True)
    (path/"config.json").write_text(json.dumps({
        "architectures":["XLMRobertaForSequenceClassification"],"model_type":"xlm-roberta",
    }),encoding="utf-8")
    (path/"tokenizer.json").write_text("{}",encoding="utf-8")
    (path/"model.safetensors").write_bytes(b"fixture")
    return path


class FakeRuntime:
    device_type="cpu"
    def __init__(self, values=(0.1,0.9,0.4), error:Exception|None=None):
        self.values=list(values); self.error=error; self.calls=[]
    def score_pairs(self,query,documents,*,batch_size,max_length):
        self.calls.append((query,tuple(documents),batch_size,max_length))
        if self.error: raise self.error
        return self.values[:len(documents)]


def request(*ids):
    return RerankRequest("synthetic query",tuple(RerankDocument(value,f"synthetic {value}") for value in ids))


def test_model_identity_requires_local_sequence_classifier_files(tmp_path):
    identity=validate_local_bge_model_path(model_dir(tmp_path))
    assert identity.architecture=="XLMRobertaForSequenceClassification"
    assert identity.weight_format=="safetensors"
    with pytest.raises(ValueError,match="directory_missing"):
        validate_local_bge_model_path(tmp_path/"missing")


@pytest.mark.parametrize("value",["tpu","gpu",""])
def test_invalid_device_is_rejected(tmp_path,value):
    with pytest.raises(ValueError,match="device_invalid"):
        LocalBGERerankProvider(model_dir(tmp_path),device=value,runtime_factory=lambda *_:FakeRuntime())


def test_device_auto_and_explicit_cuda_are_fail_closed():
    cpu_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
    gpu_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True))
    assert select_local_bge_device("auto",cpu_torch)=="cpu"
    assert select_local_bge_device("auto",gpu_torch)=="cuda"
    with pytest.raises(RerankerUnavailableError,match="cuda_unavailable"):
        select_local_bge_device("cuda",cpu_torch)


@pytest.mark.asyncio
async def test_provider_loads_once_returns_complete_finite_scores_and_safe_diagnostics(tmp_path):
    runtime=FakeRuntime(); loads=[]
    provider=LocalBGERerankProvider(model_dir(tmp_path),batch_size=2,max_length=128,
        runtime_factory=lambda path,device:(loads.append((path,device)) or runtime))
    first=await provider.score(request("A","B","C")); second=await provider.score(request("A","B","C"))
    assert [row.candidate_id for row in first]==["A","B","C"]
    assert len(loads)==1 and len(second)==3
    assert provider.last_diagnostics["candidate_count"]==3
    assert provider.last_diagnostics["batch_count"]==2
    assert "model_path" not in provider.last_diagnostics


@pytest.mark.asyncio
async def test_duplicate_ids_and_non_finite_scores_are_rejected(tmp_path):
    provider=LocalBGERerankProvider(model_dir(tmp_path),runtime_factory=lambda *_:FakeRuntime())
    with pytest.raises(RerankerMalformedResponse,match="identity"):
        await provider.score(request("A","A"))
    bad=LocalBGERerankProvider(model_dir(tmp_path/"other"),runtime_factory=lambda *_:FakeRuntime((float("nan"),)))
    with pytest.raises(RerankerMalformedResponse,match="non-finite"):
        await bad.score(request("A"))


class EmptyParents:
    def get_parent(self,*_): return None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure",["load","inference"])
async def test_load_or_inference_failure_falls_back_to_rrf(tmp_path,failure):
    def factory(*_):
        if failure=="load": raise RuntimeError("private path must not escape")
        return FakeRuntime(error=RuntimeError("private content must not escape"))
    provider=LocalBGERerankProvider(model_dir(tmp_path),runtime_factory=factory)
    builder=ContextBuilderV2(parent_expander=ParentExpander(EmptyParents()),
        reranker=ConfigurableModelReranker(provider,max_attempts=1),rerank_enabled=True,
        parent_expansion_enabled=False)
    result=await builder.build("synthetic",[fused("A",1),fused("B",2)])
    assert [row.supporting_candidate_ids[0] for row in result.contexts]==["A","B"]
    assert result.diagnostics.rerank_used is False


@pytest.mark.asyncio
async def test_equal_scores_use_rrf_then_candidate_id_stable_tie_break(tmp_path):
    items=[fused("B",1),fused("A",1)]
    items[1]=replace(items[1],rrf_score=items[0].rrf_score)
    provider=LocalBGERerankProvider(model_dir(tmp_path),runtime_factory=lambda *_:FakeRuntime((1.0,1.0)))
    result=await ConfigurableModelReranker(provider,max_attempts=1).rerank("q",items,2)
    assert [row.candidate_id for row in result.candidates]==["A","B"]


def test_default_factory_stays_disabled_without_loading_model(monkeypatch):
    settings=Settings(_env_file=None)
    reranker=create_reranker(settings)
    assert type(reranker).__name__=="DisabledReranker"


def test_enabled_local_bge_requires_existing_valid_path(tmp_path):
    with pytest.raises(ValueError,match="existing local directory"):
        Settings(_env_file=None,rerank_enabled=True,rerank_provider="local_bge",rerank_model_path=str(tmp_path/"missing"))
    settings=Settings(_env_file=None,rerank_enabled=True,rerank_provider="local_bge",rerank_model_path=str(model_dir(tmp_path)))
    assert settings.rerank_device=="auto" and settings.rerank_batch_size==4
