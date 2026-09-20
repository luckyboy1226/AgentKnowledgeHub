"""Lazy runtime adapters for Phase G; construction performs zero I/O."""
from __future__ import annotations
import uuid
from typing import Any
from services.hybrid_retrieval_v2_real_evaluation import GRAPH_VARIANTS
from services.hybrid_retrieval_v2_adapter import HybridRetrievalV2ProductionAdapter

class FakePhaseGRuntime:
    """Pure fake adapter for CLI and lifecycle tests; no network dependencies."""
    async def ingest(self,d:dict[str,Any],run_id:str)->dict[str,Any]: return {"document_id":str(uuid.uuid5(uuid.NAMESPACE_URL,run_id+str(d['id']))),"version":1,"content_hash":"fake"}
    async def verify_document(self,_:str)->dict[str,Any]: return {k:True for k in ('ready','current','parents_ready','children_ready','vectors_current','graph_provenance_current')}
    async def verify_scope(self,_)->dict[str,Any]: return {k:True for k in ('mongo','chroma','neo4j','bm25')}
    async def build_query_plan(self,q:dict[str,Any],_:str)->dict[str,Any]: return {'queries':[q['question_id']],'entities':[],'keywords':[],'intent':'factoid'}
    async def evaluate(self,p:dict[str,Any],v:str,allow,*,trace:bool,graph_trace:bool)->dict[str,Any]:
        docs=sorted(allow); return {'final_context_ids':[f'{v}:{p["question_id"]}'],'document_ranks':docs[:8],'candidate_ranks':[1], 'latency':{'retrieval_total_ms':1.0},'graph_metrics':{'applicable':v in GRAPH_VARIANTS},'call_audit':{'vector_searches':1,'bm25_searches':int(v in {'bm25_vector_rrf','hybrid_v2_no_rerank'}),'graph_queries':int(v in GRAPH_VARIANTS),'reranker_calls':0}}

class RuntimeCallAudit:
    def __init__(self): self.query_plan_llm_calls=self.embedding_queries=self.vector_searches=self.bm25_searches=self.graph_queries=self.reranker_calls=0
    def snapshot(self): return self.__dict__.copy()
    def delta(self,before): return {k:getattr(self,k)-before[k] for k in before}

class HybridRetrievalV2RealRuntime:
    """Lazy composition point for Coordinator, BM25, HybridRetrieverV2, RRF and ContextBuilderV2.
    Factories are injected so constructor cannot allocate clients or invoke providers.
    """
    def __init__(self, *, coordinator_factory:Any, services_factory:Any, plan_factory:Any, production_adapter_factory:Any|None=None, embedding_snapshot:Any|None=None, run_id:str|None=None, embedding_identity:dict[str,Any]|None=None, query_plans_hash:str|None=None):
        self._coordinator_factory=coordinator_factory; self._services_factory=services_factory; self._plan_factory=plan_factory; self._production_adapter_factory=production_adapter_factory; self._embedding_snapshot=embedding_snapshot; self._run_id=run_id; self._embedding_identity=embedding_identity; self._query_plans_hash=query_plans_hash; self.audit=RuntimeCallAudit(); self.last_ingestion_operation=None
    async def ingest(self, document, run_id):
        operation_id=str(uuid.uuid4()); self.last_ingestion_operation={'logical_key':document.get('id'),'operation_id':operation_id}
        coordinator=self._coordinator_factory(); result=await coordinator.create_document_version(filename=document['filename'],content=document['content'].encode() if isinstance(document['content'],str) else document['content'],logical_key=document['id'],namespace='evaluation',operation_id=operation_id)
        return {'document_id':result['document_id'],'version':result.get('version'),'status':result.get('status'),'content_hash':result.get('content_hash'),'source':document['filename'],'operation_id':operation_id}
    async def verify_document(self,document_id):
        s=self._services_factory(); return await s.verify_document(document_id)
    async def verify_scope(self,allow):
        if not allow: raise ValueError('empty_scope')
        s=self._services_factory(); return await s.verify_scope(allow)
    async def verify_recovery_identity(self, *, logical_key, source, document_id, operation_id):
        s=self._services_factory()
        return await s.verify_recovery_identity(logical_key=logical_key, source=source, document_id=document_id, operation_id=operation_id)
    async def build_query_plan(self,question,run_id):
        builder=self._plan_factory(); self.audit.query_plan_llm_calls+=2; plan=await builder.build_evaluation_query_plan(question['question'],run_id=run_id,question_id=question['question_id'])
        return {'question_id':plan.question_id,'queries':list(plan.queries) or [plan.normalized_query],'entities':list(plan.entities),'keywords':list(plan.keywords),'intent':getattr(plan.intent,'value',plan.intent)}
    async def evaluate(self,plan,variant,allow,*,trace=False,graph_trace=False):
        if variant not in {'vector_only','bm25_vector_rrf','vector_graph_rrf','hybrid_v2_no_rerank'}: raise ValueError('invalid_variant')
        if graph_trace and variant not in GRAPH_VARIANTS: raise ValueError('graph_trace_not_applicable')
        before=self.audit.snapshot()
        query_embedding=self._embedding_snapshot.vector_for(run_id=self._run_id,plan=plan,query_plans_hash=self._query_plans_hash,embedding=self._embedding_identity) if self._embedding_snapshot is not None else None
        if self._production_adapter_factory is not None:
            result=await self._production_adapter_factory().run_variant(plan,variant,frozenset(allow),trace=trace,graph_trace=graph_trace,audit=self.audit,query_embedding=query_embedding)
        else:
            s=self._services_factory(); entities=plan.get('entities',[]) if variant in GRAPH_VARIANTS else []; bm25=variant in {'bm25_vector_rrf','hybrid_v2_no_rerank'}
            result=await s.run_variant(plan,variant,frozenset(allow),entities=entities,bm25_enabled=bm25,trace=trace,graph_trace=graph_trace,audit=self.audit)
        if trace and self._embedding_snapshot is not None:
            # An observer may correlate ON/OFF calls by the committed vector
            # hash, but must never receive vector values or query text.
            result.setdefault('trace_diagnostics', {})['query_embedding']={
                'vector_sha256':self._embedding_snapshot.vector_hash_for(run_id=self._run_id,plan=plan,query_plans_hash=self._query_plans_hash,embedding=self._embedding_identity),
            }
        return {**result,'call_audit':self.audit.delta(before)}


class RealRuntimeStorageVerifier:
    """Read-only, fail-closed provenance checks for a lifecycle-owned host.

    The verifier deliberately has no mutation APIs.  Coordinator writes are
    complete before it is used; this class only reads exact document UUIDs and
    versions recorded by the registry.
    """
    def __init__(self, components: dict[str, Any]):
        required = {"DocumentRegistry", "ChunkRepository", "VectorStoreService", "KnowledgeGraphService", "BM25Retriever"}
        missing = sorted(required.difference(components))
        if missing:
            raise RuntimeError("real_runtime_missing_lifecycle_dependencies:" + ",".join(missing))
        self.registry = components["DocumentRegistry"]
        self.chunks = components["ChunkRepository"]
        self.vectors = components["VectorStoreService"]
        self.graph = components["KnowledgeGraphService"]
        self.bm25 = components["BM25Retriever"]

    async def verify_document(self, document_id: str) -> dict[str, bool]:
        document = self.registry.find(document_id)
        version = int(document.get("current_version") or 0) if document else 0
        version_row = next((row for row in self.registry.versions_for(document_id)
                            if int(row.get("version") or 0) == version), None)
        ready = bool(document and document.get("status") == "ready" and version_row
                     and version_row.get("status") == "ready" and version_row.get("is_current") is True)
        catalog = list(self.chunks.collection.find(
            {"document_id": str(document_id), "document_version": version}, {"_id": 0}
        )) if version else []
        parents_ready = bool([row for row in catalog if row.get("kind") == "parent"] and
                             all(row.get("status") == "ready" and row.get("is_current") is True
                                 for row in catalog if row.get("kind") == "parent"))
        children_ready = bool([row for row in catalog if row.get("kind") == "child"] and
                              all(row.get("status") == "ready" and row.get("is_current") is True
                                  for row in catalog if row.get("kind") == "child"))
        vector_rows = self.vectors._version_records(str(document_id), version) if version else []
        vectors_current = bool(vector_rows) and all(
            str(metadata.get("document_id")) == str(document_id)
            and metadata.get("document_version") == version
            and metadata.get("status") == "ready" and metadata.get("is_current") is True
            for _, metadata in vector_rows
        )
        graph_rows = await self.graph.execute_cypher(
            """
            MATCH (dv:DocumentVersion {document_id: $document_id, document_version: $version})
            RETURN dv.status AS status, dv.is_current AS is_current,
              dv.content_hash AS content_hash
            """, {"document_id": str(document_id), "version": version}
        ) if version else []
        graph_provenance_current = bool(graph_rows) and all(
            row.get("status") == "ready" and row.get("is_current") is True
            and row.get("content_hash") == version_row.get("content_hash")
            for row in graph_rows
        )
        return {"ready": ready, "current": ready, "parents_ready": parents_ready,
                "children_ready": children_ready, "vectors_current": vectors_current,
                "graph_provenance_current": graph_provenance_current}

    async def verify_scope(self, allow: frozenset[str]) -> dict[str, bool]:
        if not allow:
            raise ValueError("empty_scope")
        ids = sorted(str(value) for value in allow)
        documents = list(self.registry.documents.find(
            {"document_id": {"$in": ids}}, {"_id": 0}
        ))
        mongo = len(documents) == len(ids) and all(
            row.get("document_id") in allow and row.get("status") == "ready"
            and isinstance(row.get("current_version"), int) for row in documents
        )
        chroma = mongo
        bm25 = mongo
        for row in documents:
            document_id, version = str(row["document_id"]), int(row["current_version"])
            vectors = self.vectors._version_records(document_id, version)
            children = self.chunks.list_current_children(document_id)
            chroma = chroma and bool(vectors) and all(
                str(meta.get("document_id")) in allow and meta.get("document_version") == version
                and meta.get("status") == "ready" and meta.get("is_current") is True
                for _, meta in vectors
            )
            bm25 = bm25 and bool(children) and all(
                str(child.get("document_id")) in allow and self.bm25._eligible(child)
                for child in children
            )
        graph_rows = await self.graph.list_scoped_evaluation_evidence(frozenset(ids))
        version_rows = await self.graph.execute_cypher(
            """
            MATCH (dv:DocumentVersion)
            WHERE dv.document_id IN $allowed_document_ids
              AND dv.is_current = true AND dv.status = 'ready'
            RETURN dv.document_id AS document_id, dv.document_version AS document_version
            """, {"allowed_document_ids": ids}
        )
        graph = len({str(row.get("document_id")) for row in version_rows}) == len(ids) and all(
            str(row.get("document_id")) in allow and row.get("status") == "ready"
            and row.get("is_current") is True for row in graph_rows
        )
        return {"mongo": bool(mongo), "chroma": bool(chroma), "neo4j": bool(graph), "bm25": bool(bm25)}

    async def verify_recovery_identity(self, *, logical_key: str, source: str, document_id: str,
                                       operation_id: str) -> dict[str, Any]:
        """Read only one known Coordinator operation and one known document UUID.

        It never searches by source and never enumerates a namespace, so a
        recovery record cannot accidentally attach a document from another run.
        """
        operation = self.registry.operations.find_one({"operation_id": str(operation_id)}, {"_id": 0})
        document = self.registry.find(str(document_id))
        if not operation or not document:
            raise ValueError("recovery_identity_not_found")
        if (operation.get("operation_type") != "create"
                or operation.get("status") != "succeeded"
                or str(operation.get("document_id")) != str(document_id)
                or str(operation.get("logical_key")) != str(logical_key)
                or str(document.get("logical_key")) != str(logical_key)
                or str(operation.get("source")) != str(source)
                or str(document.get("filename")) != str(source)):
            raise ValueError("recovery_identity_mismatch")
        verified = await self.verify_document(str(document_id))
        if not all(verified.values()):
            raise ValueError("recovery_document_not_ready_current")
        version = int(operation.get("version") or 0)
        version_row = next((row for row in self.registry.versions_for(str(document_id))
                            if int(row.get("version") or 0) == version), None)
        if not version_row:
            raise ValueError("recovery_version_not_found")
        return {"logical_key": str(logical_key), "source": str(source), "document_id": str(document_id),
                "operation_id": str(operation_id), "document_version": version,
                "content_hash": str(version_row.get("content_hash") or ""),
                "status": "ready"}


def build_hosted_real_runtime(host: Any, *, trace_run_id: str = 'phase-g', embedding_snapshot:Any|None=None, query_plans_hash:str|None=None) -> HybridRetrievalV2RealRuntime:
    """Bind a real runtime only to dependencies opened by an authorized host."""
    components = host.components
    production = HybridRetrievalV2ProductionAdapter(
        hybrid_factory=lambda: components["HybridRetrieverV2"],
        fusion_factory=lambda: components["RRFFusion"],
        context_builder_factory=lambda: components["ContextBuilderV2"],
        trace_run_id=trace_run_id,
    )
    return HybridRetrievalV2RealRuntime(
        coordinator_factory=lambda: components["DocumentUpdateCoordinator"],
        services_factory=lambda: RealRuntimeStorageVerifier(components),
        plan_factory=lambda: (_ for _ in ()).throw(RuntimeError("query_plans_require_g3_authorization")),
        production_adapter_factory=lambda: production,
        embedding_snapshot=embedding_snapshot, run_id=trace_run_id,
        embedding_identity={"provider":host.settings.embedding_config.provider,"model":host.settings.embedding_config.model,"dimension":host.settings.embedding_dimensions,"embedding_space_id":host.settings.resolved_embedding_space_id},
        query_plans_hash=query_plans_hash,
    )


def build_query_planning_runtime(settings: Any) -> HybridRetrievalV2RealRuntime:
    """Inject only the application's chat-backed evaluation planner for G3.

    No storage host, embedding provider, retriever, graph, BM25, reranker, or
    context builder is constructed on this path.
    """
    from agents.qa_agent import QAAgent
    from providers.factory import create_chat_provider
    runtime = build_real_runtime(settings)
    planner = QAAgent(create_chat_provider(settings))
    runtime._plan_factory = lambda: planner
    runtime.construction_metadata["query_planner_type"] = type(planner).__name__
    runtime.construction_metadata["planning_io_performed"] = False
    return runtime

def build_real_runtime(settings: Any) -> HybridRetrievalV2RealRuntime:
    """Return the production runtime wiring without instantiating an I/O client.

    The deferred factories are deliberately unusable until a later, separately
    authorized G2 host supplies lifecycle-owned service factories.
    """
    def deferred(name: str):
        def factory(): raise RuntimeError(f"real_runtime_{name}_requires_authorized_g2_host")
        return factory
    production = HybridRetrievalV2ProductionAdapter(
        hybrid_factory=deferred("hybrid"), fusion_factory=deferred("rrf"),
        context_builder_factory=deferred("context_builder"),
    )
    runtime = HybridRetrievalV2RealRuntime(
        coordinator_factory=deferred("coordinator"), services_factory=deferred("services"),
        plan_factory=deferred("query_plan"), production_adapter_factory=lambda: production,
    )
    runtime.construction_metadata = {
        "runtime_type": type(runtime).__name__, "production_adapter_type": type(production).__name__,
        "mongo_endpoint": settings.mongodb_uri, "chroma_endpoint": f"{settings.chroma_host}:{settings.chroma_port}",
        "neo4j_endpoint": settings.neo4j_uri, "embedding_provider": settings.embedding_config.provider,
        "embedding_model": settings.embedding_config.model, "embedding_dimension": settings.embedding_dimensions,
        "embedding_space_id": settings.resolved_embedding_space_id, "io_performed": False,
    }
    return runtime
