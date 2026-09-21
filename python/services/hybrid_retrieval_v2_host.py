"""Read-only lifecycle host for Phase G runtime validation.

It never calls service ``init`` methods because those may create indexes or
collections.  Real ingestion remains a separately authorized operation.
"""
from __future__ import annotations
import urllib.request
from contextlib import AbstractContextManager
from typing import Any

class RealRuntimeHost(AbstractContextManager):
    def __init__(self, settings:Any, *, mongo_factory:Any, neo4j_factory:Any, component_factory:Any):
        self.settings=settings; self._mongo_factory=mongo_factory; self._neo4j_factory=neo4j_factory; self._component_factory=component_factory; self.mongo=None; self.neo=None; self.components={}; self.closed=False; self._ingestion_open=False
    def __enter__(self):
        try:
            self.mongo=self._mongo_factory(self.settings.mongodb_uri)
            self.neo=self._neo4j_factory(
                self.settings.neo4j_uri, self.settings.neo4j_user, self.settings.neo4j_password,
            )
            self.components=self._component_factory(self.settings,self.mongo)
            return self
        except BaseException:
            # ``with`` cannot call __exit__ when __enter__ itself fails.  Keep
            # partially opened read-only clients from leaking on this path.
            self.__exit__(None, None, None)
            raise
    def __exit__(self,*_):
        try:
            if self.neo: self.neo.close()
        finally:
            if self.mongo: self.mongo.close()
            self.closed=True
    def validate_readiness(self):
        mongo=bool(self.mongo.admin.command('ping').get('ok')==1)
        chroma=urllib.request.urlopen(f'http://{self.settings.chroma_host}:{self.settings.chroma_port}/api/v2/heartbeat',timeout=5).status==200
        neo=self.neo.execute_query('RETURN 1 AS value',database_='neo4j').records[0]['value']==1
        return {'mongo_ready':mongo,'chroma_ready':chroma,'neo4j_ready':neo,'dependencies':sorted(self.components),'io_performed':True,'write_operations':0}

    async def open_ingestion_dependencies(self):
        """Open existing production stores without creating schema or collections.

        This is intentionally separate from read-only validation.  It acquires
        an existing Chroma collection and an async Neo4j driver; all document
        mutations remain solely inside ``DocumentUpdateCoordinator``.
        """
        if not self.components:
            raise RuntimeError('host_not_open')
        vector=self.components['VectorStoreService']; graph=self.components['KnowledgeGraphService']
        import chromadb
        from neo4j import AsyncGraphDatabase
        client=chromadb.HttpClient(host=vector.chroma_http_host(self.settings.chroma_host),port=self.settings.chroma_port)
        vector._store=client.get_collection(name=vector.collection_name)
        vector._validate_collection_identity(getattr(vector._store,'metadata',None))
        graph._driver=AsyncGraphDatabase.driver(self.settings.neo4j_uri,auth=(self.settings.neo4j_user,self.settings.neo4j_password))
        await graph._driver.verify_connectivity()
        self._ingestion_open=True

    async def close_ingestion_dependencies(self):
        graph=self.components.get('KnowledgeGraphService')
        if graph is not None:
            await graph.close()
        self._ingestion_open=False

def build_real_runtime_host(settings:Any)->RealRuntimeHost:
    def mongo(uri):
        from pymongo import MongoClient
        return MongoClient(uri,serverSelectionTimeoutMS=5000)
    def neo(uri,user,password):
        from neo4j import GraphDatabase
        driver=GraphDatabase.driver(uri,auth=(user,password),connection_timeout=5); driver.verify_connectivity(); return driver
    def components(s,mongo_client):
        from api.main import build_document_coordinator
        from services.document_registry import DocumentRegistry
        from services.chunk_repository import ChunkRepository
        from services.vector_store import VectorStoreService
        from services.knowledge_graph import KnowledgeGraphService
        from retrieval.bm25_retriever import BM25Retriever
        from retrieval.hybrid_retriever import HybridRetrieverV2
        from retrieval.fusion import RRFFusion
        from retrieval.parent_expander import ParentExpander
        from retrieval.context_builder import ContextBuilderV2
        from retrieval.reranker_factory import create_reranker
        from providers.factory import create_chat_provider, create_embedding_provider
        db=mongo_client[s.mongodb_database]; registry=DocumentRegistry(db); chunks=ChunkRepository(db); embedding=create_embedding_provider(s); vector=VectorStoreService(embedding); graph=KnowledgeGraphService(); bm25=BM25Retriever(chunks,max_indexed_children=s.bm25_max_indexed_children); coordinator=build_document_coordinator(registry,vector,graph,create_chat_provider(s),temp_root=s.upload_dir,chunks=chunks,derived_index=bm25); hybrid=HybridRetrieverV2(bm25=bm25,vector_store=vector,knowledge_graph=graph); fusion=RRFFusion(s.rrf_k,s.rrf_fusion_top_k); parent=ParentExpander(chunks); builder=ContextBuilderV2(parent_expander=parent,reranker=create_reranker(s),rerank_enabled=s.rerank_enabled,parent_expansion_enabled=True,rerank_input_top_k=s.rerank_input_top_k,rerank_output_top_k=s.rerank_output_top_k,final_context_top_k=s.final_context_top_k,final_context_token_budget=s.final_context_token_budget)
        return {'DocumentRegistry':registry,'DocumentUpdateCoordinator':coordinator,'ChunkRepository':chunks,'VectorStoreService':vector,'KnowledgeGraphService':graph,'BM25Retriever':bm25,'HybridRetrieverV2':hybrid,'RRFFusion':fusion,'ParentExpander':parent,'ContextBuilderV2':builder}
    return RealRuntimeHost(settings,mongo_factory=mongo,neo4j_factory=neo,component_factory=components)
