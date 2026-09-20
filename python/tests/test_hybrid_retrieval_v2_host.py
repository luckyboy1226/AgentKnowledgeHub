from services.hybrid_retrieval_v2_host import RealRuntimeHost
import pytest

class Mongo:
 def __init__(self): self.closed=False; self.admin=type('A',(),{'command':lambda _,x:{'ok':1}})()
 def close(self): self.closed=True
class Neo:
 def __init__(self): self.closed=False
 def execute_query(self,*_,**__): return type('R',(),{'records':[{'value':1}]})()
 def close(self): self.closed=True
class Settings: mongodb_uri='m'; neo4j_uri='n'; neo4j_user='u'; neo4j_password='p'; chroma_host='127.0.0.1'; chroma_port=8000

def test_host_closes_on_readiness_failure(monkeypatch):
 m=Mongo(); n=Neo(); h=RealRuntimeHost(Settings(),mongo_factory=lambda _:m,neo4j_factory=lambda *_:n,component_factory=lambda *_:{'x':object()})
 monkeypatch.setattr('urllib.request.urlopen',lambda *_,**__: (_ for _ in ()).throw(OSError('unavailable')))
 with pytest.raises(OSError, match='unavailable'):
  with h as open_host: open_host.validate_readiness()
 assert m.closed and n.closed and h.closed

def test_host_closes_partially_opened_clients_when_component_construction_fails():
 m=Mongo(); n=Neo()
 h=RealRuntimeHost(
  Settings(), mongo_factory=lambda _:m, neo4j_factory=lambda *_:n,
  component_factory=lambda *_: (_ for _ in ()).throw(RuntimeError('component failure')),
 )
 with pytest.raises(RuntimeError, match='component failure'):
  h.__enter__()
 assert m.closed and n.closed and h.closed
