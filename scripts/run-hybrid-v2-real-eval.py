"""Phase G command boundary. Only prepare is intentionally self-contained."""
from __future__ import annotations
import argparse, subprocess, asyncio, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'python'))
from services.hybrid_retrieval_v2_real_evaluation import ControlledRealEvaluationRunner
from services.hybrid_retrieval_v2_runtime import FakePhaseGRuntime

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--run-id',required=True); p.add_argument('--step',required=True,choices=('prepare','ingest','verify-scope','build-query-plans','trace-equivalence','evaluate','report','show-cleanup-plan')); p.add_argument('--runtime',choices=('fake','real'),default='fake'); p.add_argument('--allow-real-io',action='store_true'); a=p.parse_args(argv)
 r=ControlledRealEvaluationRunner(ROOT/'.runtime'/'evaluation',a.run_id,ROOT/'benchmarks'/'enterprise_20docs_retrieval_v2',{'final_top_k':8,'rrf_k':60,'token_budget':6000})
 if a.step=='prepare':
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(); print(r.prepare(git_commit=commit,embedding={'provider':'qwen','model':'qwen3.7-text-embedding-flash','dimension':1024,'embedding_space_id':'qwen:qwen3.7-text-embedding-flash:1024'})); return 0
 if a.step=='show-cleanup-plan': print(r.cleanup_plan()); return 0
 if a.runtime=='real' and a.step not in {'prepare','show-cleanup-plan'} and not a.allow_real_io: p.error('--runtime real requires --allow-real-io')
 if a.runtime=='real': p.error('real runtime requires explicit production adapter construction and is disabled in G1.5')
 fake=FakePhaseGRuntime()
 async def step():
  if a.step=='ingest':
   docs=json.loads((ROOT/'benchmarks'/'enterprise_20docs_60q_expanded'/'benchmark_documents.json').read_text(encoding='utf-8'))['documents']; return await r.ingest(fake,docs)
  if a.step=='verify-scope': return await r.verify_scope(fake)
  if a.step=='build-query-plans': return await r.build_query_plans(fake)
  if a.step=='trace-equivalence': return await r.trace_equivalence(fake)
  if a.step=='evaluate': return await r.evaluate(fake)
  if a.step=='report': return r.report()
 print(asyncio.run(step())); return 0
if __name__=='__main__': raise SystemExit(main())
