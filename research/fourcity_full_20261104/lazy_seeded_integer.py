"""Single-search exact integer-candidate separation; unchanged project menus."""
import time, traceback
import numpy as np
import gurobipy as gp
from certifier import Certifier,strengthen
USE_LOCAL_CACHE=True
REUSE_FINAL=True
SEED_CUTS={}

def solve(d,dem,method,deadline,checkpoint):
 start=time.perf_counter();ns=len(d['groups']);n=len(d['cap'])
 cert=Certifier(d,cache=USE_LOCAL_CACHE);times=dict(build=0.,optimize=0.,recovery=0.,certification=0.,preprocess=time.perf_counter()-start,old_cut_check=0.,cut_strengthening=0.)
 tick=time.perf_counter();m=gp.Model();m.Params.OutputFlag=0;m.Params.Threads=1;m.Params.Seed=0
 m.Params.MIPGap=0;m.Params.MIPGapAbs=0;m.Params.LazyConstraints=1
 opts=[[(0,0),(2,5),(4,8)] if j<n else [(0,0),(4,13),(8,23)] for j in range(ns)]
 x=[[m.addVar(vtype=gp.GRB.BINARY,obj=c) for a,c in op] for op in opts]
 for vs in x:m.addConstr(gp.quicksum(vs)==1)
 gain=[gp.quicksum(a*v for (a,c),v in zip(op,vs)) for op,vs in zip(opts,x)]
 for g,q in enumerate(d['quotas']):m.addConstr(gp.quicksum(gain[j] for j in np.flatnonzero(d['groups']==g))<=int(q))
 m.addConstr(gp.quicksum(gain)>=int(dem.sum(axis=1).max()-d['cap'].sum()))
 for N,rhs in SEED_CUTS.items():m.addConstr(gp.quicksum(gain[j] for j in N)>=rhs)
 flat=[v for vs in x for v in vs];gains=np.array([[a for a,c in op] for op in opts],dtype=np.int64)
 times['build']=time.perf_counter()-tick
 accepted=set();pool=dict(SEED_CUTS);records=[];errors=[];stats=dict(candidates=0,old_cut_rejections=0,accepted_candidates=0,lazy_calls=0,callback_seconds=0.)
 def restore(values):return gains[np.arange(ns),np.argmax(np.asarray(values).reshape(ns,3),axis=1)]
 def callback(model,where):
  if where!=gp.GRB.Callback.MIPSOL:return
  begin=time.perf_counter()
  try:
   stats['candidates']+=1;t=time.perf_counter();plan=restore(model.cbGetSolution(flat));times['recovery']+=time.perf_counter()-t
   t=time.perf_counter();new={N:rhs for N,rhs in pool.items() if int(plan[list(N)].sum())<rhs};times['old_cut_check']+=time.perf_counter()-t
   deficit=None
   if new:stats['old_cut_rejections']+=1
   else:
    t=time.perf_counter();new,metrics,_,witnesses=cert.evaluate(plan,dem);times['certification']+=time.perf_counter()-t;deficit=metrics['total_deficit']
    t=time.perf_counter()
    for N,rhs in strengthen(d,witnesses,dem).items():new[N]=max(new.get(N,-10**18),rhs)
    times['cut_strengthening']+=time.perf_counter()-t
   t=time.perf_counter()
   for N,rhs in new.items():
    assert int(plan[list(N)].sum())<rhs
    # Re-add violated old lazy constraints too; the solver may revisit them.
    model.cbLazy(gp.quicksum(gain[j] for j in N)>=rhs);stats['lazy_calls']+=1
    pool[N]=max(pool.get(N,-10**18),rhs)
   times['build']+=time.perf_counter()-t
   if not new:stats['accepted_candidates']+=1;accepted.add(plan.tobytes())
   records.append(dict(candidate=stats['candidates'],elapsed=time.perf_counter()-start,cost=float(model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)),bound=float(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)),nodes=float(model.cbGet(gp.GRB.Callback.MIPSOL_NODCNT)),added=len(new),pool_size=len(pool),deficit=deficit))
   if stats['candidates']%25==0:checkpoint(dict(candidates=stats['candidates'],last=records[-1]))
  except Exception:
   errors.append(traceback.format_exc());model.terminate()
  finally:stats['callback_seconds']+=time.perf_counter()-begin
 m.Params.TimeLimit=max(.01,deadline-time.perf_counter());t=time.perf_counter();m.optimize(callback);wall=time.perf_counter()-t
 # Optimize wall includes all callback work; separate it to avoid double counting.
 times['optimize']=wall-stats['callback_seconds']
 if errors:m.dispose();raise RuntimeError('\n'.join(errors))
 plan=None;valid=False;objective=None
 if m.SolCount:
  t=time.perf_counter();plan=restore(m.getAttr('X',flat));times['recovery']+=time.perf_counter()-t
  t=time.perf_counter();valid=plan.tobytes() in accepted;assert valid,'final plan lacks complete callback certificate'
  if not REUSE_FINAL:
   valid=not cert.evaluate(plan,dem)[0]
   assert valid;times['certification']+=time.perf_counter()-t
  objective=float(m.ObjVal)
  assert abs(sum(dict(op)[int(a)] for op,a in zip(opts,plan))-objective)<0.5
 status=int(m.Status)
 out=dict(status=status,certified_optimal=valid and status==2,certified_feasible=valid,infeasible=status==3,objective=objective,bound=float(m.ObjBound),times=times,rounds=records,pool_size=len(pool),variables=m.NumVars,constraints=m.NumConstrs,compression=None,callback_stats=stats,optimizer_wall_including_callbacks=wall,node_count=float(m.NodeCount),optimize_calls=1)
 m.dispose();return out,plan,cert.evaluate,cert
