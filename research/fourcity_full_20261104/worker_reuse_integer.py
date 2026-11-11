import time
BOOT=time.perf_counter()
import sys,json,hashlib
from pathlib import Path
import numpy as np
import gurobipy as gp
HERE=Path(__file__).resolve().parent
R=HERE.parent
sys.path.insert(0,str(R/'certification_20261101'))
import worker as old
from structure import components,subset
from certifier import Certifier
sys.path.insert(0,str(R/'new2_20261103'))
import lazy_seeded_integer as lazy_reuse
import new_local
from reduced import ReducedCertifier
sys.path.insert(0,str(R/'laminar_20261103'))
from laminar import LaminarTree
from oracle import LocalOracle,evaluate_original_once

def tree_for(d):
 ss=[[] for j in d['groups']]
 for i,js in enumerate(d['edges']):
  for j in js:ss[j].append(i)
 return LaminarTree(ss,len(d['edges']))

class TreeCert(ReducedCertifier):
 def __init__(self,d,cache=True):
  super().__init__(d,cache);self.tree=tree_for(d)
 def one(self,plan,row):
  if not self.tree.laminar:return super().one(plan,row)
  self.stats['requests']+=1;t=time.perf_counter();key=(plan.tobytes(),row.tobytes())
  if self.cache and key in self.exact:
   self.stats['exact_hits']+=1;return self.exact[key]
  cap=self.base+plan;c=self.tree.capacities(cap);loss,S=self.tree.evaluate(c,row);w=None
  self.stats['formula_calls']+=1
  if loss:
   N=sorted({j for i in S for j in self.d['edges'][i]});need=int(row[S].sum());capacity=int(cap[N].sum());assert need-capacity==loss
   w=dict(demand_cells=S,supply_indices=N,demand=need,capacity=capacity,deficit=loss,rhs=need-int(self.base[N].sum()))
  ans=(loss,w)
  if self.cache:self.exact[key]=ans
  self.stats['aggregation_seconds']+=time.perf_counter()-t
  return ans

def flow_solve(d,dem,deadline,hierarchy,checkpoint,oracle=None,seed=None):
 tick=time.perf_counter();ns=len(d['groups']);n=len(d['cap']);base=np.r_[d['cap'],np.zeros(ns-n,dtype=int)]
 m=gp.Model();m.Params.OutputFlag=0;m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=0;m.Params.MIPGapAbs=0
 opts=[[(0,0),(2,5),(4,8)] if j<n else [(0,0),(4,13),(8,23)] for j in range(ns)]
 x=[[m.addVar(vtype=gp.GRB.BINARY,obj=c) for a,c in op] for op in opts]
 gain=[]
 for op,vs in zip(opts,x):m.addConstr(gp.quicksum(vs)==1);gain.append(gp.quicksum(a*v for (a,c),v in zip(op,vs)))
 for g,q in enumerate(d['quotas']):m.addConstr(gp.quicksum(gain[j] for j in np.flatnonzero(d['groups']==g))<=int(q))
 for N,rhs in (seed or {}).items():m.addConstr(gp.quicksum(gain[j] for j in N)>=rhs)
 parts=oracle.parts if oracle is not None else [dict(part=b) for b in components(dict(d,quotas=np.full(len(d['quotas']),10**12,dtype=np.int64)))]
 trees=0
 for entry in parts:
  b=entry['part']
  if not b['cells']:continue
  sub=entry['sub'] if oracle is not None else subset(d,b);tree=entry['tree'] if oracle is not None else tree_for(sub)
  local_rows=entry['index'].maximal if oracle is not None else dem[:,b['cells']]
  if hierarchy and tree.laminar:
   trees+=1;cv=[gp.LinExpr() for v in tree.nodes]
   for j,v in enumerate(tree.assignment):
    if v>=0:cv[v]+=int(base[b['sites'][j]])+gain[b['sites'][j]]
   for row in local_rows:
    residual=[m.addVar(lb=0) for v in tree.nodes]
    for v in range(tree.m):m.addConstr(residual[v]>=int(row[v])-cv[v])
    for v in range(tree.m,len(tree.nodes)):m.addConstr(residual[v]>=gp.quicksum(residual[u] for u in tree.children[v])-cv[v])
    m.addConstr(residual[tree.root]==0)
  else:
   for row in local_rows:
    inc={j:[] for j in b['sites']}
    for loc,i in enumerate(b['cells']):
     if row[loc]==0:continue
     fs=[]
     for j in d['edges'][i]:
      f=m.addVar();fs.append(f);inc[j].append(f)
     m.addConstr(gp.quicksum(fs)>=int(row[loc]))
    for j,fs in inc.items():
     if fs:m.addConstr(gp.quicksum(fs)<=int(base[j])+gain[j])
 build=time.perf_counter()-tick;m.update();nv,nc,nz=m.NumVars,m.NumConstrs,m.NumNZs
 last=[0.]
 def monitor(model,where):
  if where==gp.GRB.Callback.MIP and time.perf_counter()-last[0]>5:
   checkpoint(dict(phase='integer_search',variables=nv,constraints=nc,nonzeros=nz,bound=float(model.cbGet(gp.GRB.Callback.MIP_OBJBND)),incumbent=float(model.cbGet(gp.GRB.Callback.MIP_OBJBST)),nodes=float(model.cbGet(gp.GRB.Callback.MIP_NODCNT))))
   last[0]=time.perf_counter()
 m.Params.TimeLimit=max(.01,deadline-time.perf_counter());tick=time.perf_counter();m.optimize(monitor);opt=time.perf_counter()-tick
 p=None
 if m.SolCount:p=np.array([op[max(range(3),key=lambda k:x[j][k].X)][0] for j,op in enumerate(opts)],dtype=np.int64)
 result=dict(status=int(m.Status),certified_optimal=m.Status==2,infeasible=m.Status==3,objective=float(m.ObjVal) if m.SolCount else None,bound=float(m.ObjBound),times=dict(build=build,optimize=opt),variables=nv,constraints=nc,nonzeros=nz,tree_components=trees,pool_size=0,node_count=float(m.NodeCount))
 m.dispose();return result,p

def main(args):
 city,method,stage,path=args;path=Path(path);deadline=BOOT+3590
 advanced=method in ('B1','B2','B3','H3');prepared=method in ('B2','B3','H3');final_reuse=method in ('B3','H3')
 old.v.v1.SOURCE=R/('history_20261023/outputs/shenzhen_shared_capital_observed_case.npz' if city=='shenzhen' else 'fourcity_pilot_20261101/inputs/'+city+'_case.npz')
 work=path.with_suffix('');work.mkdir(exist_ok=True);old.v.H=work;old.v.v1.H=work
 tick=time.perf_counter();d,dem,hold,stages=old.v.prepare();preparation=time.perf_counter()-tick
 train=dem['local'];sel=list(stages['hourly_union']);order=list(sel)
 # Farthest-first Euclidean spatial shape+level distance, with fixed per-region scaling.
 scaled=train/np.maximum(1,train.max(axis=0))[None,:]
 dist=np.full(len(train),np.inf)
 for k in sel:dist=np.minimum(dist,((scaled-scaled[k])**2).sum(axis=1))
 dist[sel]=-1
 while len(order)<len(train):
  k=int(np.argmax(dist));order.append(k);dist=np.minimum(dist,((scaled-scaled[k])**2).sum(axis=1));dist[order]=-1
 count=len(train) if stage=='all' else int(stage);ids=sorted(order[:count]);target=train[ids]
 manifest=dict(city=city,method=method,stage=stage,design_periods=len(ids),selected_training_indices=ids,training_periods=len(train),holdout_periods=len(hold['local']),source_sha256=hashlib.sha256(old.v.v1.SOURCE.read_bytes()).hexdigest(),selection='nested farthest-first L2 demand vectors normalized by fixed training cell maxima; starts historical 24; training only')
 path.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 tick=time.perf_counter();blocks=components(d);struct=time.perf_counter()-tick
 plan=np.zeros(len(d['groups']),dtype=np.int64);records=[];fees=0;complete=True;infeasible=False
 new_local.Certifier=TreeCert;lazy_reuse.Certifier=new_local.LocalCertifier
 for bi,b in enumerate(blocks):
  sub=subset(d,b);rows=target[:,b['cells']];tick=time.perf_counter()
  cert=LocalOracle(sub,cache=True,prepared_tree=prepared) if advanced else Certifier(sub,cache=False)
  zero=np.zeros(len(sub['groups']),dtype=np.int64)
  cuts,ev,_,zero_witness=cert.check(zero,rows,stop_first=True) if advanced else cert.evaluate(zero,rows)
  pre=time.perf_counter()-tick;seed=dict(cuts)
  pre_stats=dict(cert.stats)
  p=None
  def checkpoint(x):path.with_suffix('.checkpoint.json').write_text(json.dumps(dict(block=bi,current=x,elapsed=time.perf_counter()-BOOT)))
  if not cuts:r=dict(status=2,certified_optimal=True,objective=0.,bound=0.,zero_exit=True,times=dict(precheck=pre));p=zero
  else:
   maximum=np.r_[np.full(len(sub['cap']),4),np.full(len(sub['groups'])-len(sub['cap']),8)]
   tick=time.perf_counter();mc,me,mw,_=cert.check(maximum,rows,stop_first=True) if advanced else cert.evaluate(maximum,rows);pre+=time.perf_counter()-tick
   pre_stats=dict(cert.stats)
   if mc:
    r=dict(status=3,infeasible=True,certified_optimal=False,objective=None,bound=None,capacity_upper_certificate=mw,max_menu_access_ignored=True,times=dict(precheck=pre));infeasible=True
   elif method.startswith('B'):
    lazy_reuse.Certifier=(lambda dd,cache=True:cert) if advanced else new_local.LocalCertifier
    lazy_reuse.SEED_CUTS=seed if advanced else {}
    r,p,_,bc=lazy_reuse.solve(sub,rows,'B-cache',deadline,checkpoint);r['certifier_stats']=dict(bc.stats);r['times']['precheck']=pre
    r['callback_delta_stats']={k:v-pre_stats.get(k,0) for k,v in bc.stats.items() if isinstance(v,(int,float))} if advanced else dict(bc.stats)
   else:
    r,p=flow_solve(sub,rows,deadline,method in ('H3','H-MILP'),checkpoint,oracle=cert if advanced else None,seed=seed if advanced else None);r['times']['precheck']=pre
  r.update(block=bi,projects=len(b['sites']),cells=len(b['cells']),precheck_certifier_stats=pre_stats)
  if r.get('capacity_upper_certificate'):r['certificate_global_cells']=[b['cells'][i] for i in r['capacity_upper_certificate']['demand_cells']];r['certificate_training_period']=ids[r['capacity_upper_certificate']['period']]
  records.append(r)
  if p is None or not r.get('certified_optimal'):
   complete=False;infeasible=bool(r.get('infeasible',False));break
  local_exact=sum(({0:0,2:5,4:8} if j<len(sub['cap']) else {0:0,4:13,8:23})[int(a)] for j,a in enumerate(p))
  r['recovered_exact_cost']=local_exact;r['solver_objective_difference']=float(r['objective'])-local_exact
  # An integral feasible objective C is optimal when a valid LB exceeds C-1.
  # The final original-graph check still validates the restored engineering plan.
  assert abs(r['solver_objective_difference'])<0.5,(r['objective'],local_exact)
  lower=float(r['bound'])
  assert local_exact-1+1e-5<lower<=local_exact+1e-5,(lower,local_exact)
  r['integer_objective_closure']=dict(cost=local_exact,reported_bound=lower,
      unit=1,bound_guard=1e-5,closed=True)
  plan[b['sites']]=p;fees+=local_exact
  path.with_suffix('.progress.json').write_text(json.dumps(dict(completed=len(records),total=len(blocks),elapsed=time.perf_counter()-BOOT)))
 evaluation={};evaltime={};witnesses={};final_stats={};evaluation_reuse={}
 if complete:
  assert all(plan[d['groups']==g].sum()<=q for g,q in enumerate(d['quotas']))
  cost=sum(({0:0,2:5,4:8} if j<len(d['cap']) else {0:0,4:13,8:23})[int(a)] for j,a in enumerate(plan));assert cost==fees
  if final_reuse:
   evaluation,witnesses,final_stats,evaltime,evaluation_reuse=evaluate_original_once(d,plan,train,hold['local'],ids,Certifier)
  else:
   final=Certifier(d,cache=False)
   for name,rows in [('design',target),('training',train),('holdout',hold['local'])]:
    tick=time.perf_counter();_,evaluation[name],witnesses[name],_=final.evaluate(plan,rows);evaltime[name]=time.perf_counter()-tick
   final_stats=final.stats
  assert evaluation['design']['total_deficit']==0
  np.save(path.with_suffix('.npy'),plan)
 result=dict(**manifest,certified_optimal=complete,infeasible=infeasible,objective=fees if complete else None,blocks=records,evaluation=evaluation,evaluation_seconds=evaltime,witnesses=witnesses,final_certifier_stats=final_stats,evaluation_reuse=evaluation_reuse,preparation_seconds=preparation,structure_seconds=struct,internal_total_seconds=time.perf_counter()-BOOT)
 path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(city,method,stage,complete,infeasible,result['objective'],result['internal_total_seconds'],flush=True)
