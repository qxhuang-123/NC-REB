import time
BOOT=time.perf_counter()
import pathlib,sys,json,importlib.util
import numpy as np
import gurobipy as gp
from certifier import Certifier,strengthen
from structure import components,subset
H=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('frozen',H.parent/'capital_siting_v2_20261031/run.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)

def compress(d,dem):
 """Identical service response + access group; exact menu convolution with recovery."""
 n=len(d['cap']);ns=len(d['groups']);support=[[] for _ in range(ns)]
 for i,js in enumerate(d['edges']):
  for j in js:support[j].append(i)
 groups={}
 for j,ss in enumerate(support):groups.setdefault((int(d['groups'][j]),tuple(ss)),[]).append(j)
 bundles=[]
 for (g,ss),js in groups.items():
  profile={0:(0,())};q=int(d['quotas'][g])
  for j in js:
   opts=[(0,0),(2,5),(4,8)] if j<n else [(0,0),(4,13),(8,23)];new={}
   for a,(c,path) in profile.items():
    for b,f in opts:
     if a+b<=q and (a+b not in new or c+f<new[a+b][0]):new[a+b]=(c+f,path+(b,))
   profile=new
  bundles.append(dict(sites=js,group=g,support=ss,base=sum(int(d['cap'][j]) for j in js if j<n),options=[(a,c,path) for a,(c,path) in sorted(profile.items())]))
 # Aggregate demand nodes having identical bundle neighborhoods.
 edge=[[] for _ in d['edges']]
 for k,b in enumerate(bundles):
  for i in b['support']:edge[i].append(k)
 dg={}
 for i,js in enumerate(edge):dg.setdefault(tuple(js),[]).append(i)
 target=np.array([dem[:,ids].sum(axis=1) for ids in dg.values()]).T
 unique=np.unique(target,axis=0);keep=[]
 for row in unique:
  if not np.any(np.all(unique>=row,axis=1)&np.any(unique>row,axis=1)):keep.append(row)
 return bundles,list(dg.keys()),np.asarray(keep),dict(supply_groups=len(bundles),demand_groups=len(dg),periods=len(keep))

def solve(d,dem,method,deadline,checkpoint):
 times=dict(build=0.,optimize=0.,recovery=0.,certification=0.,preprocess=0.,old_cut_check=0.,cut_strengthening=0.)
 start=time.perf_counter();n=len(d['cap']);ns=len(d['groups']);is_b=method.startswith('B');fast=method!='flow-old' and method!='B-old';cache=method in ('flow-cache','flow-compressed','B-cache')
 certifier=Certifier(d,cache=cache) if fast else None;times['preprocess']+=time.perf_counter()-start
 def certify(p,rows):
  if fast:return certifier.evaluate(p,rows)
  cuts,metrics,w=v.separate(d,p,rows);return cuts,metrics,w,[]
 tick=time.perf_counter()
 if method=='flow-compressed':bundles,edges,target,compression=compress(d,dem)
 else:
  bundles=[]
  for j in range(ns):bundles.append(dict(sites=[j],group=int(d['groups'][j]),base=int(d['cap'][j]) if j<n else 0,options=[(a,c,(a,)) for a,c in ([(0,0),(2,5),(4,8)] if j<n else [(0,0),(4,13),(8,23)])]))
  edges=d['edges'];target=dem;compression=None
 times['preprocess']+=time.perf_counter()-tick;tick=time.perf_counter()
 m=gp.Model();m.Params.OutputFlag=0;m.Params.Threads=1;m.Params.MIPGap=0;m.Params.MIPGapAbs=0;m.Params.Seed=0
 x={};gain=[]
 for j,b in enumerate(bundles):
  for k,(a,c,path) in enumerate(b['options']):x[j,k]=m.addVar(vtype=gp.GRB.BINARY,obj=c)
  m.addConstr(gp.quicksum(x[j,k] for k in range(len(b['options'])))==1)
  gain.append(gp.quicksum(a*x[j,k] for k,(a,c,path) in enumerate(b['options'])))
 for g,q in enumerate(d['quotas']):m.addConstr(gp.quicksum(gain[j] for j,b in enumerate(bundles) if b['group']==g)<=int(q))
 constraints={};pool={}
 if is_b:m.addConstr(gp.quicksum(gain)>=int(dem.sum(axis=1).max()-d['cap'].sum()))
 else:
  for row in target:
   inc=[[] for b in bundles]
   for i,need in enumerate(row):
    if not need:continue
    fs=[]
    for j in edges[i]:f=m.addVar();fs.append(f);inc[j].append(f)
    m.addConstr(gp.quicksum(fs)>=int(need))
   for j,fl in enumerate(inc):
    if fl:m.addConstr(gp.quicksum(fl)<=bundles[j]['base']+gain[j])
 times['build']+=time.perf_counter()-tick;rounds=[];plan=None;valid=False;optimal=False;status=None;old_hits=0;incumbent=None
 while time.perf_counter()<deadline:
  m.Params.TimeLimit=max(.01,min(60,deadline-time.perf_counter()));tick=time.perf_counter();m.optimize();times['optimize']+=time.perf_counter()-tick;status=int(m.Status)
  if not m.SolCount:
   if status==3:break
   checkpoint(dict(status='master_no_incumbent',rounds=len(rounds),times=times));continue
  tick=time.perf_counter();plan=np.zeros(ns,dtype=np.int64)
  for j,b in enumerate(bundles):
   k=max(range(len(b['options'])),key=lambda k:x[j,k].X);plan[b['sites']]=b['options'][k][2]
  incumbent=float(m.ObjVal);times['recovery']+=time.perf_counter()-tick
  # Same-contract pool constraints are also in the master. Hits should normally be zero.
  tick=time.perf_counter();violated={N:rhs for N,rhs in pool.items() if int(plan[list(N)].sum())<rhs} if method=='B-cache' else {};times['old_cut_check']+=time.perf_counter()-tick
  if violated:old_hits+=1;new=violated;metrics=None
  else:
   tick=time.perf_counter();new,metrics,w,witnesses=certify(plan,dem);times['certification']+=time.perf_counter()-tick
   if method=='B-cache':
    tick=time.perf_counter();lift=strengthen(d,witnesses,dem)
    for N,rhs in lift.items():new[N]=max(new.get(N,-10**18),rhs)
    times['cut_strengthening']+=time.perf_counter()-tick
  valid=not new;optimal=valid and status==2
  if valid:
   if optimal:break
   checkpoint(dict(status='feasible_not_optimal',cost=incumbent,times=times));continue
  assert is_b,'explicit model failed service certification'
  tick=time.perf_counter();changed=0
  for N,rhs in new.items():
   if rhs<=pool.get(N,-10**18):continue
   pool[N]=rhs;changed+=1
   if N in constraints:constraints[N].RHS=rhs
   else:constraints[N]=m.addConstr(gp.quicksum(gain[j] for j in N)>=rhs)
  times['build']+=time.perf_counter()-tick
  assert changed,'old violated cuts already imposed; integer solution inconsistent'
  rounds.append(dict(cuts=len(pool),added=changed,cost=incumbent,deficit=metrics['total_deficit'] if metrics else None))
  checkpoint(dict(status='separating',rounds=rounds,times=times))
 out=dict(status=status,certified_optimal=optimal,certified_feasible=valid,infeasible=status==3,objective=incumbent if valid else None,bound=float(m.ObjBound),times=times,rounds=rounds,pool_size=len(pool),old_cut_priority_hits=old_hits,variables=m.NumVars,constraints=m.NumConstrs,compression=compression)
 m.dispose();return out,plan if valid else None,certify,certifier

def main(args):
 method,task,path=args;path=pathlib.Path(path);work=path.with_suffix('');work.mkdir(exist_ok=True);v.H=work;v.v1.H=work
 import_seconds=time.perf_counter()-BOOT;t=time.perf_counter();d,dem,hold,stages=v.prepare();prep=time.perf_counter()-t
 selection=None
 if task.startswith('hard'):
  rank=int(task[4:]);t=time.perf_counter();selection=components(d)[rank];d=subset(d,selection)
  dem={k:x[:,selection['cells']] for k,x in dem.items()};hold={k:x[:,selection['cells']] for k,x in hold.items()};prep+=time.perf_counter()-t
 target=dem['local'][stages['hourly_union']]
 def checkpoint(o):path.with_suffix('.checkpoint.json').write_text(json.dumps(o,indent=2),encoding='utf-8')
 result,p,certify,certifier=solve(d,target,method,BOOT+3590,checkpoint)
 result.update(method=method,task=task,import_seconds=import_seconds,prepare_seconds=prep,projects=len(d['groups']),design_periods=len(target),solver_version=gp.gurobi.version(),selection=selection)
 if p is not None:
  t=time.perf_counter();result['training']=certify(p,dem['local'])[1];result['training_seconds']=time.perf_counter()-t
  t=time.perf_counter();result['holdout']=certify(p,hold['local'])[1];result['holdout_seconds']=time.perf_counter()-t
  assert all(p[d['groups']==a].sum()<=q for a,q in enumerate(d['quotas']))
  np.save(path.with_suffix('.npy'),p)
 if certifier:result['certifier_stats']=certifier.stats
 result['internal_total_seconds']=time.perf_counter()-BOOT
 path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(method,task,result['certified_optimal'],result['objective'],result['internal_total_seconds'],flush=True)
if __name__=='__main__':main(sys.argv[1:])
