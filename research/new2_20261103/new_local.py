import time
import numpy as np
from reduced import ReducedCertifier as Certifier
from structure import components,subset

class LocalCertifier:
 def __init__(self,d,cache=True):
  t=time.perf_counter();self.cache=cache;self.d=d;self.base=np.r_[d['cap'],np.zeros(len(d['groups'])-len(d['cap']),dtype=np.int64)]
  parts=components(dict(d,quotas=np.full(len(d['quotas']),10**12,dtype=np.int64)))
  self.parts=[(p,Certifier(subset(d,p),cache=cache),{}) for p in parts if p['cells']]
  self.stats=dict(topology_seconds=time.perf_counter()-t,requests=0,profile_hits=0,profile_misses=0,flows=0,cache_seconds=0.)
 def evaluate(self,plan,dem):
  plan=np.asarray(plan,dtype=np.int64);dem=np.asarray(dem,dtype=np.int64)
  losses=np.zeros(len(dem),dtype=np.int64);sets=[[] for _ in dem]
  for part,cert,profiles in self.parts:
   tick=time.perf_counter();pp=plan[part['sites']];rows=dem[:,part['cells']];key=(pp.tobytes(),rows.tobytes());self.stats['requests']+=1
   cached=profiles.get(key) if self.cache else None;self.stats['cache_seconds']+=time.perf_counter()-tick
   if cached is None:
    self.stats['profile_misses']+=1;cached=[]
    for row in rows:
     loss,w=cert.one(pp,row);cached.append((loss,[part['cells'][i] for i in w['demand_cells']] if w else []))
    if self.cache:profiles[key]=cached
   else:self.stats['profile_hits']+=1
   for t,(loss,S) in enumerate(cached):losses[t]+=loss;sets[t].extend(S)
  cuts={};witnesses=[];worst=None
  for t,S in enumerate(sets):
   if not losses[t]:continue
   N=sorted({j for i in S for j in self.d['edges'][i]});need=int(dem[t,S].sum());cap=int((self.base+plan)[N].sum());assert need-cap==losses[t]
   w=dict(demand_cells=S,supply_indices=N,demand=need,capacity=cap,deficit=int(losses[t]),rhs=need-int(self.base[N].sum()),period=t)
   key=tuple(N);cuts[key]=max(cuts.get(key,-10**18),w['rhs']);witnesses.append(w)
   if worst is None or w['deficit']>worst['deficit']:worst=w
  self.stats['flows']=sum(c.stats['flows'] for _,c,_ in self.parts)
  for key in ['aggregation_seconds','formula_calls','original_supply','reduced_supply','exact_hits','primal_hits','graph_seconds','flow_seconds','cut_seconds']:
   self.stats[key]=sum(c.stats[key] for _,c,_ in self.parts)
  return cuts,dict(total_deficit=int(losses.sum()),worst_period_deficit=int(losses.max()) if len(losses) else 0,unmet_periods=int((losses>0).sum()),periods=len(dem)),worst,witnesses
