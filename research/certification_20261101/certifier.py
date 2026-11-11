"""Exact fixed-topology flow certificates with scoped caches and primal reuse."""
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow,breadth_first_order

class Certifier:
 def __init__(self,d,cache=True):
  start=time.perf_counter();self.d=d;self.cache=cache
  self.ns=len(d['groups']);self.nd=len(d['edges']);self.sink=self.ns+self.nd+1
  self.base=np.r_[d['cap'],np.zeros(self.ns-len(d['cap']),dtype=np.int64)]
  rr=[0]*self.ns;cc=list(range(1,self.ns+1))
  for i,js in enumerate(d['edges']):rr.extend(j+1 for j in js);cc.extend([self.ns+i+1]*len(js))
  self.ec=len(rr)-self.ns;rr.extend(range(self.ns+1,self.ns+self.nd+1));cc.extend([self.sink]*self.nd)
  template=csr_matrix((np.arange(1,len(rr)+1,dtype=np.int64),(np.asarray(rr,np.int32),np.asarray(cc,np.int32))),shape=(self.sink+1,self.sink+1))
  self.order=template.data-1;self.indices=template.indices;self.indptr=template.indptr
  self.exact={};self.feasible={};self.stats=dict(topology_seconds=time.perf_counter()-start,requests=0,flows=0,exact_hits=0,primal_hits=0,graph_seconds=0.,flow_seconds=0.,cut_seconds=0.,cache_seconds=0.)
 def one(self,plan,row):
  self.stats['requests']+=1;t=time.perf_counter();pb=plan.tobytes();rb=row.tobytes();key=(pb,rb);capacity=self.base+plan
  if self.cache and key in self.exact:
   self.stats['exact_hits']+=1;self.stats['cache_seconds']+=time.perf_counter()-t;return self.exact[key]
  if self.cache and rb in self.feasible:
   js,used=self.feasible[rb]
   if np.all(capacity[js]>=used):
    self.stats['primal_hits']+=1;answer=(0,None);self.exact[key]=answer;self.stats['cache_seconds']+=time.perf_counter()-t;return answer
  self.stats['cache_seconds']+=time.perf_counter()-t;t=time.perf_counter();total=int(row.sum())
  vals=np.r_[capacity,np.full(self.ec,total+1,dtype=np.int64),row]
  graph=csr_matrix((vals[self.order],self.indices,self.indptr),shape=(self.sink+1,self.sink+1),copy=False)
  self.stats['graph_seconds']+=time.perf_counter()-t;t=time.perf_counter();flow=maximum_flow(graph,0,self.sink)
  self.stats['flow_seconds']+=time.perf_counter()-t;self.stats['flows']+=1;loss=total-int(flow.flow_value);w=None;t=time.perf_counter()
  if loss:
   residual=graph-flow.flow;residual.data=(residual.data>0).astype(np.int32);residual.eliminate_zeros()
   reachable=set(breadth_first_order(residual,0,directed=True,return_predecessors=False).tolist())
   S=[i for i in range(self.nd) if self.ns+i+1 not in reachable];N=sorted(set(j for i in S for j in self.d['edges'][i]))
   need=int(row[S].sum());cap=int(capacity[N].sum());assert need-cap==loss
   w=dict(demand_cells=S,supply_indices=N,demand=need,capacity=cap,deficit=loss,rhs=need-int(self.base[N].sum()))
  elif self.cache:
   source=flow.flow.getrow(0);mask=source.data>0;js=source.indices[mask]-1;used=source.data[mask].copy()
   assert np.all(js<self.ns);self.feasible[rb]=(js,used)
  self.stats['cut_seconds']+=time.perf_counter()-t;answer=(loss,w)
  if self.cache:self.exact[key]=answer
  return answer
 def evaluate(self,plan,dem):
  plan=np.asarray(plan,dtype=np.int64);dem=np.asarray(dem,dtype=np.int64);cuts={};losses=[];worst=None;witnesses=[]
  for t,row in enumerate(dem):
   loss,w=self.one(plan,row);losses.append(loss)
   if w:
    key=tuple(w['supply_indices']);cuts[key]=max(cuts.get(key,-10**18),w['rhs'])
    witness=dict(w,period=t);witnesses.append(witness)
    if worst is None or loss>worst['deficit']:worst=witness
  return cuts,dict(total_deficit=sum(losses),worst_period_deficit=max(losses,default=0),unmet_periods=sum(x>0 for x in losses),periods=len(losses)),worst,witnesses

def strengthen(d,witnesses,dem):
 """For each discovered demand set, lift its cut to all design periods."""
 base=np.r_[d['cap'],np.zeros(len(d['groups'])-len(d['cap']),dtype=np.int64)];out={}
 for w in witnesses:
  S=w['demand_cells'];N=tuple(w['supply_indices']);rhs=int(dem[:,S].sum(axis=1).max()-base[list(N)].sum())
  out[N]=max(out.get(N,-10**18),rhs)
 return out
