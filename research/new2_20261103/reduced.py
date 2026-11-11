import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow,breadth_first_order
from certifier import Certifier

class ReducedCertifier(Certifier):
 def __init__(self,d,cache=True):
  tick=time.perf_counter();self.d=d;self.cache=cache;self.ns=len(d['groups']);self.nd=len(d['edges'])
  self.base=np.r_[d['cap'],np.zeros(self.ns-len(d['cap']),dtype=np.int64)]
  support=[[] for _ in range(self.ns)]
  for i,js in enumerate(d['edges']):
   for j in js:support[j].append(i)
  self.private=[[] for _ in range(self.nd)];self.hub=[];partial={}
  for j,ss in enumerate(support):
   if len(ss)==self.nd and ss:self.hub.append(j)
   elif len(ss)==1:self.private[ss[0]].append(j)
   elif ss:partial.setdefault(tuple(ss),[]).append(j)
  self.partial=list(partial.items());self.q=len(self.partial);self.sink=self.q+self.nd+1
  rr=[0]*self.q;cc=list(range(1,self.q+1))
  for k,(ss,js) in enumerate(self.partial):rr.extend([k+1]*len(ss));cc.extend(self.q+i+1 for i in ss)
  self.ec=len(rr)-self.q;rr.extend(range(self.q+1,self.q+self.nd+1));cc.extend([self.sink]*self.nd)
  template=csr_matrix((np.arange(1,len(rr)+1,dtype=np.int64),(rr,cc)),shape=(self.sink+1,self.sink+1))
  self.order=template.data-1;self.indices=template.indices;self.indptr=template.indptr
  self.exact={};self.feasible={};self.last=None
  self.stats=dict(topology_seconds=time.perf_counter()-tick,requests=0,flows=0,exact_hits=0,primal_hits=0,graph_seconds=0.,flow_seconds=0.,cut_seconds=0.,cache_seconds=0.,aggregation_seconds=0.,formula_calls=0,original_supply=self.ns,reduced_supply=self.q)
 def one(self,plan,row):
  tick=time.perf_counter();self.stats['requests']+=1;pb=plan.tobytes();rb=row.tobytes();key=(pb,rb)
  cap=self.base+plan
  if self.cache and key in self.exact:
   self.stats['exact_hits']+=1;self.stats['cache_seconds']+=time.perf_counter()-tick;return self.exact[key]
  if self.cache and rb in self.feasible:
   js,used=self.feasible[rb]
   if np.all(cap[js]>=used):
    answer=(0,None);self.exact[key]=answer;self.stats['primal_hits']+=1;self.stats['cache_seconds']+=time.perf_counter()-tick;return answer
  self.stats['cache_seconds']+=time.perf_counter()-tick;tick=time.perf_counter()
  if self.last is None or self.last[0]!=pb:
   self.last=(pb,np.array([cap[js].sum() for js in self.private],dtype=np.int64),int(cap[self.hub].sum()),np.array([cap[js].sum() for ss,js in self.partial],dtype=np.int64))
  _,priv,y,pcap=self.last;r=np.maximum(row-priv,0);total=int(r.sum());self.stats['aggregation_seconds']+=time.perf_counter()-tick
  flow=None
  if self.q and total:
   tick=time.perf_counter();vals=np.r_[pcap,np.full(self.ec,total+1,dtype=np.int64),r]
   graph=csr_matrix((vals[self.order],self.indices,self.indptr),shape=(self.sink+1,self.sink+1),copy=False);self.stats['graph_seconds']+=time.perf_counter()-tick
   tick=time.perf_counter();flow=maximum_flow(graph,0,self.sink);value=int(flow.flow_value);self.stats['flows']+=1;self.stats['flow_seconds']+=time.perf_counter()-tick
  else:value=0;self.stats['formula_calls']+=1
  loss=max(0,total-value-y);w=None;tick=time.perf_counter()
  if loss:
   if flow is None:S=np.flatnonzero(r>0).tolist()
   else:
    residual=graph-flow.flow;residual.data=(residual.data>0).astype(np.int32);residual.eliminate_zeros()
    reached=set(breadth_first_order(residual,0,directed=True,return_predecessors=False).tolist())
    S=[i for i in range(self.nd) if r[i]>0 and self.q+i+1 not in reached]
   N=sorted({j for i in S for j in self.d['edges'][i]});need=int(row[S].sum());capacity=int(cap[N].sum());assert need-capacity==loss
   w=dict(demand_cells=S,supply_indices=N,demand=need,capacity=capacity,deficit=loss,rhs=need-int(self.base[N].sum()))
  elif self.cache:
   used=np.zeros(self.ns,dtype=np.int64)
   def allocate(js,amount):
    for j in js:
     a=min(int(cap[j]),amount);used[j]=a;amount-=a
    assert amount==0
   for i,js in enumerate(self.private):allocate(js,int(min(row[i],priv[i])))
   if flow is not None:
    src=flow.flow.getrow(0).toarray().ravel()
    for k,(ss,js) in enumerate(self.partial):allocate(js,int(src[k+1]))
   allocate(self.hub,total-value)
   js=np.flatnonzero(used);self.feasible[rb]=(js,used[js].copy())
  self.stats['cut_seconds']+=time.perf_counter()-tick;answer=(loss,w)
  if self.cache:self.exact[key]=answer
  return answer
