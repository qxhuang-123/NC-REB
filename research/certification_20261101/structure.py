import numpy as np

def components(d):
 ns=len(d['groups']);nd=len(d['edges']);n=len(d['cap']);p=list(range(ns+nd))
 def find(a):
  while p[a]!=a:p[a]=p[p[a]];a=p[a]
  return a
 def join(a,b):p[find(a)]=find(b)
 for i,js in enumerate(d['edges']):
  for j in js:join(ns+i,j)
 for g,q in enumerate(d['quotas']):
  js=np.where(d['groups']==g)[0].tolist();mx=[max(x for x in ((0,2,4) if j<n else (0,4,8)) if x<=q) for j in js]
  if sum(mx)>q:
   active=[j for j,u in zip(js,mx) if u]
   for j in active[1:]:join(active[0],j)
 blocks={}
 for j in range(ns):blocks.setdefault(find(j),dict(sites=[],cells=[]))['sites'].append(j)
 for i in range(nd):blocks[find(ns+i)]['cells'].append(i)
 return sorted(blocks.values(),key=lambda b:(-len(b['sites']),b['sites'][0]))

def subset(d,block):
 sites=block['sites'];cells=block['cells'];mapping={j:k for k,j in enumerate(sites)};gs=sorted(set(int(d['groups'][j]) for j in sites));gm={g:i for i,g in enumerate(gs)}
 return dict(cap=np.array([d['cap'][j] for j in sites if j<len(d['cap'])]),groups=np.array([gm[int(d['groups'][j])] for j in sites]),quotas=d['quotas'][gs],edges=[[mapping[j] for j in d['edges'][i]] for i in cells])

def audit(d):
 report=[]
 for b in components(d):
  support={j:[] for j in b['sites']}
  for i in b['cells']:
   for j in d['edges'][i]:support[j].append(i)
  supply_groups={}
  for j,ss in support.items():supply_groups.setdefault((int(d['groups'][j]),tuple(ss)),[]).append(j)
  demand_groups={}
  for i in b['cells']:demand_groups.setdefault(tuple(d['edges'][i]),[]).append(i)
  report.append(dict(projects=len(b['sites']),demand_cells=len(b['cells']),supply_response_groups=len(supply_groups),demand_response_groups=len(demand_groups),max_supply_group=max(map(len,supply_groups.values())),sites=b['sites'],cells=b['cells']))
 return report
