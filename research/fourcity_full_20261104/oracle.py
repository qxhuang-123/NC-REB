"""Exact componentwise contract quotient; feasibility certificates, not deficit estimates."""
import time
import numpy as np
from structure import components,subset
from reduced import ReducedCertifier
from laminar import LaminarTree


class ContractIndex:
    def __init__(self, rows):
        self.rows=np.asarray(rows,dtype=np.int64)
        unique,first,inverse=np.unique(self.rows,axis=0,return_index=True,return_inverse=True)
        keep=[];cover=np.full(len(unique),-1,dtype=int)
        for k in np.argsort(-unique.sum(axis=1),kind='stable'):
            hit=np.flatnonzero(np.all(unique[keep]>=unique[k],axis=1)) if keep else []
            if len(hit):cover[k]=keep[int(hit[0])]
            else:keep.append(int(k));cover[k]=k
        self.maximal=unique[keep];self.periods=first[keep]
        pos={k:i for i,k in enumerate(keep)}
        self.cover=np.array([pos[int(cover[k])] for k in inverse],dtype=int)
        self.unique_count=len(unique)
        assert np.all(self.maximal[self.cover]>=self.rows)


class LocalOracle:
    def __init__(self,d,cache=True,prepared_tree=True):
        tick=time.perf_counter();self.d=d;self.cache=cache;self.prepared_tree=prepared_tree
        self.base=np.r_[d['cap'],np.zeros(len(d['groups'])-len(d['cap']),dtype=np.int64)]
        self.parts=[]
        for p in components(dict(d,quotas=np.full(len(d['quotas']),10**12,dtype=np.int64))):
            if not p['cells']:continue
            sub=subset(d,p);ss=[[] for _ in sub['groups']]
            for i,js in enumerate(sub['edges']):
                for j in js:ss[j].append(i)
            tree=LaminarTree(ss,len(sub['edges']))
            self.parts.append(dict(part=p,sub=sub,tree=tree,flow=None if tree.laminar else ReducedCertifier(sub,cache=cache),profiles={}))
        self.dem=None
        self.stats=dict(topology_seconds=time.perf_counter()-tick,contract_seconds=0.,requests=0,profile_hits=0,profile_misses=0,flows=0,formula_calls=0,capacity_aggregations=0,original_local_rows=0,unique_local_rows=0,maximal_local_rows=0,certificate_seconds=0.,early_failures=0)

    def bind(self,dem):
        if dem is self.dem:return
        tick=time.perf_counter();self.dem=dem
        for e in self.parts:
            e['index']=ContractIndex(dem[:,e['part']['cells']]);e['profiles'].clear()
        self.stats['contract_seconds']+=time.perf_counter()-tick
        self.stats['original_local_rows']=len(dem)*len(self.parts)
        self.stats['unique_local_rows']=sum(e['index'].unique_count for e in self.parts)
        self.stats['maximal_local_rows']=sum(len(e['index'].maximal) for e in self.parts)

    def check(self,plan,dem,stop_first=False):
        self.bind(dem);tick=time.perf_counter();self.stats['requests']+=1
        cap=self.base+plan;cuts={};witnesses=[]
        for e in self.parts:
            p=e['part'];tree=e['tree'];ix=e['index'];pp=plan[p['sites']]
            c=None
            if tree.laminar:
                c=tree.capacities(cap[p['sites']]);self.stats['capacity_aggregations']+=1;key=tuple(c)
            else:key=pp.tobytes()
            profile=e['profiles'].get(key) if self.cache else None
            if profile is None:
                self.stats['profile_misses']+=1;profile={}
                if self.cache:e['profiles'][key]=profile
            else:self.stats['profile_hits']+=1
            for k,row in enumerate(ix.maximal):
                ans=profile.get(k)
                if ans is None:
                    if tree.laminar:
                        if not self.prepared_tree:
                            c=tree.capacities(cap[p['sites']]);self.stats['capacity_aggregations']+=1
                        loss,S=tree.evaluate(c,row);self.stats['formula_calls']+=1
                    else:
                        loss,w=e['flow'].one(pp,row);S=w['demand_cells'] if w else []
                    ans=(int(loss),S);profile[k]=ans
                loss,S=ans
                if not loss:continue
                global_S=[p['cells'][i] for i in S];N=sorted({j for i in global_S for j in self.d['edges'][i]})
                t=int(ix.periods[k]);need=int(dem[t,global_S].sum());available=int(cap[N].sum());assert need-available==loss
                w=dict(demand_cells=global_S,supply_indices=N,demand=need,capacity=available,deficit=loss,rhs=need-int(self.base[N].sum()),period=t)
                cuts[tuple(N)]=max(cuts.get(tuple(N),-10**18),w['rhs']);witnesses.append(w)
                if stop_first:self.stats['early_failures']+=1;break
            if stop_first and cuts:break
        self.stats['flows']=sum(e['flow'].stats['flows'] for e in self.parts if e['flow'] is not None)
        self.stats['certificate_seconds']+=time.perf_counter()-tick
        # Omitted dominated rows need not have the same deficit; never report their total.
        metrics=dict(total_deficit=None,feasible=not cuts,kind='exact_feasibility_only',checked_violation_sum=sum(w['deficit'] for w in witnesses),periods=len(dem))
        worst=max(witnesses,key=lambda w:w['deficit']) if witnesses else None
        return cuts,metrics,worst,witnesses

    def evaluate(self,plan,dem):return self.check(plan,dem,stop_first=False)


def evaluate_original_once(d,plan,train,hold,design_ids,certifier_class):
    """Exact original-graph deficits; each distinct full demand row solved once."""
    tick=time.perf_counter();rows=np.vstack([train,hold]);unique,inverse=np.unique(rows,axis=0,return_inverse=True)
    cert=certifier_class(d,cache=False);answers=[cert.one(plan,row) for row in unique]
    groups={'design':np.asarray(design_ids,dtype=int),'training':np.arange(len(train)),'holdout':np.arange(len(train),len(rows))}
    ev={};witnesses={}
    for name,ids in groups.items():
        vals=[answers[int(inverse[i])] for i in ids];loss=[a[0] for a in vals]
        ev[name]=dict(total_deficit=int(sum(loss)),worst_period_deficit=int(max(loss,default=0)),unmet_periods=sum(v>0 for v in loss),periods=len(ids))
        witnesses[name]=None
        if any(loss):
            k=int(np.argmax(loss));witnesses[name]=dict(vals[k][1],period=k)
    return ev,witnesses,cert.stats,dict(all_original_evaluation=time.perf_counter()-tick),dict(unique_rows=len(unique),original_requests=len(train)+len(hold)+len(design_ids))
