"""Policy, bottleneck and nested-time extension. Scenario planning only."""
import importlib.util
import json
import pathlib
import sys
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
import gurobipy as gp

H=pathlib.Path(__file__).resolve().parent
P=H.parent/'capital_siting_20261031/experiment.py'
s=importlib.util.spec_from_file_location('v1',P); v1=importlib.util.module_from_spec(s); s.loader.exec_module(v1)
v1.H=H

def prepare():
    d=v1.load(); z=np.load(v1.SOURCE)
    dates=z['times'].astype('datetime64[D]'); days=np.unique(dates); split=days[int(len(days)*.7)]
    ix=np.where(dates<split)[0]; tr=z['demand'][ix].astype(int)
    hours=z['times'][ix].astype('datetime64[h]').astype(np.int64)%24
    first=[]
    for lo in (0,6,12,18):
        ids=np.where((hours>=lo)&(hours<lo+6))[0];first.append(int(ids[np.argmax(tr[ids].sum(axis=1))]))
    other=[]
    for h in range(24):
        ids=np.where(hours==h)[0]
        if len(ids):other.append(int(ids[np.argmax(tr[ids].sum(axis=1))]))
    medium=sorted(set(first+other)); stages={'4':sorted(first),'hourly_union':medium,'all':list(range(len(tr)))}
    dem={k:v1.contract(tr,k)[0] for k in ('zero','local','migration','concentration')}
    hold={k:v1.contract(d['hold'],k)[0] for k in dem}
    manifest=dict(date='2026-10-31',source_sha256=__import__('hashlib').sha256(v1.SOURCE.read_bytes()).hexdigest(),
       stages={k:len(v) for k,v in stages.items()},stage_indices=stages,source_training_indices=ix.tolist(),
       contracts='generate once on all 525 training timestamps; nested subsets; total increment equal across all-training contracts, not necessarily selected subsets',
       holdout='same frozen 233 timestamps, never added to optimization; stress transformed, not observed future demand',
       policies=['onsite_only','joint'],diagnostics='2x2 menu x2 / shared-access removed on frozen full-training scenarios',
       resources='functional trial 60s per optimization/300s outer B; no 12GB hard cap or cold repetitions; no acceleration claim',
       B='new exact MILP master + batched Hall separation; not the historical DP implementation',
       assumptions='same V1 normalized costs, grid-center hypothetical sites, 2km synthetic access groups, reported point equivalents')
    (H/'protocol.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return d,dem,hold,stages

def graph_data(d):
    n=len(d['cap']); ns=len(d['groups']); nd=len(d['edges']); sink=ns+nd+1
    rr=[0]*ns; cc=list(range(1,ns+1))
    for i,js in enumerate(d['edges']):
        rr.extend(j+1 for j in js);cc.extend([ns+i+1]*len(js))
    ec=len(rr)-ns;rr.extend(range(ns+1,ns+nd+1));cc.extend([sink]*nd)
    return ns,nd,sink,rr,cc,ec

def separate(d,plan,dem,certificate=False):
    ns,nd,sink,rr,cc,ec=graph_data(d);base=np.r_[d['cap'],np.zeros(nd,dtype=int)]
    capacities=base+plan; cuts={}; deficits=[]; witness=None
    for t,row in enumerate(dem):
        total=int(row.sum()); vals=np.r_[capacities,np.full(ec,total+1,dtype=np.int64),row]
        graph=csr_matrix((vals,(rr,cc)),shape=(sink+1,sink+1))
        flow=maximum_flow(graph,0,sink);loss=total-int(flow.flow_value);deficits.append(loss)
        if loss:
            residual=graph-flow.flow;residual.data=(residual.data>0).astype(np.int32);residual.eliminate_zeros()
            reachable=set(breadth_first_order(residual,0,directed=True,return_predecessors=False).tolist())
            S=[i for i in range(nd) if ns+i+1 not in reachable]
            N=tuple(sorted(set(j for i in S for j in d['edges'][i])))
            rhs=int(row[S].sum()-base[list(N)].sum())
            assert rhs-int(plan[list(N)].sum())==loss
            cuts[N]=max(cuts.get(N,0),rhs)
            if witness is None or loss>witness['deficit']:
                witness=dict(period=t,demand_cells=S,supply_indices=list(N),demand=int(row[S].sum()),
                             capacity=int(capacities[list(N)].sum()),deficit=loss,
                             test='sum demand over S exceeds capacity of all supporting sites N(S)')
    return cuts,dict(total_deficit=sum(deficits),worst_period_deficit=max(deficits,default=0),
        unmet_periods=sum(x>0 for x in deficits),periods=len(dem)),witness

def solve(d,dem,policy='joint',menu_scale=1,access=True,engine='flow'):
    start=time.perf_counter();n=len(d['cap']);ns=len(d['groups'])
    m=gp.Model();m.Params.OutputFlag=0;m.Params.Threads=1;m.Params.TimeLimit=60
    m.Params.MIPGap=0;m.Params.MIPGapAbs=0;m.Params.Seed=0
    gains=[np.array((0,2,4) if j<n else (0,4,8))*menu_scale for j in range(ns)]
    fees=[(0,5,8) if j<n else (0,13,23) for j in range(ns)]
    x={(j,k):m.addVar(vtype=gp.GRB.BINARY,obj=fees[j][k],ub=0 if policy=='onsite_only' and j>=n and k else 1) for j in range(ns) for k in range(3)}
    g=[gp.quicksum(int(gains[j][k])*x[j,k] for k in range(3)) for j in range(ns)]
    for j in range(ns):m.addConstr(gp.quicksum(x[j,k] for k in range(3))==1)
    if access:
        for a,q in enumerate(d['quotas']):m.addConstr(gp.quicksum(g[j] for j in np.where(d['groups']==a)[0])<=int(q))
    rows={};calls=0;cert_seconds=0;rounds=[];plan=None;cert=None;witness=None
    if engine=='flow':
        for row in dem:
            incoming=[[] for j in range(ns)]
            for i,need in enumerate(row):
                if need:
                    fs=[]
                    for j in d['edges'][i]:
                        f=m.addVar();fs.append(f);incoming[j].append(f)
                    m.addConstr(gp.quicksum(fs)>=int(need))
            for j,fl in enumerate(incoming):
                if fl:m.addConstr(gp.quicksum(fl)<=(int(d['cap'][j]) if j<n else 0)+g[j])
    else:
        m.addConstr(gp.quicksum(g)>=int(dem.sum(axis=1).max()-d['cap'].sum()))
    while True:
        m.Params.TimeLimit=max(.01,min(60,300-(time.perf_counter()-start)))
        m.optimize();calls+=1
        if not m.SolCount:break
        plan=np.array([int(round(g[j].getValue())) for j in range(ns)])
        tick=time.perf_counter();new,cert,witness=separate(d,plan,dem);cert_seconds+=time.perf_counter()-tick
        if not new:break
        if engine=='flow':raise AssertionError('Flow solution failed integer network certificate')
        for N,rhs in new.items():
            if N in rows:rows[N].RHS=max(rows[N].RHS,rhs)
            else:rows[N]=m.addConstr(gp.quicksum(g[j] for j in N)>=rhs)
        rounds.append(dict(calls=calls,new_cuts=len(new),pool=len(rows),deficit=cert['total_deficit']))
        if time.perf_counter()-start>=300:break
    status=int(m.Status)
    valid=plan is not None and cert is not None and cert['total_deficit']==0
    out=dict(status=status,certified_feasible=valid,certified_optimal=bool(valid and status==2),
        infeasible=bool(status==3),objective=float(m.ObjVal) if valid else None,
        master_bound=float(m.ObjBound),gap=float(m.MIPGap) if valid else None,
        calls=calls,cut_pool=len(rows),rounds=rounds,certification_seconds=cert_seconds,
        variables=m.NumVars,constraints=m.NumConstrs,seconds=time.perf_counter()-start,
        engine=engine,policy=policy,menu_scale=menu_scale,shared_access=access)
    out['termination_reason']=('certified' if valid else ('infeasible_master' if status==3 else ('outer_time_limit' if time.perf_counter()-start>=300 else 'optimizer_limit_or_no_certificate')))
    if valid:
        out.update(new_sites=int(np.count_nonzero(plan[n:])),expansion_sites=int(np.count_nonzero(plan[:n])),design_service=cert)
        assert not access or all(plan[d['groups']==a].sum()<=q for a,q in enumerate(d['quotas']))
    else:plan=None
    m.dispose();return out,plan

def run():
    d,dem,hold,stages=prepare();results=[]
    jobs=[]
    # Four-snapshot diagnostics use subsets of the frozen full-training contracts.
    for k in dem:
        for policy in ('onsite_only','joint'):jobs.append((k,'4',policy,1,True,'flow'))
    for k in ('migration','concentration'):
        for scale,access in ((1,True),(2,True),(1,False),(2,False)):
            jobs.append((k,'4','joint',scale,access,'flow'))
    for stage in ('hourly_union','all'):
        for k in ('zero','local'):
            for policy in ('onsite_only','joint'):jobs.append((k,stage,policy,1,True,'B'))
    for k in ('local','migration','concentration'):jobs.append((k,'4','joint',1,True,'B'))
    (H/'jobs.json').write_text(json.dumps(jobs),encoding='utf-8')
    for number,(k,stage,policy,scale,access,engine) in enumerate(jobs):
        target=dem[k][stages[stage]]
        result,plan=solve(d,target,policy,scale,access,engine)
        result.update(number=number,contract=k,stage=stage,design_periods=len(target))
        # A relaxed-access maximum-menu cut is a rigorous infeasibility certificate if violated.
        if plan is None:
            mx=np.r_[np.full(len(d['cap']),4*scale),np.full(len(d['edges']),8*scale if policy=='joint' else 0)]
            _,capcert,witness=separate(d,mx,target)
            result.update(max_menu_relaxation_service=capcert,spatial_infeasibility_witness=witness)
        else:
            _,result['heldout_service'],_=separate(d,plan,hold[k])
            np.save(H/('plan_'+str(number)+'.npy'),plan)
        results.append(result)
        (H/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(number,k,stage,policy,scale,access,engine,result['status'],result['objective'],flush=True)
    (H/'complete.json').write_text(json.dumps(dict(expected=len(jobs),completed=len(results))),encoding='utf-8')

if __name__=='__main__':run()
