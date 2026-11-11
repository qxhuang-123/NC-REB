"""Frozen exploratory city planning trial; not a performance benchmark."""
import hashlib
import json
import math
import pathlib
import time
import sys
import numpy as np
from scipy.spatial import cKDTree
import gurobipy as gp
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

H = pathlib.Path(__file__).resolve().parent
SOURCE = H.parents[1] / 'research/history_20261023/outputs/shenzhen_shared_capital_observed_case.npz'

def allocate(total, weights):
    weights = np.asarray(weights, float)
    raw = total * weights / weights.sum()
    out = np.floor(raw).astype(int)
    order = np.argsort(-(raw-out), kind='stable')
    out[order[:total-int(out.sum())]] += 1
    assert out.sum() == total
    return out

def contract(base, kind):
    if kind == 'zero':
        return base.copy(), 0
    amount = int(round(.20 * base.sum()))
    weights = base.astype(float)
    if kind == 'migration':
        # Controlled spatial redistribution, not a forecast or observed migration.
        weights = np.roll(weights, weights.shape[1]//2, axis=1)
    elif kind == 'concentration':
        weights = np.zeros_like(weights)
        # Fixed top-quarter concentration rule for both design and heldout sets.
        peaks=np.argsort(-base.sum(axis=1),kind='stable')[:max(1,int(math.ceil(len(base)/4))) ]
        weights[peaks] = base[peaks]
    return base + allocate(amount, weights.ravel()).reshape(base.shape), amount

def load():
    z = np.load(SOURCE, allow_pickle=False)
    dates = z['times'].astype('datetime64[D]')
    unique = np.unique(dates)
    split = unique[int(len(unique)*.7)]
    train = dates < split
    hold = ~train
    # Four observed simultaneous snapshots chosen using training data only.
    hour = (z['times'].astype('datetime64[h]').astype(np.int64) % 24)
    selected = []
    for lo in (0,6,12,18):
        ix = np.where(train & (hour>=lo) & (hour<lo+6))[0]
        selected.append(int(ix[np.argmax(z['demand'][ix].sum(axis=1))]))
    cap = z['baseline_capacity'][train].max(axis=0).astype(int)
    # Verify the stated no-investment witness on the chosen design snapshots.
    assert np.all(cap[None,:] >= z['busy'][selected])
    xy = z['coordinates_m']
    cells = (z['cell_keys']+.5)*500
    edges = [sorted(a) for a in cKDTree(np.vstack([xy,cells])).query_ball_point(cells,800)]
    owner = z['owner_cell']
    assert all(int(j) in edges[int(i)] for j,i in enumerate(owner))
    allxy = np.vstack([xy,cells])
    groups = np.floor(allxy/2000).astype(int)
    keys = sorted(set(map(tuple,groups)))
    gid = {k:i for i,k in enumerate(keys)}
    groups = np.array([gid[tuple(k)] for k in groups])
    quotas = np.array([max(8,int(math.ceil(.5*cap[groups[:len(cap)]==i].sum()))) for i in range(len(keys))])
    data = dict(cap=cap, edges=edges, groups=groups, quotas=quotas,
                base=z['demand'][selected].astype(int), hold=z['demand'][hold].astype(int))
    manifest = dict(source=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        city='Shenzhen',existing_supply_records=len(cap),demand_cells=len(cells),candidate_new_sites=len(cells),
        candidate_rule='500m grid centers, hypothetical, no land or connection clearance',
        service_unit='reported simultaneous occupied-point equivalents; not kWh or identified latent demand',
        radius_m=800,service_edges=sum(map(len,edges)),training_dates=[str(unique[0]),str(split)],
        split_rule='first 70 percent distinct dates; split date belongs to holdout',
        train_timestamps=int(train.sum()),holdout_timestamps=int(hold.sum()),
        design_timestamps=[str(z['times'][i]) for i in selected],
        baseline_capacity='maximum reported capacity over training timestamps; static scenario envelope, not verified nameplate',
        existing_menu=[[0,0],[2,5],[4,8]],new_menu=[[0,0],[4,13],[8,23]],
        costs='normalized scenario units, not CNY',shared_access='synthetic 2km spatial groups; gain quota max(8,ceil(0.5*existing capacity))',
        modes=['full','pooled','periodwise','continuous'],contracts=['zero','local','migration','concentration'],
        growth=.2,trial_timelimit_per_solve=30,threads=1,mip_gap=0,
        concentration_rule='allocate added demand to top ceil(T/4) total-load snapshots, in both design and heldout sets; stress transformation, not a forecast',
        formal_resource_protocol='future benchmark: 3600s/12GB, cold repeats; this trial has no enforced memory cap and reports no speedup',
        B_status='not integrated: new shared-access and siting constraints require interface verification')
    (H/'design.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return data

def solve(d, demand, mode='full', floor=None):
    start=time.perf_counter(); n=len(d['cap']); ns=n+len(d['edges'])
    gains=[(0,2,4) if j<n else (0,4,8) for j in range(ns)]
    fees=[(0,5,8) if j<n else (0,13,23) for j in range(ns)]
    m=gp.Model(); m.Params.OutputFlag=0; m.Params.Threads=1
    m.Params.TimeLimit=30; m.Params.MIPGap=0; m.Params.MIPGapAbs=0; m.Params.Seed=0
    vtype=gp.GRB.CONTINUOUS if mode=='continuous' else gp.GRB.BINARY
    x={(j,k):m.addVar(lb=0,ub=1,vtype=vtype,obj=fees[j][k]) for j in range(ns) for k in range(3)}
    g=[gp.quicksum(gains[j][k]*x[j,k] for k in range(3)) for j in range(ns)]
    for j in range(ns):
        m.addConstr(gp.quicksum(x[j,k] for k in range(3))==1)
        if floor is not None: m.addConstr(g[j]>=floor[j])
    for a,q in enumerate(d['quotas']):
        m.addConstr(gp.quicksum(g[j] for j in np.where(d['groups']==a)[0])<=int(q))
    for t,row in enumerate(demand):
        if mode=='pooled':
            m.addConstr(int(d['cap'].sum())+gp.quicksum(g)>=int(row.sum()))
        else:
            incoming=[[] for _ in range(ns)]
            for i,need in enumerate(row):
                if not need: continue
                flow=[]
                for j in d['edges'][i]:
                    f=m.addVar(lb=0); flow.append(f); incoming[j].append(f)
                m.addConstr(gp.quicksum(flow)>=int(need))
            for j,fl in enumerate(incoming):
                if fl: m.addConstr(gp.quicksum(fl)<= (int(d['cap'][j]) if j<n else 0)+g[j])
    m.optimize()
    result=dict(status=int(m.Status),seconds=time.perf_counter()-start,variables=m.NumVars,constraints=m.NumConstrs,
                objective=float(m.ObjVal) if m.SolCount else None,bound=float(m.ObjBound) if m.IsMIP else None,
                gap=float(m.MIPGap) if m.IsMIP and m.SolCount else None)
    plan=None
    if m.SolCount:
        raw=np.array([g[j].getValue() for j in range(ns)])
        plan=np.array([next((v for v in gains[j] if v+1e-5>=raw[j]),gains[j][-1]) for j in range(ns)])
        result.update(plan_cost=sum(fees[j][gains[j].index(int(plan[j]))] for j in range(ns)),
            expansion_sites=int(np.count_nonzero(plan[:n])),new_sites=int(np.count_nonzero(plan[n:])),
            rounded_access_feasible=bool(all(plan[d['groups']==a].sum()<=q for a,q in enumerate(d['quotas']))))
    m.dispose(); return result,plan

def evaluate(d,plan,dem):
    n=len(d['cap']); cap=np.r_[d['cap'],np.zeros(len(d['edges']),int)]+plan
    ns=len(cap); nd=len(d['edges']); sink=ns+nd+1
    rr=list(np.zeros(ns,dtype=int)); cc=list(range(1,ns+1)); fixed=[]
    for i,js in enumerate(d['edges']):
        for j in js: rr.append(j+1); cc.append(ns+i+1); fixed.append(1)
    edge_count=len(fixed)
    rr.extend(range(ns+1,ns+nd+1)); cc.extend([sink]*nd)
    deficits=[]
    for row in dem:
        total=int(row.sum())
        values=np.r_[cap,np.full(edge_count,total,dtype=np.int64),row].astype(np.int64)
        graph=csr_matrix((values,(rr,cc)),shape=(sink+1,sink+1))
        deficits.append(total-int(maximum_flow(graph,0,sink).flow_value))
    return dict(total_deficit=int(sum(deficits)),worst_period_deficit=int(max(deficits,default=0)),
                unmet_periods=int(sum(v>0 for v in deficits)),periods=len(deficits))

def run():
    d=load(); results=[]
    for kind in ('zero','local','migration','concentration'):
        dem,amount=contract(d['base'],kind)
        # All held-out observed timestamps, independently transformed by the same scenario rule.
        test,_=contract(d['hold'],kind)
        for mode in ('full','pooled','periodwise','continuous'):
            start=time.perf_counter()
            if mode=='periodwise':
                sub=[solve(d,dem[t:t+1]) for t in range(len(dem))]
                plans=[p for r,p in sub]
                plan=np.max(plans,axis=0) if all(p is not None for p in plans) else None
                result=dict(subproblems=[r for r,p in sub],objective=None)
            else: result,plan=solve(d,dem,mode)
            result.update(contract=kind,mode=mode,added_design_demand=amount)
            if plan is not None:
                n=len(d['cap'])
                result.update(plan_cost=int(sum(({0:0,2:5,4:8} if j<n else {0:0,4:13,8:23})[int(v)] for j,v in enumerate(plan))),
                    expansion_sites=int(np.count_nonzero(plan[:n])),new_sites=int(np.count_nonzero(plan[n:])),
                    access_feasible=bool(all(plan[d['groups']==a].sum()<=q for a,q in enumerate(d['quotas']))),
                    design_service=evaluate(d,plan,dem),heldout_service=evaluate(d,plan,test))
                # Minimal completion with installed capacities retained; may be infeasible under access quotas.
                if mode!='full': result['repair'],_=solve(d,dem,floor=plan)
                np.save(H/(kind+'_'+mode+'_plan.npy'),plan)
            result['end_to_end_seconds']=time.perf_counter()-start
            results.append(result)
            (H/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            print(kind,mode,result.get('objective'),result.get('design_service'),flush=True)
    assert len(results)==16
    assert len(set(r['added_design_demand'] for r in results if r['contract']!='zero'))==1

if __name__=='__main__': run()
