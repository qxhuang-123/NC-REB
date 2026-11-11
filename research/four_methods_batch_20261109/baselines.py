# coding: utf-8
"""Explicit-flow baselines for the frozen common-capital contracts."""
from pathlib import Path
import json,time,sys,hashlib,math
import numpy as np
import gurobipy as gp
H=Path(__file__).resolve().parent
R=H.parent
sys.path[:0]=[str(R/'fourcity_full_20261104'),str(R/'certification_20261101'),str(R/'laminar_20261103'),str(R/'new2_20261103')]
import worker_reuse_integer as w
from structure import components,subset
from oracle import LocalOracle
from certifier import Certifier
from progress_io import atomic_write

def atomic(path,data):
    atomic_write(path,data)

def make_model(data,raw,policy):
    model=gp.Model()
    model.Params.OutputFlag=0
    model.Params.Threads=1
    model.Params.Seed=0
    model.Params.MIPGap=0
    model.Params.MIPGapAbs=0
    n=len(data['cap'])
    count=len(data['groups'])
    baseline=np.r_[data['cap'],np.zeros(count-n,dtype=np.int64)]
    menus=[[(0,0),(2,5),(4,8)] if j<n else [(0,0),(4,13),(8,23)] for j in range(count)]
    choices=[]
    for j,menu in enumerate(menus):
        variables=[model.addVar(vtype=gp.GRB.BINARY,obj=fee,ub=0 if policy=='onsite' and j>=n and k else 1) for k,(gain,fee) in enumerate(menu)]
        choices.append(variables)
        model.addConstr(gp.quicksum(variables)==1)
    gains=[gp.quicksum(option[0]*variable for option,variable in zip(menu,variables)) for menu,variables in zip(menus,choices)]
    access=[model.addConstr(gp.quicksum(gains[j] for j in np.flatnonzero(data['groups']==g))<=int(quota)) for g,quota in enumerate(data['quotas'])]
    requirements=[]
    for row in raw:
        outgoing=[[] for j in range(count)]
        for i,neighbors in enumerate(data['edges']):
            amount=int(row[i])
            if amount==0:
                continue
            flows=[]
            for j in neighbors:
                flow=model.addVar(lb=0)
                flows.append(flow)
                outgoing[j].append(flow)
            constraint=model.addConstr(gp.quicksum(flows)>=amount)
            requirements.append((constraint,amount))
        for j,flows in enumerate(outgoing):
            if flows:
                model.addConstr(gp.quicksum(flows)<=int(baseline[j])+gains[j])
    model.update()
    return dict(model=model,choices=choices,menus=menus,access=access,requirements=requirements)

def optimize(built,data,pressure,access_percent,start):
    model=built['model']
    for constraint,amount in built['requirements']:
        constraint.RHS=(amount*pressure+99)//100
    for constraint,quota in zip(built['access'],data['quotas']):
        constraint.RHS=int(quota)*access_percent//100
    if start is not None:
        for j,variables in enumerate(built['choices']):
            for k,variable in enumerate(variables):
                variable.Start=int(built['menus'][j][k][0]==start[j])
    model.optimize()
    if model.Status==gp.GRB.INFEASIBLE:
        return None,None,'infeasible'
    if model.Status!=gp.GRB.OPTIMAL:
        raise RuntimeError(f'Unclosed optimizer status {model.Status}')
    plan=np.array([built['menus'][j][max(range(3),key=lambda k:vs[k].X)][0] for j,vs in enumerate(built['choices'])],dtype=np.int64)
    return plan,int(round(model.ObjVal)),'optimal'

def run(city,method,job):
    boot=time.perf_counter()
    job=Path(job)
    job.mkdir(parents=True,exist_ok=True)
    source=R/('history_20261023/outputs/shenzhen_shared_capital_observed_case.npz' if city=='shenzhen' else f'fourcity_pilot_20261101/inputs/{city}_case.npz')
    w.old.v.v1.SOURCE=source
    w.old.v.v1.H=job
    data=w.old.v.v1.load()
    archive=np.load(source)
    days=archive['times'].astype('datetime64[D]')
    split=np.unique(days)[int(len(np.unique(days))*.7)]
    raw=archive['demand'][days<split].astype(np.int64)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    preparation=time.perf_counter()-boot
    output=job/'result.json'
    records=[]
    if output.exists():
        saved=json.loads(output.read_text(encoding='utf-8'))
        assert saved['source_sha256']==digest
        records=[r for r in saved['records'] if r['status'] in ('optimal','infeasible')]
    completed={(r['policy'],r['pressure_percent'],r['access_percent']) for r in records}
    def save(active=None):
        atomic(output,dict(city=city,method=method,records=records,source_sha256=digest,prep_seconds=preparation,elapsed=time.perf_counter()-boot,active=active,complete=len(records)==154))
    save()
    blocks=components(data) if method=='D-MIP' else [dict(sites=list(range(len(data['groups']))),cells=list(range(len(data['edges']))))]
    zero_tests={}
    originals=Certifier(data,cache=False)
    verification={}
    for policy in ['onsite','joint']:
        kept=None
        starts={}
        prior = [r for r in records if r['policy']==policy and r['status']=='optimal']
        if prior:
            last = prior[-1]
            plan_path = job/f"{policy}_{last['pressure_percent']}_{last['access_percent']}.npy"
            if plan_path.exists():
                previous_plan = np.load(plan_path,allow_pickle=False)
                for block_id,block in enumerate(blocks):
                    starts[block_id] = previous_plan[block['sites']]
        for pressure in range(100,151,5):
            demand=(raw*pressure+99)//100
            for access_percent in [200,175,150,125,100,75,50]:
                key=(policy,pressure,access_percent)
                if key in completed:
                    continue
                tick=time.perf_counter()
                save(dict(policy=policy,pressure=pressure,access=access_percent,phase='begin'))
                plan=np.zeros(len(data['groups']),dtype=np.int64)
                cost=0
                status='optimal'
                build_seconds=solve_seconds=0.
                for block_id,block in enumerate(blocks):
                    sub=subset(data,block)
                    rows=raw[:,block['cells']]
                    target=demand[:,block['cells']]
                    zero_key=(block_id,pressure)
                    if zero_key not in zero_tests:
                        cert=LocalOracle(sub,cache=True,prepared_tree=True)
                        zero_tests[zero_key]=not cert.check(np.zeros(len(sub['groups']),dtype=np.int64),target,stop_first=True)[0]
                    if zero_tests[zero_key]:
                        continue
                    start_build=time.perf_counter()
                    save(dict(policy=policy,pressure=pressure,access=access_percent,block=block_id,phase='build'))
                    built=kept if block_id==0 and kept is not None else make_model(sub,rows,policy)
                    build_seconds+=time.perf_counter()-start_build
                    if block_id==0:
                        kept=built
                    start_solve=time.perf_counter()
                    p,value,state=optimize(built,sub,pressure,access_percent,starts.get(block_id))
                    solve_seconds+=time.perf_counter()-start_solve
                    if block_id!=0:
                        built['model'].dispose()
                    if state=='infeasible':
                        status=state
                        break
                    starts[block_id]=p
                    plan[block['sites']]=p
                    cost+=value
                if status=='optimal':
                    quotas=data['quotas']*access_percent//100
                    assert all(plan[data['groups']==g].sum()<=q for g,q in enumerate(quotas))
                    if policy=='onsite':
                        assert not np.any(plan[len(data['cap']):])
                    exact_cost=sum(({0:0,2:5,4:8} if j<len(data['cap']) else {0:0,4:13,8:23})[int(v)] for j,v in enumerate(plan))
                    assert exact_cost==cost
                    plan_hash=hashlib.sha256(plan.tobytes()).hexdigest()
                    if verification.get(plan_hash,0)<pressure:
                        for row in demand:
                            assert originals.one(plan,row)[0]==0
                        verification[plan_hash]=pressure
                    name=f'{policy}_{pressure}_{access_percent}.npy'
                    np.save(job/name,plan)
                records.append(dict(policy=policy,pressure_percent=pressure,access_percent=access_percent,status=status,cost=cost if status=='optimal' else None,bound=cost if status=='optimal' else None,seconds=time.perf_counter()-tick,build_seconds=build_seconds,solve_seconds=solve_seconds,original_graph_verified=status=='optimal'))
                completed.add(key)
                save()
                print(city,method,len(records),key,status,cost,flush=True)
        if kept is not None:
            kept['model'].dispose()
    save()
