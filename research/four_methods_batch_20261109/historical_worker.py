# coding: utf-8
"""Frozen all-training parametric batch. Run child under the Windows Job harness."""
from pathlib import Path
import sys,time,json,hashlib,math,traceback
H=Path(__file__).resolve().parent;R=H.parent
def write(p,obj):
    q=Path(str(p)+'.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)

def worker(city,method,resume_milp=False):
    boot=time.perf_counter()
    deadline=boot+3580 if method=='CRBBC' else math.inf
    if False:
        raise ValueError('Only the MILP continuation has an unlimited deadline')
    import numpy as np
    import gurobipy as gp
    sys.path[:0]=[str(R/'fourcity_full_20261104'),str(R/'certification_20261101'),str(R/'laminar_20261103'),str(R/'new2_20261103')]
    import worker_reuse_integer as w
    from oracle import LocalOracle
    from certifier import Certifier
    source=R/('history_20261023/outputs/shenzhen_shared_capital_observed_case.npz' if city=='shenzhen' else 'fourcity_pilot_20261101/inputs/'+city+'_case.npz')
    work=H/(city+'_'+method);work.mkdir(exist_ok=True)
    w.old.v.v1.SOURCE=source;w.old.v.v1.H=work
    d=w.old.v.v1.load();z=np.load(source);days=z['times'].astype('datetime64[D]');split=np.unique(days)[int(len(np.unique(days))*.7)]
    raw=z['demand'][days<split].astype(np.int64);held=z['demand'][days>=split].astype(np.int64)
    n=len(d['cap']);ns=len(d['groups']);base=np.r_[d['cap'],np.zeros(ns-n,dtype=np.int64)]
    quotas=d['quotas'].copy();out=H/(city+'_'+method+'.json');records=[];bottlenecks=[]
    prep=time.perf_counter()-boot;stats={'candidates':0,'optimizer_calls':0,'region_hits':0,'zero_points':0};validation={}
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    contracts=[(policy,lp,rp) for policy in ('onsite','joint')
               for lp in range(100,151,5) for rp in (200,175,150,125,100,75,50)]
    def contract(rec):
        return rec['policy'],rec['pressure_percent'],rec['access_percent']
    def keep_for_validation(rec,p):
        pk=rec['plan_hash'];lp=rec['pressure_percent']
        if pk not in validation or validation[pk]['lp']<lp:
            validation[pk]=dict(lp=lp,plan=p.copy())
    def records_file():
        write(out,dict(city=city,method=method,records=records,stats=stats,
                       elapsed=time.perf_counter()-boot,prep_seconds=prep,
                       source_sha256=source_hash if resume_milp else hashlib.sha256(source.read_bytes()).hexdigest(),
                       training_periods=len(raw),holdout_periods=len(held),complete=len(records)==154,
                       unlimited_milp_continuation=resume_milp))
    saved={}
    saved_plans={}
    if resume_milp and out.exists():
        checkpoint=json.loads(out.read_text(encoding='utf-8'))
        if (checkpoint['city'],checkpoint['method'],checkpoint['source_sha256'])!=(city,method,source_hash):
            raise ValueError('Checkpoint does not match the frozen input')
        for rec in checkpoint['records']:
            key=contract(rec)
            if key not in contracts or key in saved:
                raise ValueError(f'Invalid or duplicate checkpoint query: {key}')
            saved[key]=rec
            if rec.get('plan'):
                p=np.load(work/rec['plan'],allow_pickle=False)
                if p.shape!=(ns,) or hashlib.sha256(p.tobytes()).hexdigest()!=rec['plan_hash']:
                    raise ValueError(f'Invalid saved plan: {key}')
                saved_plans[key]=p
            if rec['status'] in ('optimal','infeasible'):
                records.append(rec)
                if rec['status']=='optimal':
                    keep_for_validation(rec,saved_plans[key])
                stats['zero_points']+=int(rec['origin']=='zero_certificate')
                stats['optimizer_calls']+=int(rec['origin']=='optimizer')
    settled={contract(rec) for rec in records}
    opts=[[(0,0),(2,5),(4,8)] if j<n else [(0,0),(4,13),(8,23)] for j in range(ns)]
    for policy in ('onsite','joint'):
        if time.perf_counter()>=deadline:break
        if all(key in settled for key in contracts if key[0]==policy):continue
        oracle=LocalOracle(d,cache=True,prepared_tree=True);m=gp.Model();m.Params.OutputFlag=0;m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=0;m.Params.MIPGapAbs=0
        xs=[[m.addVar(vtype=gp.GRB.BINARY,obj=c,ub=0 if policy=='onsite' and j>=n and k else 1) for k,(a,c) in enumerate(op)] for j,op in enumerate(opts)]
        gain=[gp.quicksum(a*v for (a,c),v in zip(op,vs)) for op,vs in zip(opts,xs)]
        for vs in xs:m.addConstr(gp.quicksum(vs)==1)
        access=[m.addConstr(gp.quicksum(gain[j] for j in np.flatnonzero(d['groups']==g))<=int(q)) for g,q in enumerate(quotas)]
        flat=[v for vs in xs for v in vs];gains=np.array([[a for a,c in op] for op in opts]);demand_constraints=[];pool={};regions=[];last_plan=None
        # Both solvers use identical local maximal contracts and laminar representation.
        initial=raw.copy();oracle.bind(initial)
        if method=='MILP':
            for e in oracle.parts:
                part=e['part'];tree=e['tree'];local=e['index'].maximal
                if tree.laminar:
                    cv=[gp.LinExpr() for _ in tree.nodes]
                    for j,v in enumerate(tree.assignment):
                        if v>=0:cv[v]+=int(base[part['sites'][j]])+gain[part['sites'][j]]
                    for row in local:
                        res=[m.addVar(lb=0) for _ in tree.nodes]
                        for i in range(tree.m):demand_constraints.append((m.addConstr(res[i]>=int(row[i])-cv[i]),int(row[i]),True,0))
                        for v in range(tree.m,len(tree.nodes)):m.addConstr(res[v]>=gp.quicksum(res[u] for u in tree.children[v])-cv[v])
                        m.addConstr(res[tree.root]==0)
                else:
                    for row in local:
                        inc={j:[] for j in part['sites']}
                        for loc,i in enumerate(part['cells']):
                            fs=[]
                            for j in d['edges'][i]:
                                f=m.addVar(lb=0);fs.append(f);inc[j].append(f)
                            demand_constraints.append((m.addConstr(gp.quicksum(fs)>=int(row[loc])),int(row[loc]),False,0))
                        for j,fs in inc.items():
                            if fs:m.addConstr(gp.quicksum(fs)<=int(base[j])+gain[j])
            m.update()
            # A laminar leaf constraint's RHS includes the fixed capacity term.
            demand_constraints=[(c,a,t,c.RHS-a) for c,a,t,_ in demand_constraints]
        else:m.Params.LazyConstraints=1
        m.update();built=time.perf_counter()
        for lp in range(100,151,5):
            if all((policy,lp,rp) in settled for rp in (200,175,150,125,100,75,50)):
                for rp in (200,175,150,125,100,75,50):
                    if (policy,lp,rp) in saved_plans:last_plan=saved_plans[policy,lp,rp]
                continue
            dem=(raw*lp+99)//100;oracle.bind(dem)
            if method=='MILP':
                for c,a,t,offset in demand_constraints:c.RHS=(a*lp+99)//100+offset
            # Retained lazy cuts have weaker RHS at ascending pressure, hence remain valid.
            for key,entry in pool.items():
                S,N=key;rhs=int(dem[:,list(S)].sum(axis=1).max()-base[list(N)].sum())
                entry['rhs']=rhs
                if entry.get('constraint') is None:entry['constraint']=m.addConstr(gp.quicksum(gain[j] for j in N)>=rhs)
                else:entry['constraint'].RHS=rhs
            for rp in (200,175,150,125,100,75,50):
                key=(policy,lp,rp)
                if key in settled:
                    if key in saved_plans:last_plan=saved_plans[key]
                    continue
                tick=time.perf_counter();rec=dict(policy=policy,pressure_percent=lp,access_percent=rp)
                if tick>=deadline:break
                current_q=quotas*rp//100
                for c,q in zip(access,current_q):c.RHS=int(q)
                def resource_ok(p):return all(int(p[d['groups']==g].sum())<=q for g,q in enumerate(current_q))
                def feasible(p):return not oracle.check(p,dem,stop_first=True)[0]
                p=None;cost=None;origin=None;status=None
                # A free solution is optimal because all menu fees are nonnegative.
                zero=np.zeros(ns,dtype=np.int64)
                if feasible(zero):p=zero;cost=0;status='optimal';origin='zero_certificate';stats['zero_points']+=1
                if status is None and method=='REGION':
                    for a in reversed(regions):
                        if a['lp']<=lp and a['rp']>=rp:
                            if a['status']=='infeasible':status='infeasible';origin='infeasible_region';break
                            candidate=a['plan']
                            if resource_ok(candidate) and feasible(candidate):
                                p=candidate;cost=a['cost'];status='optimal';origin='optimal_region';rec['anchor']=a['id'];stats['region_hits']+=1;break
                errors=[];accepted=set()
                def callback(model,where):
                    if where!=gp.GRB.Callback.MIPSOL:return
                    try:
                        stats['candidates']+=1
                        pp=gains[np.arange(ns),np.argmax(np.asarray(model.cbGetSolution(flat)).reshape(ns,3),axis=1)]
                        violated=[(key,v) for key,v in pool.items() if int(pp[list(key[1])].sum())<v['rhs']]
                        if not violated:
                            cuts,metrics,worst,ws=oracle.check(pp,dem,stop_first=True)
                            for ww in ws:
                                S=tuple(ww['demand_cells']);N=tuple(ww['supply_indices']);rhs=int(dem[:,list(S)].sum(axis=1).max()-base[list(N)].sum())
                                key=(S,N);pool.setdefault(key,dict(rhs=rhs));pool[key]['rhs']=rhs;violated.append((key,pool[key]))
                                if len(bottlenecks)<1000:bottlenecks.append(dict(policy=policy,lp=lp,rp=rp,**ww))
                        for (S,N),v in violated:model.cbLazy(gp.quicksum(gain[j] for j in N)>=v['rhs'])
                        if not violated:accepted.add(pp.tobytes())
                    except Exception:errors.append(traceback.format_exc());model.terminate()
                if status is None:
                    start_plan=saved_plans.get(key,last_plan)
                    if start_plan is not None:
                        for j,vs in enumerate(xs):
                            for k,v in enumerate(vs):v.Start=int(opts[j][k][0]==start_plan[j])
                    if not resume_milp:m.Params.TimeLimit=max(.01,deadline-time.perf_counter())
                    stats['optimizer_calls']+=1
                    if method=='MILP':m.optimize()
                    else:m.optimize(callback)
                    if errors:raise RuntimeError('\n'.join(errors))
                    rec['bound']=float(m.ObjBound) if math.isfinite(m.ObjBound) else None
                    if m.Status==gp.GRB.INFEASIBLE:status='infeasible'
                    elif m.SolCount:
                        p=gains[np.arange(ns),np.argmax(np.array(m.getAttr('X',flat)).reshape(ns,3),axis=1)]
                        assert feasible(p) and resource_ok(p)
                        cost=int(round(m.ObjVal));status='optimal' if m.Status==2 else 'unfinished_feasible'
                    else:status='unfinished'
                    origin='optimizer'
                rec.update(status=status,cost=cost,origin=origin,seconds=time.perf_counter()-tick,pool_size=len(pool))
                if p is not None:
                    assert resource_ok(p);last_plan=p.copy()
                    rec['plan_hash']=hashlib.sha256(p.tobytes()).hexdigest()
                    name=policy+'_'+str(lp)+'_'+str(rp)
                    if resume_milp:name+='_'+rec['plan_hash']
                    name+='.npy'
                    np.save(work/name,p)
                    rec.update(plan=name,new_sites=int(np.count_nonzero(p[n:])),expansions=int(np.count_nonzero(p[:n])),access_used=int(p.sum()))
                    keep_for_validation(rec,p)
                if status in ('optimal','infeasible') and origin=='optimizer':regions.append(dict(lp=lp,rp=rp,status=status,cost=cost,plan=None if p is None else p.copy(),id=len(records)))
                records.append(rec);records_file()
            if time.perf_counter()>=deadline:break
        m.dispose()
    # Every unattempted point is explicit, even if the outer resource guard stops later.
    done={(a['policy'],a['pressure_percent'],a['access_percent']) for a in records}
    for policy in ('onsite','joint'):
        for lp in range(100,151,5):
            for rp in (200,175,150,125,100,75,50):
                if (policy,lp,rp) not in done:records.append(dict(policy=policy,pressure_percent=lp,access_percent=rp,status='unfinished',origin='batch_budget',cost=None))
    # Independent full original-graph verification, shared only for identical plans
    # at weaker pressures. It is charged inside the same batch budget.
    if resume_milp:
        order={key:i for i,key in enumerate(contracts)}
        records.sort(key=lambda rec:order[contract(rec)])
    checked={};original=Certifier(d,cache=False)
    for pk,v in validation.items():
        if time.perf_counter()>=deadline:break
        target=(raw*v['lp']+99)//100;loss=0;finished=True
        for row in target:
            if time.perf_counter()>=deadline:finished=False;break
            loss+=int(original.one(v['plan'],row)[0])
        if finished:
            assert loss==0
            checked[pk]=v['lp']
    for rec in records:
        if rec.get('plan_hash'):rec['original_graph_verified']=checked.get(rec['plan_hash'],0)>=rec['pressure_percent']
    stats['original_verified_plans']=len(checked);stats['original_verification_requests']=original.stats
    write(work/'bottlenecks.json',bottlenecks);records_file()

def queue(cities):
    import importlib.util
    sp=importlib.util.spec_from_file_location('limits',R/'fourcity_full_20261104/resources_integer.py');lim=importlib.util.module_from_spec(sp);sp.loader.exec_module(lim)
    lim.HERE=H;lim.__file__=str(Path(__file__).resolve())
    for city in cities:
        for method in ('CRBBC','REGION','MILP'):
            key=city+'_'+method
            if not (H/(key+'.resource.json')).exists():lim.controlled([city,method],key)
    print('queue complete',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='child':
        assert sys.stdin.readline().strip()=='go';worker(*sys.argv[2:])
    else:queue(sys.argv[1:] or ['shenzhen','beijing','shanghai','chongqing'])
