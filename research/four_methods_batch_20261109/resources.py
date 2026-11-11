"""Isolated resource protocol; historical implementations/results remain intact."""
import ctypes as C
from ctypes import wintypes as W
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from progress_io import read_snapshot, atomic_write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LIMIT_BYTES = 12_000_000_000  # Decimal GB, not GiB.
SECONDS = 3600


def prepare():
    dst = HERE/'runtime_src/nc_reb'
    shutil.copytree(ROOT/'src/nc_reb', dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    changes = {
        'target_b.py': [('dp_seconds=5,state_limit=50000', "dp_seconds=float('inf'),state_limit=float('inf')")],
        'active_b.py': [('dp_seconds=5,state_limit=50000', "dp_seconds=float('inf'),state_limit=float('inf')")],
        'pure_b.py': [('>5:', ">float('inf'):"), ('>2000000:', ">float('inf'):"), ('>50000:', ">float('inf'):")],
    }
    hashes = {}
    for name, pairs in changes.items():
        path = dst/name
        before = path.read_bytes()
        code = before.decode('utf-8')
        for a,b in pairs:
            assert a in code, (name,a)
            code = code.replace(a,b)
        path.write_text(code, encoding='utf-8')
        hashes[name] = dict(original=hashlib.sha256(before).hexdigest(), resource_version=hashlib.sha256(path.read_bytes()).hexdigest())
    code = (ROOT/'research/target_contract_20261027/benchmark.py').read_text(encoding='utf-8')
    code = code.replace("str(ROOT/'src')", "str(HERE/'runtime_src')")
    code = code.replace('deadline=boot+150', 'deadline=boot+3600')
    code = code.replace('min(deadline,tick+90)', 'deadline')
    (HERE/'snapshot_benchmark.py').write_text(code, encoding='utf-8')
    manifest = dict(seconds=SECONDS, memory_bytes=LIMIT_BYTES, memory_unit='12 decimal GB; Windows Job aggregate committed memory',
                    budget_unit='one method / one complete city contract / one cold process',
                    state_limit=None, transition_limit=None, inner_DP_limit=None,
                    checkpoint_cache_note='200000 retained-checkpoint cache bound remains; it does not stop solving',
                    no_state_threshold_sensitivity=True, hashes=hashes,
                    hybrid_note='No old 5s/state-trigger fallback under this protocol; no new switching strategy claimed')
    (HERE/'protocol.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


class BASIC(C.Structure):
    _fields_ = [('PerProcessUserTimeLimit', C.c_longlong), ('PerJobUserTimeLimit', C.c_longlong),
                ('LimitFlags', W.DWORD), ('MinimumWorkingSetSize', C.c_size_t), ('MaximumWorkingSetSize', C.c_size_t),
                ('ActiveProcessLimit', W.DWORD), ('Affinity', C.c_size_t), ('PriorityClass', W.DWORD), ('SchedulingClass', W.DWORD)]


class IO(C.Structure):
    _fields_ = [(s, C.c_ulonglong) for s in ('ReadOperationCount','WriteOperationCount','OtherOperationCount',
                                          'ReadTransferCount','WriteTransferCount','OtherTransferCount')]


class EXT(C.Structure):
    _fields_ = [('BasicLimitInformation', BASIC), ('IoInfo', IO), ('ProcessMemoryLimit', C.c_size_t),
                ('JobMemoryLimit', C.c_size_t), ('PeakProcessMemoryUsed', C.c_size_t), ('PeakJobMemoryUsed', C.c_size_t)]


def controlled(args, name, seconds=SECONDS, memory=LIMIT_BYTES):
    k = C.WinDLL('kernel32', use_last_error=True)
    k.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]; k.CreateJobObjectW.restype = W.HANDLE
    k.SetInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]
    k.QueryInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p]
    k.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
    k.TerminateJobObject.argtypes = [W.HANDLE, W.UINT]
    k.CloseHandle.argtypes = [W.HANDLE]
    def check(ok):
        if not ok: raise C.WinError(C.get_last_error())
    job = k.CreateJobObjectW(None, None); check(job)
    info = EXT(); info.BasicLimitInformation.LimitFlags = 0x2000 | (0x200 if memory is not None else 0)
    if memory is not None:
        info.JobMemoryLimit = memory
    check(k.SetInformationJobObject(job, 9, C.byref(info), C.sizeof(info)))
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    start = time.perf_counter(); p = None
    try:
        with (HERE/(name+'.stdout.log')).open('w', encoding='utf-8') as out, (HERE/(name+'.stderr.log')).open('w', encoding='utf-8') as err:
            p = subprocess.Popen([sys.executable, '-X', 'utf8', __file__, 'child', *args], stdin=subprocess.PIPE,
                                 stdout=out, stderr=err, env=env)
            # Child waits before imports/allocations until assigned to the Job.
            check(k.AssignProcessToJobObject(job, W.HANDLE(int(p._handle))))
            readback = EXT(); check(k.QueryInformationJobObject(job, 9, C.byref(readback), C.sizeof(readback), None))
            if memory is not None:
                assert abs(readback.JobMemoryLimit-memory) < 4096, (readback.JobMemoryLimit, memory)
            enforced_limit = readback.JobMemoryLimit if memory is not None else None
            p.stdin.write(b'go\n'); p.stdin.flush(); p.stdin.close()
            timed_out = False
            next_checkpoint = 0
            checkpoint_rows = []
            job_path = Path(args[2])
            continuation = read_snapshot(job_path/'resume_context.json') or {}
            thresholds = [31, 62, 93, 124, 154]
            while p.poll() is None:
                elapsed = time.perf_counter() - start
                if seconds is not None and elapsed >= seconds:
                    timed_out = True
                    check(k.TerminateJobObject(job, 124))
                    p.wait()
                    break
                result_path = job_path/'result.json'
                if result_path.exists():
                    current = read_snapshot(result_path)
                    if current is None:
                        time.sleep(.2)
                        continue
                    settled = [r for r in current['records'] if r['status'] in ('optimal','infeasible')]
                    while next_checkpoint < len(thresholds) and len(settled) >= thresholds[next_checkpoint]:
                        count = thresholds[next_checkpoint]
                        snap = EXT()
                        check(k.QueryInformationJobObject(job, 9, C.byref(snap), C.sizeof(snap), None))
                        checkpoint_rows.append(dict(queries=count,settled=len(settled),wall_seconds=elapsed,
                            worker_core_seconds=max(0,current['elapsed']-current['prep_seconds']),
                            query_seconds=sum(r.get('seconds',0) for r in settled[:count]),
                            peak_job_commit_bytes=snap.PeakJobMemoryUsed,
                            continuation=continuation))
                        next_checkpoint += 1
                        atomic_write(job_path/'checkpoints.json',checkpoint_rows)
                time.sleep(.2)
            check(k.QueryInformationJobObject(job, 9, C.byref(readback), C.sizeof(readback), None))
            result = dict(exit_code=p.returncode, timeout=timed_out, cold_seconds=time.perf_counter()-start,
                          memory_limit_bytes=memory, enforced_limit_bytes=enforced_limit, peak_job_commit_bytes=readback.PeakJobMemoryUsed,
                          job_limit_readback_verified=memory is not None, time_limit_seconds=seconds)
            result['continuation'] = continuation
            (HERE/(name+'.resource.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
            return result
    finally:
        if p is not None and p.poll() is None:
            k.TerminateJobObject(job, 125); p.wait()
        k.CloseHandle(job)


def child(args):
    assert sys.stdin.readline().strip() == 'go'
    from task_worker import run
    run(*args)

if __name__ == '__main__':
    if sys.argv[1] == 'child': child(sys.argv[2:])
