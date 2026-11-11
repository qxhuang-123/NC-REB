"""Run the archived four-city methods with measured resources."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
BATCH = ROOT / 'research/four_methods_batch_20261109'


def check_inputs():
    entries = json.loads((ROOT / 'data/manifest.json').read_text())
    for entry in entries:
        if entry['role'] != 'planning_input':
            continue
        path = ROOT / entry['path']
        if not path.exists():
            raise FileNotFoundError(f'Extract the planning-input archive first: {path}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['sha256']:
            raise ValueError(f'Input checksum mismatch: {path}')
        print(f"Verified {entry['city']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', nargs='+', choices=['shenzhen', 'beijing', 'shanghai', 'chongqing'], default=['shenzhen', 'beijing', 'shanghai', 'chongqing'])
    parser.add_argument('--method', nargs='+', choices=['MILP', 'D-MIP', 'HR-MIP', 'CR-BBC'], default=['CR-BBC', 'D-MIP', 'MILP', 'HR-MIP'])
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    check_inputs()
    if args.check_inputs:
        return
    if sys.platform != 'win32':
        parser.error('Measured Job-object resource enforcement requires Windows.')
    import ctypes
    import msvcrt
    sys.path.insert(0, str(BATCH))
    from resources import controlled
    lock = (BATCH / 'queue.lock').open('a+b')
    lock.write(b'0')
    lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for city in args.city:
            for method in args.method:
                name = f'{city}_{method}'
                if (BATCH / f'{name}.resource.json').exists():
                    print(f'Skipped completed resource record: {name}')
                    continue
                job = BATCH / 'jobs' / name
                job.mkdir(parents=True, exist_ok=True)
                limit = 3600 if method == 'CR-BBC' else None
                print(name, controlled([city, method, str(job)], name, seconds=limit), flush=True)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__ == '__main__':
    main()
