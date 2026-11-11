# coding: utf-8
from pathlib import Path
import importlib.util,json,shutil
from progress_io import atomic_write
H=Path(__file__).resolve().parent

def run(city,method,folder):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    if method in ['MILP','D-MIP']:
        from baselines import run as baseline
        baseline(city,method,folder)
        return
    spec=importlib.util.spec_from_file_location('historical',H/'historical_worker.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.H=folder
    module.R=H.parent
    internal='MILP' if method=='HR-MIP' else 'CRBBC'
    expected=folder/f'{city}_{internal}.json'
    result=folder/'result.json'
    if result.exists():
        shutil.copy2(result,expected)
    original_write=module.write
    def write(path,data):
        if Path(path)==expected:
            atomic_write(result,data)
        else:
            atomic_write(Path(path),data)
    module.write=write
    module.worker(city,internal,resume_milp=True)
