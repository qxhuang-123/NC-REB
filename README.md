# NC-REB

Exact common-capital planning for EV charging infrastructure with discrete engineering menus, spatial service requirements and shared access limits.

## Methods and frozen experiment

- MILP uses explicit time-indexed service flows.
- D-MIP solves investment-closed components, retains the largest component model and rebuilds other component models using engineering starts.
- HR-MIP uses local maximal contracts and laminar service reformulation.
- CR-BBC uses an integer engineering master with spatial separation and certificate reuse.

The frozen experiment has 154 contracts per city: two engineering policies, eleven demand multipliers from 1.00 to 1.50, and seven access-quota multipliers. All training periods are jointly enforced. Engineering fees, capacity increments, candidate sites and access groups are planning scenarios. Occupancy observations supply the baseline service profiles.

`results/` contains the recorded four-city runs, including memory stops and an interrupted Shenzhen D-MIP run. The latter has incomplete first-segment resource measurements. Values reported as recorded segment sums omit unmeasured interruption time. Raw observations are attributed to Yu ruotao, DOI https://doi.org/10.6084/m9.figshare.32917595.v1 . The original data license is CC BY 4.0.

## Install and data

Python 3.9 was used. Install `requirements.txt`. Gurobi requires a valid license. Windows is required for the measured Job-object resource controller.

Download `NCREB_four_city_planning_inputs.zip` from the [v0.1.0 release](https://github.com/qxhuang-123/NC-REB/releases/tag/v0.1.0) and extract it at the repository root. Required paths and SHA256 values are in `data/manifest.json`. The raw four-city databases and all returned engineering vectors are supplied in separate data archives. The full-database Zenodo deposition is pending account authorization; the planning-input archive is also available as a GitHub release asset. Source observations remain available at the DOI above.

```bash
python run_experiments.py --check-inputs
python run_experiments.py --city shenzhen --method CR-BBC
python run_experiments.py --city shenzhen --method MILP D-MIP HR-MIP CR-BBC
python run_experiments.py
```

Each completed worker stores `result.json`, engineering vectors and sampled progress checkpoints under `research/four_methods_batch_20261109/jobs/`. Resource outcomes are written alongside the worker scripts. Commands skip existing final resource records. Preserve an interrupted run before beginning a continuation. The CLI rejects a concurrent invocation through a file lock.

The resource protocol is single-threaded, zero relative and absolute MIP gaps, and a 12,000,000,000-byte Job memory cap. CR-BBC has a 3,600-second whole-batch limit. The MILP variants run without a time cap, as recorded in the protocol. System-awake requests are active while the queue runs. Data processing does not need an online service; Gurobi licensing follows the installed license.

## Timing

Worker processing time includes model construction, optimization, solution recovery, certification and record management, excluding initial input preparation. Process elapsed time starts outside the worker. The two quantities are close because input preparation is short. Optimizer-only time is available in the explicit baseline per-query fields, but was not separately recorded for every historical method. No missing optimizer-only times are inferred.

Progress memory is sampled at 0.2-second intervals. A fast sequence may cross a threshold before it is sampled, so actual settled counts are retained. The 100% checkpoint uses the final independent verification and final process resource record.

## Recorded outcomes

The four CR-BBC batches each settled all 154 contracts. Original MILP, D-MIP and HR-MIP include memory-limited batches. `results/final_validation_20261111.json` compares returned statuses and costs. Each configuration was run once in this cohort. Models, results and measurement conventions must be matched when comparing performance.

Code is released under MIT. Data keeps the separate CC BY 4.0 attribution. The code archive excludes databases, Gurobi licenses, credentials and local IDE settings.
