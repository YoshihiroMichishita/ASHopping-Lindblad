# Local Observables IO And Script Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add reusable `src2` helpers and a CLI script that compute central two-site effective temperature/current either from a saved correlator file (`jld2` or CSV pair) or by computing a fresh correlator when no input file is provided.

**Architecture:** Extend `ASHoppingGKSL/src2/` with correlator-loading helpers and an aggregator that turns a full one-particle correlator into central-block observables. Add a standalone script that accepts either `--input` or model parameters; if `--input` is absent it builds the model and computes `C` with the `nr` or `redfield` one-particle solver.

**Tech Stack:** Julia, ASHoppingGKSL, LinearAlgebra, CSV, DataFrames, JLD2, Test

---

### Task 1: Add failing tests for file loading and compute fallback

**Files:**
- Modify: `ASHoppingGKSL/test/src2_correlator_test.jl`
- Create: `ASHoppingGKSL/test/local_observables_script_test.jl`

**Step 1: Write the failing test**

- Add tests for loading a correlator from `jld2`.
- Add tests for loading a correlator from `*_real.csv` and `*_imag.csv`.
- Add a CLI test that runs without `--input` and confirms output is produced from a fresh correlator computation.

**Step 2: Run tests to verify they fail**

Run:

```bash
julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl
julia --project=ASHoppingGKSL ASHoppingGKSL/test/local_observables_script_test.jl
```

Expected: FAIL because the IO helpers and script do not exist yet.

### Task 2: Implement src2 IO and aggregation helpers

**Files:**
- Modify: `ASHoppingGKSL/src2/ASHoppingGKSL1P.jl`
- Create: `ASHoppingGKSL/src2/local_observables_io.jl`

**Step 1: Write minimal implementation**

- Add `load_correlator_jld2`, `load_correlator_csv_pair`, and `load_correlator`.
- Add `local_observables_from_C(C, t)` and `local_observables_from_file(...)`.
- Export the new API from `ASHoppingGKSL1P`.

**Step 2: Run tests**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl`

Expected: PASS

### Task 3: Add the compute-or-load CLI script

**Files:**
- Create: `ASHoppingGKSL/script/compute_local_observables.jl`

**Step 1: Write minimal implementation**

- Parse `--input`, `--format`, `--method`, `--nsites`, and bath/model parameters.
- If `--input` is present, load `C` from file.
- If `--input` is absent, build the model and compute `C` with `src2`.
- Print central indices, `beta_eff`, `temp_eff`, `mu_eff`, `current`, and residuals.

**Step 2: Run script test**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/local_observables_script_test.jl`

Expected: PASS

### Task 4: Verify CLI end-to-end

**Files:**
- Modify: `ASHoppingGKSL/script/compute_local_observables.jl` if needed

**Step 1: Run compute fallback**

```bash
julia --project=ASHoppingGKSL ASHoppingGKSL/script/compute_local_observables.jl \
  --method nr --nsites 4 --t 1.0 --omega 0.0 --mu 0.8 --beta 8.0 --Gamma 1.0 --tau-c 0.2
```

**Step 2: Run file input path**

```bash
julia --project=ASHoppingGKSL ASHoppingGKSL/script/compute_local_observables.jl \
  --input /tmp/local_corr_test.jld2 --t 1.0
```

**Step 3: Verify output**

- Confirm both commands exit 0.
- Confirm printed output contains `beta_eff`, `mu_eff`, and `current`.
