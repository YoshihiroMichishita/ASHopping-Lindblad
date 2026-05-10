# Local Observables Gamma Sweep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reusable `src2` function that scans `Gamma_*` correlator directories and returns vectors of `Gamma`, `beta_eff`, `temp_eff`, `mu_eff`, and `current`.

**Architecture:** Reuse the existing `load_correlator` and `local_observables_from_file` helpers. Implement a small sweep utility that parses `Gamma` from directory names, loads each correlator file, computes local observables, sorts by `Gamma`, and returns a named tuple of vectors.

**Tech Stack:** Julia, ASHoppingGKSL, JLD2, CSV, Test

---

### Task 1: Add failing tests for Gamma sweep aggregation

**Files:**
- Modify: `ASHoppingGKSL/test/src2_correlator_test.jl`

**Step 1: Write the failing test**

- Create a temporary `Gamma_*` directory tree with small synthetic correlator files.
- Assert that the sweep helper returns sorted `Gamma`, `beta_eff`, `temp_eff`, `mu_eff`, and `current` vectors.

**Step 2: Run test to verify it fails**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl`

Expected: FAIL because the sweep helper does not exist yet.

### Task 2: Implement the sweep helper

**Files:**
- Modify: `ASHoppingGKSL/src2/ASHoppingGKSL1P.jl`
- Create: `ASHoppingGKSL/src2/local_observables_sweep.jl`

**Step 1: Write minimal implementation**

- Add `gamma_from_path(path)` to parse `Gamma_*`.
- Add `collect_local_observables_over_gamma(rootdir; filename=\"nr_correlator.jld2\", t=1.0, format=:jld2)`.
- Export the new helper from `ASHoppingGKSL1P`.

**Step 2: Run test**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl`

Expected: PASS
