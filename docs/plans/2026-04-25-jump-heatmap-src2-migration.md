# Jump Heatmap Src2 Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move jump-heatmap computation onto the new `ASHoppingGKSL/src2/` one-particle correlator path so `nr` and `redfield` heatmap generation no longer depends on full-Fock steady-state solves.

**Architecture:** Add heatmap-block builders to `src2` for bath-resolved `loss/gain` coefficient matrices and weighted matrices derived from the cached one-particle correlator `C`. Rewrite `script/jump_heatmap_lib.jl` as a thin wrapper around `src2`, keep `plot_jump_heatmap.jl` focused on CLI/output concerns, and update tests to validate the new path without calling `steady_state_tb*`.

**Tech Stack:** Julia, ASHoppingGKSL, LinearAlgebra, CairoMakie, CSV, DataFrames, JLD2, Test

---

### Task 1: Lock in src2-backed heatmap behavior with failing tests

**Files:**
- Modify: `ASHoppingGKSL/test/jump_heatmap_test.jl`

**Step 1: Write the failing test**

- Update weighted-matrix tests to use one-particle correlators from `ASHoppingGKSL/src2/ASHoppingGKSL1P.jl`.
- Add checks that `JumpHeatmapLib` no longer exports or relies on `default_rho_for_method`.
- Keep coefficient-diagonal compatibility checks against existing local coefficient helpers.

**Step 2: Run test to verify it fails**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/jump_heatmap_test.jl`

Expected: FAIL because the current script library still computes weighted outputs via full-Fock `rho`.

### Task 2: Move heatmap block computation into src2

**Files:**
- Modify: `ASHoppingGKSL/src2/ASHoppingGKSL1P.jl`
- Create: `ASHoppingGKSL/src2/jump_heatmap_blocks.jl`

**Step 1: Write minimal implementation**

- Add `nr_jump_blocks_by_bath_1p(m)` and `redfield_jump_blocks_by_bath_1p(m)` to compute `loss/gain_coeff` matrices.
- Add a helper to attach `loss/gain_weighted` from `steady_correlator_nr(m)` / `steady_correlator_redfield(m)`.
- Export the new APIs from `ASHoppingGKSL1P`.

**Step 2: Run targeted tests**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl`

Expected: PASS

### Task 3: Rewrite the script wrapper to use src2

**Files:**
- Modify: `ASHoppingGKSL/script/jump_heatmap_lib.jl`
- Modify: `ASHoppingGKSL/script/plot_jump_heatmap.jl`

**Step 1: Write minimal implementation**

- Include `src2/ASHoppingGKSL1P.jl` and delegate all matrix generation to the new `src2` APIs.
- Remove full-Fock correlator helpers and the `default_rho_for_method` export.
- Keep CSV/JLD2/image save behavior unchanged.

**Step 2: Run jump-heatmap tests**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/jump_heatmap_test.jl`

Expected: PASS

### Task 4: Verify CLI output end-to-end

**Files:**
- Modify: `ASHoppingGKSL/script/*` if needed

**Step 1: Run CLI verification**

Run:

```bash
julia --project=ASHoppingGKSL ASHoppingGKSL/script/plot_jump_heatmap.jl \
  --nsites 3 --mu 0.8 --beta 8.0 --Gamma 1.0 --tau-c 0.2 \
  --method nr --formats png --matrix-formats csv,jld2 \
  --components abs --outdir /tmp/jump_heatmap_src2_nr
```

Run:

```bash
julia --project=ASHoppingGKSL ASHoppingGKSL/script/plot_jump_heatmap.jl \
  --nsites 3 --mu 0.8 --beta 8.0 --Gamma 1.0 --tau-c 0.2 \
  --method redfield --formats png --matrix-formats csv,jld2 \
  --components abs --outdir /tmp/jump_heatmap_src2_redfield
```

**Step 2: Verify output**

- Confirm both commands exit 0.
- Confirm `loss/gain_coeff` and `loss/gain_weighted` files are created.
- Confirm no `corr` files are created.
