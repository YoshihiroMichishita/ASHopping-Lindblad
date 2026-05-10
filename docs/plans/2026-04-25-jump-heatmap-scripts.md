# Jump Heatmap Scripts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add standalone scripts that compute bath-resolved jump-term coefficient matrices, save them as CSV/JLD2, and render heatmaps as PNG/PDF without modifying existing package source files.

**Architecture:** Keep all new logic in new files under `ASHoppingGKSL/script/` and validate it with a new standalone test file under `ASHoppingGKSL/test/`. Reuse existing package functions such as `nr_jumps_by_bath`, `redfield_local_gain_loss_coeffs`, and `operator_linear_coeffs_in_site_basis` to build full `i,j` coefficient matrices and rho-weighted diagnostics.

**Tech Stack:** Julia, ASHoppingGKSL, Test, CSV, DataFrames, JLD2, CairoMakie

---

### Task 1: Add regression tests for jump-matrix assembly

**Files:**
- Create: `ASHoppingGKSL/test/jump_heatmap_test.jl`
- Create: `ASHoppingGKSL/script/jump_heatmap_lib.jl`

**Step 1: Write the failing test**

- Test that the NR bath-resolved coefficient matrices have diagonals matching `nr_local_gain_loss_coeffs_bath`.
- Test that the summed Redfield coefficient matrix diagonals match `redfield_local_gain_loss_coeffs`.
- Test that rho-correlator matrices have the expected shapes and traces for a steady state.

**Step 2: Run test to verify it fails**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/jump_heatmap_test.jl`

Expected: failure because `ASHoppingGKSL/script/jump_heatmap_lib.jl` does not exist yet.

**Step 3: Write minimal implementation**

- Add `jump_heatmap_lib.jl` with helpers to build models, assemble NR and Redfield coefficient matrices, compute correlator and weighted matrices, and save matrices.

**Step 4: Run test to verify it passes**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/jump_heatmap_test.jl`

Expected: PASS

### Task 2: Add a CLI script for export and plotting

**Files:**
- Create: `ASHoppingGKSL/script/plot_jump_heatmap.jl`

**Step 1: Write the failing test**

- Extend the standalone test file to exercise the CLI entrypoint with a small system and verify that PNG and CSV outputs are created.

**Step 2: Run test to verify it fails**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/jump_heatmap_test.jl`

Expected: failure because the plotting/export script does not exist yet.

**Step 3: Write minimal implementation**

- Add CLI parsing.
- Export coefficient, correlator, and weighted matrices to CSV and JLD2.
- Render bath-resolved heatmaps to PNG and optionally PDF with CairoMakie.

**Step 4: Run test to verify it passes**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/jump_heatmap_test.jl`

Expected: PASS

### Task 3: Verify end-to-end execution

**Files:**
- Create: `ASHoppingGKSL/script/plot_jump_heatmap.jl`

**Step 1: Run the production command**

Run:

```bash
julia --project=ASHoppingGKSL ASHoppingGKSL/script/plot_jump_heatmap.jl \
  --nsites 3 --mu 0.8 --beta 8.0 --Gamma 1.0 --tau-c 0.2 \
  --method nr --formats png --matrix-formats csv,jld2
```

**Step 2: Verify outputs**

- Check that output matrices exist in the selected output directory.
- Check that heatmaps exist as PNG files.

**Step 3: Optional PDF verification**

Run:

```bash
julia --project=ASHoppingGKSL ASHoppingGKSL/script/plot_jump_heatmap.jl \
  --nsites 3 --mu 0.8 --beta 8.0 --Gamma 1.0 --tau-c 0.2 \
  --method redfield --formats png,pdf --matrix-formats csv,jld2
```

**Step 4: Summarize verification evidence**

- Report the commands run, exit codes, and representative output files created.
