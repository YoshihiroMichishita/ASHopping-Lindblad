# Replot Jump Heatmap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a standalone script that reloads existing jump heatmap JLD2 matrices and redraws them as publication-ready heatmaps with larger fonts and simplified titles.

**Architecture:** Keep the existing matrix-generation workflow unchanged. Add a separate plotting-only script that scans a directory of JLD2 files, loads stored matrices and metadata, and writes styled PNG/PDF outputs into a `replot/` directory.

**Tech Stack:** Julia, JLD2, CairoMakie

---

### Task 1: Add a failing script-level verification target

**Files:**
- Create: `ASHoppingGKSL/script/replot_jump_heatmap.jl`

**Step 1: Define the expected behavior**

- Accept an input directory like `out/nr_sweep6/Gamma_0.2`.
- Discover `*_weighted.jld2` files by default.
- Render new heatmaps with larger fonts and short titles.
- Save outputs into `replot/` under the input directory unless overridden.

**Step 2: Run the script to verify it fails**

Run:

```bash
julia --project=ASHoppingGKSL ASHoppingGKSL/script/replot_jump_heatmap.jl --indir out/nr_sweep6/Gamma_0.2
```

Expected: FAIL because the script does not exist yet.

### Task 2: Implement the JLD2 reload-and-render script

**Files:**
- Create: `ASHoppingGKSL/script/replot_jump_heatmap.jl`

**Step 1: Write minimal implementation**

- Add CLI parsing for input directory, output directory, formats, component, and filename pattern.
- Load `matrix` and optional `metadata` from each selected JLD2 file.
- Generate clean titles from filenames.
- Render heatmaps with larger paper-style font settings.
- Save outputs to the selected directory.

**Step 2: Run the script**

Run:

```bash
julia --project=ASHoppingGKSL ASHoppingGKSL/script/replot_jump_heatmap.jl --indir out/nr_sweep6/Gamma_0.2
```

Expected: PASS and write new output images.

### Task 3: Verify representative outputs

**Files:**
- Create: `ASHoppingGKSL/script/replot_jump_heatmap.jl`

**Step 1: Check output files**

- Confirm that publication-style images are created for the four weighted bath matrices.
- Confirm both output naming and destination directory are correct.

**Step 2: Optional PDF verification**

Run:

```bash
julia --project=ASHoppingGKSL ASHoppingGKSL/script/replot_jump_heatmap.jl --indir out/nr_sweep6/Gamma_0.2 --formats png,pdf
```

Expected: PASS with both PNG and PDF outputs.
