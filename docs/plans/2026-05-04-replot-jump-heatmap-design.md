# Replot Jump Heatmap Design

**Goal:** Add a standalone script that reads already-computed jump heatmap matrices from JLD2 files and redraws publication-ready heatmaps with larger fonts and cleaner titles, without recomputing any model data.

## Context

The existing script `ASHoppingGKSL/script/plot_jump_heatmap.jl` computes jump matrices, writes CSV/JLD2 outputs, and renders heatmaps directly. The directory `out/nr_sweep6/Gamma_0.2` already contains the saved JLD2 matrices:

- `nr_bath1_gain_weighted.jld2`
- `nr_bath1_loss_weighted.jld2`
- `nr_bath2_gain_weighted.jld2`
- `nr_bath2_loss_weighted.jld2`
- related `*_coeff.jld2` and `nr_correlator.jld2`

Each JLD2 file stores a `matrix` dataset and a `metadata` dictionary. Since the underlying matrices already exist, the new work should focus only on loading, styling, and saving figures.

## Chosen Approach

Add a new script under `ASHoppingGKSL/script/` dedicated to replotting existing JLD2 files.

The script will:

- take an input directory such as `out/nr_sweep6/Gamma_0.2`
- discover target `*.jld2` matrix files
- load `matrix` and optional `metadata`
- render the same matrix values with improved visual settings
- write new image files to a safe output directory, defaulting to a nested `replot/`

This avoids changing the meaning of the original export script and avoids any risk of silently recomputing different data.

## Rendering Changes

The new figure style should be paper-oriented rather than quick-look oriented:

- larger title font
- larger axis-label and tick-label fonts
- larger colorbar label and tick-label fonts
- shorter, cleaner titles such as `Bath 1 Gain` instead of long internal filenames
- support for both PNG and PDF output

The default plotted component should be `abs`, matching the current `*_weighted_abs.png` convention, but the script should allow `real`, `imag`, and `abs` when desired.

## File-Level Design

### `ASHoppingGKSL/script/replot_jump_heatmap.jl`

Responsibilities:

- parse CLI options
- collect matching JLD2 files from an input directory
- map file names to human-readable titles
- load matrices and metadata
- call a local rendering helper with paper-style font sizes
- save outputs into an output directory

Expected default target files:

- `*_weighted.jld2`

Optional expansion:

- allow `*_coeff.jld2` via a pattern option

## Compatibility Strategy

Keep the new script independent from `plot_jump_heatmap.jl`.

- no recomputation of matrices
- no behavior changes to existing scripts
- only light reuse of existing helper logic when it reduces duplication safely

If a small amount of shared rendering code is useful, it may be extracted later, but the first implementation should prioritize clarity and isolation.

## Validation Plan

The implementation is correct if:

- it successfully reads JLD2 matrix files from `out/nr_sweep6/Gamma_0.2`
- it produces a new set of image files with the expected names in the output directory
- the script exits successfully for the provided directory
- the generated figures visually preserve the original data while using larger, paper-friendly text
