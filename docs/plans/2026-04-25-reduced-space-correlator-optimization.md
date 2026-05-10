# Reduced-Space Correlator Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite `ASHoppingGKSL/src2/` correlator assembly to evaluate adjoint actions on the `vacuum + one-particle` reduced space instead of the full Fock space.

**Architecture:** Replace the full-Fock bilinear-basis projection with reduced-space operators of size `(n_sites + 1) × (n_sites + 1)`. Use reduced annihilation/creation operators, reconstruct source and one-particle blocks directly from reduced operators, and keep full-Fock reference tests only for correctness checks at `nsites=4`.

**Tech Stack:** Julia, ASHoppingGKSL, LinearAlgebra, Test

---

### Task 1: Add failing reduced-space tests

**Files:**
- Modify: `ASHoppingGKSL/test/src2_correlator_test.jl`

**Step 1: Write the failing test**

- Add tests for `reduced_annihilation_ops`, `reduced_creation_ops`, `reduced_bilinear_basis`, and `source_and_bilinear_from_reduced`.
- Switch the correlator-reference tests to `nsites=4`.

**Step 2: Run test to verify it fails**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl`

Expected: fail because reduced-space helpers do not exist yet.

### Task 2: Implement reduced-space correlator assembly

**Files:**
- Modify: `ASHoppingGKSL/src2/ASHoppingGKSL1P.jl`
- Modify: `ASHoppingGKSL/src2/bilinear_basis.jl`
- Modify: `ASHoppingGKSL/src2/nr_correlator.jl`
- Modify: `ASHoppingGKSL/src2/redfield_correlator.jl`

**Step 1: Write minimal implementation**

- Add reduced-space operator builders.
- Rewrite NR and Redfield adjoint actions to work on reduced operators.
- Read source terms and bilinear blocks directly from reduced matrices.

**Step 2: Run tests to verify they pass**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl`

Expected: PASS

### Task 3: Verify sample execution

**Files:**
- Modify: `ASHoppingGKSL/src2/*`

**Step 1: Run reduced-space sample commands**

Run sample `nr` and `redfield` commands at `nsites=4`.

**Step 2: Verify output**

- Confirm each command exits 0.
- Confirm each correlator has shape `(4, 4)`.
