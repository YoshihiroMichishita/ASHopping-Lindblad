# Isolated Canonical Correlator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `src2` helpers that build the isolated-system canonical one-particle correlator at fixed particle number and reuse the existing local-temperature/current helpers on that correlator.

**Architecture:** Work entirely in the single-particle eigenbasis of the isolated Hamiltonian. Enumerate fixed-particle-number occupations at half filling, compute canonical mode occupations, reconstruct the site-basis correlator `C`, and then pass `C` through `central_block`, `local_beta_mu_from_Cmid`, and `local_current_from_Cmid`.

**Tech Stack:** Julia, ASHoppingGKSL, LinearAlgebra, Test

---

### Task 1: Add failing tests for isolated canonical correlators

**Files:**
- Modify: `ASHoppingGKSL/test/src2_correlator_test.jl`

**Step 1: Write the failing test**

- Add a two-site `omega=0`, half-filling test that checks the isolated canonical correlator reproduces the known `C = [[1/2, r], [r, 1/2]]` form and returns `beta_eff ≈ beta`, `mu_eff ≈ 0`, and `current ≈ 0`.
- Add a four-site trace/Hermitian sanity check for the isolated canonical correlator.

**Step 2: Run test to verify it fails**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl`

Expected: FAIL because the isolated canonical helpers do not exist yet.

### Task 2: Implement isolated canonical helpers in src2

**Files:**
- Modify: `ASHoppingGKSL/src2/ASHoppingGKSL1P.jl`
- Create: `ASHoppingGKSL/src2/isolated_canonical.jl`

**Step 1: Write minimal implementation**

- Add a fixed-particle-number canonical occupation helper in the single-particle eigenbasis.
- Add `isolated_canonical_correlator(sys, beta; N=sys.n_sites ÷ 2)`.
- Add `isolated_local_observables(sys, beta; N=sys.n_sites ÷ 2)` that reuses the existing local observable helpers.

**Step 2: Run tests**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl`

Expected: PASS
