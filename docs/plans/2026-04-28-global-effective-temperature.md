# Global Effective Temperature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `src2` helper that estimates a global effective temperature from the full one-particle correlator `C` by fitting isolated-system mode occupations to a Fermi-Dirac form.

**Architecture:** Diagonalize the isolated single-particle Hamiltonian, rotate `C` into the mode basis, extract diagonal occupations, and fit them to a thermal distribution. For the primary `omega=0` use case, default to a `μ_eff = 0` one-parameter fit and report both off-diagonal mode-basis norm and fit residual.

**Tech Stack:** Julia, ASHoppingGKSL, LinearAlgebra, Test

---

### Task 1: Add failing tests for synthetic thermal correlators

**Files:**
- Modify: `ASHoppingGKSL/test/src2_correlator_test.jl`

**Step 1: Write the failing test**

- Build a synthetic thermal correlator `C = U * diag(f(β ε)) * U'` from the isolated single-particle Hamiltonian.
- Assert that the helper recovers `beta_eff ≈ beta`, `mu_eff ≈ 0`, small `offdiag_norm`, and small `fit_residual`.

**Step 2: Run test to verify it fails**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl`

Expected: FAIL because the global effective-temperature helper does not exist yet.

### Task 2: Implement the global effective-temperature helper

**Files:**
- Modify: `ASHoppingGKSL/src2/ASHoppingGKSL1P.jl`
- Create: `ASHoppingGKSL/src2/global_observables.jl`

**Step 1: Write minimal implementation**

- Add `global_effective_temperature(C, sys; fix_mu_zero=true, eps=1e-12)`.
- Implement the `μ_eff = 0` fit from mode occupations and report diagnostic norms.

**Step 2: Run tests**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl`

Expected: PASS
