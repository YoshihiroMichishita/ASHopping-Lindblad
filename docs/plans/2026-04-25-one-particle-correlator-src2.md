# One-Particle Correlator Src2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new `ASHoppingGKSL/src2/` implementation that computes steady one-particle correlators for `nr` and `redfield` without solving the full density matrix.

**Architecture:** Build a bilinear operator basis over `I` and `c_i^\dagger c_j`, assemble the adjoint generator on that reduced basis, and solve the resulting `n_sites^2` linear system for the correlator matrix `C`. Keep all new logic isolated under `src2/` and validate against the existing full-density-matrix reference for small systems.

**Tech Stack:** Julia, ASHoppingGKSL, LinearAlgebra, Test

---

### Task 1: Add failing tests for the new src2 module

**Files:**
- Create: `ASHoppingGKSL/test/src2_correlator_test.jl`
- Create: `ASHoppingGKSL/src2/ASHoppingGKSL1P.jl`

**Step 1: Write the failing test**

- Reference `ASHoppingGKSL/src2/ASHoppingGKSL1P.jl`.
- Compare `steady_correlator_nr(m)` to `tr(c_i' * c_j * rho_nr)` from the existing `steady_state_tb_nr`.
- Compare `steady_correlator_redfield(m)` to `tr(c_i' * c_j * rho_red)` from the existing `steady_state_tb`.

**Step 2: Run test to verify it fails**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl`

Expected: fail because `ASHoppingGKSL/src2/ASHoppingGKSL1P.jl` does not exist yet.

### Task 2: Implement the reduced bilinear-basis solver

**Files:**
- Create: `ASHoppingGKSL/src2/ASHoppingGKSL1P.jl`
- Create: `ASHoppingGKSL/src2/bilinear_basis.jl`
- Create: `ASHoppingGKSL/src2/nr_correlator.jl`
- Create: `ASHoppingGKSL/src2/redfield_correlator.jl`
- Create: `ASHoppingGKSL/src2/weighted_observables.jl`

**Step 1: Write minimal implementation**

- Add a bilinear basis type and projection helpers.
- Implement adjoint generators for `nr` and `redfield`.
- Solve `M x = -b` and reshape into `C`.
- Add helpers to reconstruct weighted observables from `C`.

**Step 2: Run tests to verify they pass**

Run: `julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl`

Expected: PASS

### Task 3: Verify sample execution

**Files:**
- Create: `ASHoppingGKSL/src2/*`

**Step 1: Run a sample command**

Run:

```bash
julia --project=ASHoppingGKSL -e 'push!(LOAD_PATH, "ASHoppingGKSL"); include("ASHoppingGKSL/src2/ASHoppingGKSL1P.jl"); using .ASHoppingGKSL1P; import ASHoppingGKSL as gksl; sys=gksl.SystemParams(6,1.0,0.0); b1=gksl.make_bath2(Gamma=1.0,mu=-0.8,beta=8.0,tau_c=0.2); b2=gksl.make_bath2(Gamma=1.0,mu=0.8,beta=8.0,tau_c=0.2); m=gksl.TwoBathModel(sys,b1,b2); C=ASHoppingGKSL1P.steady_correlator_nr(m); println(size(C)); println(real.(diag(C)));'
```

**Step 2: Verify output**

- Confirm the command exits 0.
- Confirm the correlator shape is `(6, 6)`.
