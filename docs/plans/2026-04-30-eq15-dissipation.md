# Eq.15 Dissipation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new `IsThereSkinEffect/src/dissipation_new.jl` that builds the current-source dissipative kernels from Eq. (15) using bond-local jump operators and validates that the resulting kernels reproduce the expected `k`-space rates.

**Architecture:** Keep the steady-state solver and observable pipeline unchanged. Add a parallel dissipator implementation that constructs site-basis `gamma_gain` and `gamma_loss` by summing bond-local kernel contributions. Verify the implementation first at the kernel level and then through the existing steady-state solve.

**Tech Stack:** Julia, LinearAlgebra, Test, IsThereSkinEffect

---

### Task 1: Add failing tests for the Eq. (15) kernels

**Files:**
- Modify: `IsThereSkinEffect/test/runtests.jl`

**Step 1: Write the failing tests**

- Add a testset for the new dissipator.
- Build `gamma_gain`, `gamma_loss` from the new function for a small `PBC` chain.
- Assert that both matrices are Hermitian.
- Fourier-transform both matrices and assert that their diagonal entries match
  `J * (2 + 2 * sin(k))` and `J * (2 - 2 * sin(k))` for the discrete lattice momenta.
- Assert that the off-diagonal elements in the Fourier basis are numerically small.

**Step 2: Run tests to verify they fail**

Run: `julia --project=IsThereSkinEffect -e 'using Pkg; Pkg.test("IsThereSkinEffect")'`

Expected: FAIL because `dissipation_new.jl` and its public API do not exist yet.

### Task 2: Implement the new dissipator module

**Files:**
- Create: `IsThereSkinEffect/src/dissipation_new.jl`
- Modify: `IsThereSkinEffect/src/IsThereSkinEffect.jl`

**Step 1: Write minimal implementation**

- Add helper(s) to enumerate active bonds for `PBC` and `OBC`.
- Add helper(s) that build the local loss and gain jump vectors on a given bond.
- Accumulate `gamma_loss` and `gamma_gain` as sums of outer products scaled by `p.J`.
- Export `dissipation_kernels_eq15` and `dissipation_kernels_new`.

**Step 2: Run tests**

Run: `julia --project=IsThereSkinEffect -e 'using Pkg; Pkg.test("IsThereSkinEffect")'`

Expected: kernel-level tests now PASS.

### Task 3: Verify compatibility with the steady-state solver

**Files:**
- Modify: `IsThereSkinEffect/test/runtests.jl`

**Step 1: Add integration tests**

- Use `single_particle_hamiltonian` together with the new kernels and solve the steady
  correlator through the existing linear equation.
- Assert that the residual is small.
- Assert that the resulting correlator is Hermitian.
- Assert that `PBC` density and current remain uniform within tolerance.

**Step 2: Run tests**

Run: `julia --project=IsThereSkinEffect -e 'using Pkg; Pkg.test("IsThereSkinEffect")'`

Expected: PASS with small residuals and uniform `PBC` observables.

### Task 4: Add OBC coverage for the local-bond extension

**Files:**
- Modify: `IsThereSkinEffect/test/runtests.jl`

**Step 1: Add OBC checks**

- Build the new kernels for `OBC`.
- Assert Hermiticity and positive-semidefinite spectrum up to numerical tolerance.
- Solve the steady correlator and assert a small residual.
- Check that bulk densities and currents are finite and numerically stable.

**Step 2: Run tests**

Run: `julia --project=IsThereSkinEffect -e 'using Pkg; Pkg.test("IsThereSkinEffect")'`

Expected: PASS

### Task 5: Decide integration surface

**Files:**
- Modify: `IsThereSkinEffect/src/steady_state.jl` or keep unchanged

**Step 1: Choose minimal exposure**

- Preferred first step: keep `steady_state.jl` unchanged and expose the new dissipator as
  an explicit alternate API.
- Optional follow-up: add a thin selector so callers can opt into the Eq. (15) kernels
  from `steady_covariance` and `steady_state`.

**Step 2: Re-run tests**

Run: `julia --project=IsThereSkinEffect -e 'using Pkg; Pkg.test("IsThereSkinEffect")'`

Expected: PASS with no regression to the old path.
