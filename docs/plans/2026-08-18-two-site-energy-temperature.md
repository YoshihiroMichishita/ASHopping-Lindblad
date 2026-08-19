# Two-Site Energy-Density Effective Temperature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reusable `src2` helper that computes `beta_eff` and `mu_eff` from the central two-site energy density and local particle number.

**Architecture:** Reuse the existing central two-site correlator flow. Compute `E_loc` and `N_loc` from `Cmid`, reconstruct the bonding and antibonding occupations, invert the Fermi functions analytically, and surface the result through `local_observables_from_C` and the existing CLI script.

**Tech Stack:** Julia, ASHoppingGKSL, LinearAlgebra, Test

---

### Task 1: Add Failing Tests For Energy-Density Temperature

**Files:**
- Modify: `ASHoppingGKSL/test/src2_correlator_test.jl`

**Step 1: Write the failing test**

Add checks in the existing `local two-site observables` testset:

```julia
energy_obs = ASHoppingGKSL1P.local_energy_temperature_from_Cmid(Cthermal, tmid)
@test energy_obs.energy_loc ≈ -tmid * nminus + tmid * nplus atol = 1e-12
@test energy_obs.number_loc ≈ nminus + nplus atol = 1e-12
@test energy_obs.beta_eff ≈ beta_ref atol = 1e-10
@test energy_obs.mu_eff ≈ mu_ref atol = 1e-10

energy_half = ASHoppingGKSL1P.local_energy_temperature_from_Cmid(
    ComplexF64[0.5 r_half; r_half 0.5],
    tmid,
)
energy_loc_half = energy_half.energy_loc
beta_from_half_formula = (2 / tmid) * atanh(-energy_loc_half / tmid)
@test energy_half.beta_eff ≈ beta_from_half_formula atol = 1e-10
```

**Step 2: Run test to verify it fails**

Run:

```bash
julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl
```

Expected: FAIL because `local_energy_temperature_from_Cmid` is not defined.

### Task 2: Implement The Helper

**Files:**
- Modify: `ASHoppingGKSL/src2/local_observables.jl`
- Modify: `ASHoppingGKSL/src2/ASHoppingGKSL1P.jl`

**Step 1: Export the helper**

Add `local_energy_temperature_from_Cmid` next to the existing local observable exports.

**Step 2: Implement the helper**

Add a function that:

- checks `Cmid` is `2x2`
- checks `t` is nonzero
- computes `energy_loc = -2 * t * real(Cmid[1, 2])`
- computes `number_loc = real(Cmid[1, 1] + Cmid[2, 2])`
- reconstructs `n_minus_energy` and `n_plus_energy`
- clamps occupations to `(eps, 1-eps)`
- computes `beta_eff`, `temp_eff`, and `mu_eff` using the existing inversion formula
- returns diagnostics for diagonal imbalance and imaginary coherence

**Step 3: Run tests**

Run:

```bash
julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl
```

Expected: PASS.

### Task 3: Surface The Energy-Density Result In Aggregators And CLI

**Files:**
- Modify: `ASHoppingGKSL/src2/local_observables_io.jl`
- Modify: `ASHoppingGKSL/src2/local_observables_sweep.jl`
- Modify: `ASHoppingGKSL/script/compute_local_observables.jl`
- Modify: `ASHoppingGKSL/test/src2_correlator_test.jl`
- Modify: `ASHoppingGKSL/test/local_observables_script_test.jl`

**Step 1: Extend `local_observables_from_C`**

Include `beta_eff_energy`, `temp_eff_energy`, `mu_eff_energy`, `energy_loc`, and
`number_loc` in the returned NamedTuple.

**Step 2: Extend gamma sweep output**

Include the same energy-density temperature fields in
`collect_local_observables_over_gamma`.

**Step 3: Extend CLI output**

Print the energy-density fields after the existing fit-based fields.

**Step 4: Add assertions**

Update tests to check the new fields are present and match the synthetic thermal
reference values.

**Step 5: Run tests**

Run:

```bash
julia --project=ASHoppingGKSL ASHoppingGKSL/test/src2_correlator_test.jl
julia --project=ASHoppingGKSL ASHoppingGKSL/test/local_observables_script_test.jl
```

Expected: PASS.

### Task 4: Verify End To End

**Files:**
- Modify: implementation files if verification reveals a mismatch

**Step 1: Run local observable CLI**

Run:

```bash
julia --project=ASHoppingGKSL ASHoppingGKSL/script/compute_local_observables.jl \
  --method nr --nsites 4 --t 1.0 --omega 0.0 --mu 0.8 --beta 8.0 --Gamma 1.0 --tau-c 0.2
```

Expected: output includes both fit-based and energy-density effective temperature fields.

**Step 2: Check formatting**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.
