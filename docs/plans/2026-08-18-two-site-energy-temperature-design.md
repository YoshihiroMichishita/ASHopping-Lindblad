# Two-Site Energy-Density Effective Temperature Design

## Goal

Add a local effective-temperature definition based on the two-site energy density, reusing the existing central two-site correlator pipeline.

## Design

The existing `src2` local-observable code works with the one-particle correlator
`C_ij = <c_i^dagger c_j>`. For the central two-site block `Cmid`, the local bond
energy and local particle number are

```math
E_loc = -2t Re C_{12},
N_loc = Re(C_{11} + C_{22}).
```

For the two-site thermal reference system, the bonding and antibonding
occupations are

```math
n_- = f(-t; beta, mu),
n_+ = f(+t; beta, mu).
```

Matching the two energy-density moments gives

```math
E_loc = -t n_- + t n_+,
N_loc = n_- + n_+.
```

This can be solved without a nonlinear solver:

```math
n_- = (N_loc - E_loc/t)/2,
n_+ = (N_loc + E_loc/t)/2.
```

Then the same inversion used by `local_beta_mu_from_Cmid` gives `beta_eff` and
`mu_eff`. This keeps the implementation small and consistent with the existing
local Fermi-fit helper.

## API

Add `local_energy_temperature_from_Cmid(Cmid, t; eps=1e-12)` to
`ASHoppingGKSL/src2/local_observables.jl`, export it from `ASHoppingGKSL1P`, and
include its output in `local_observables_from_C`.

The new helper returns a NamedTuple with `beta_eff`, `temp_eff`, `mu_eff`,
`energy_loc`, `number_loc`, `n_minus_energy`, `n_plus_energy`, and diagnostics.

## Testing

Extend the existing local two-site observable tests using synthetic thermal
correlators. The helper should recover the reference `beta` and `mu` for a
general grand-canonical two-site state and agree with the half-filled analytic
energy-density formula when `mu = 0`.
