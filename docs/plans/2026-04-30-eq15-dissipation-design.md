# Eq.15 Dissipation Design

**Goal:** Replace the current dissipative kernel in `IsThereSkinEffect` with a new kernel derived from Eq. (15) of `paper.tex`, while preserving the existing steady-state solver interface based on `gamma_gain` and `gamma_loss`.

## Context

The current implementation in `IsThereSkinEffect/src/dissipation.jl` constructs
site-basis gain and loss kernels from a shift matrix and an antisymmetric matrix
`A = (T - T') / (2im)`. This already produces a translationally invariant
nearest-neighbor dissipator, but it is not explicitly organized as the current-source
jump process written in Eq. (15) of the paper.

Eq. (15) is the wave-number representation of the current-source dissipator:

- loss rate: `Gamma_loss(k) = J * (2 - 2 * sin(k))`
- gain rate: `Gamma_gain(k) = J * (2 + 2 * sin(k))`

Here we reuse `ModelParams.J` as the common dissipative strength, with
`gamma^(l)(epsilon_k) = gamma^(g)(epsilon_k) = J`.

## Chosen Approach

Implement the new dissipator from local two-site jump operators and keep the
existing correlation-matrix equation unchanged.

Recommended local jump structure:

- loss jump on bond `(j, j+1)`: proportional to `c_j - i c_{j+1}`
- gain jump on bond `(j, j+1)`: proportional to `c_j^dagger + i c_{j+1}^dagger`

Summing these bond-local contributions over all bonds gives site-basis Hermitian
kernel matrices `gamma_loss` and `gamma_gain`. Under periodic boundary conditions,
their Fourier transform reproduces the Eq. (15) mode-resolved rates. Under open
boundary conditions, the same construction extends naturally by omitting the closing
bond `(L, 1)`.

## Why This Approach

- It matches the paper-level physics more directly than the current ad hoc kernel.
- It preserves the existing `steady_state.jl` API, which already expects
  `gamma_gain`, `gamma_loss`, and uses them in the closed equation for the one-body
  correlator.
- It supports both `PBC` and `OBC` without introducing a special k-space-only path.
- It keeps the new implementation isolated in a new file, making comparison against
  the old dissipator straightforward.

## File-Level Design

### `IsThereSkinEffect/src/dissipation_new.jl`

Add a new implementation with the following responsibilities:

- enumerate active bonds for `PBC` and `OBC`
- build local jump vectors for loss and gain on each bond
- accumulate site-basis kernel matrices from those jump vectors
- expose a public function returning `(gamma_gain, gamma_loss)` in the same format as
  the old code

Planned public entry points:

- `dissipation_kernels_eq15(p::ModelParams, bc::BoundaryCondition)`
- `dissipation_kernels_new(p::ModelParams, bc::BoundaryCondition)`

Planned helper functions:

- `active_bonds(L::Int, bc::BoundaryCondition)`
- `loss_jump_vector(L::Int, j::Int, jp1::Int)`
- `gain_jump_vector(L::Int, j::Int, jp1::Int)`

## Mathematical Mapping

For a single linear jump operator with site-basis coefficient vector `v`, the
correlator equation contributes through a positive semidefinite kernel proportional
to `v * v'`.

Therefore, the new site-basis kernels are assembled as

- `gamma_loss = J * sum(v_loss * v_loss' for each active bond)`
- `gamma_gain = J * sum(v_gain * v_gain' for each active bond)`

with

- `v_loss(j, j+1) = e_j - i e_{j+1}`
- `v_gain(j, j+1) = e_j + i e_{j+1}`

This yields:

- diagonal onsite dissipative weight from each neighboring bond
- nearest-neighbor purely imaginary off-diagonal terms
- translational invariance under `PBC`
- a direct route to the desired `2 +/- 2 sin(k)` mode-space rates

## Compatibility Strategy

The existing `dissipation.jl` remains untouched initially.

Integration path:

1. Add `dissipation_new.jl` and export its public constructor(s).
2. Keep `steady_covariance` unchanged for now.
3. Either call the new function explicitly in tests first, or add a thin switch later
   once the new kernels are verified.

This reduces regression risk and allows side-by-side numerical checks against the
current implementation.

## Validation Plan

The new design is correct if the following hold:

- In `PBC`, Fourier-transforming `gamma_loss` and `gamma_gain` produces diagonal
  entries matching `J * (2 - 2 * sin(k))` and `J * (2 + 2 * sin(k))`.
- `gamma_loss` and `gamma_gain` are Hermitian.
- Their eigenvalues are nonnegative up to numerical tolerance.
- The steady-state residual remains small when the new kernels are used.
- `PBC` still gives uniform density and bond current.
- `OBC` remains solvable and gives bulk behavior comparable to the `PBC` case.

## Risks

- The normalization between the Lindblad jump amplitude and the correlator-level
  kernel may be off by an overall factor of 2 if the mapping is implemented
  carelessly.
- The sign convention in the jump vectors controls whether the Fourier-space rates
  come out as `2 - 2 sin(k)` or `2 + 2 sin(k)`.
- `OBC` is an extension of the local construction, not a direct statement of Eq. (15),
  so tests should emphasize bulk behavior rather than exact k-space matching.
