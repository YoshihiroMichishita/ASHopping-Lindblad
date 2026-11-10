function _gibbs_checked_correlator(C::AbstractMatrix; physical_tol::Real=1e-8)
    size(C, 1) == size(C, 2) && size(C, 1) > 0 ||
        throw(DimensionMismatch("A correlator must be a nonempty square matrix"))
    isfinite(physical_tol) && physical_tol >= 0 ||
        throw(ArgumentError("physical_tol must be finite and nonnegative"))
    M = Matrix{ComplexF64}(C)
    all(isfinite, M) || throw(ArgumentError("A correlator must contain finite entries"))
    norm(M - M') <= physical_tol * max(1.0, norm(M)) ||
        throw(ArgumentError("A correlator must be Hermitian"))
    F = eigen(Hermitian((M + M') / 2))
    all(x -> -physical_tol <= x <= 1 + physical_tol, F.values) ||
        throw(DomainError(F.values, "Correlator eigenvalues must lie in [0, 1]"))
    # Correct only roundoff-size violations; reject unphysical input above.
    values = clamp.(F.values, 0.0, 1.0)
    return F.vectors * Diagonal(values) * F.vectors', values
end

function _gibbs_check_eps(eps::Real)
    isfinite(eps) && 0 < eps < 0.5 ||
        throw(ArgumentError("eps must lie strictly between 0 and 0.5"))
end

_gibbs_xlogx(x::Real) = iszero(x) ? 0.0 : x * log(x)

function _gibbs_cross_entropy(C, F; eps)
    q = clamp.(F.values, eps, 1 - eps)
    occupations = real.(diag(F.vectors' * C * F.vectors))
    return -sum(occupations .* log.(q) + (1 .- occupations) .* log1p.(-q))
end

"""
    gaussian_relative_entropy(C, Q; eps=1e-12, physical_tol=1e-8)

Quantum relative entropy `D(rho_C || rho_Q)` of number-conserving fermionic
Gaussian states with correlators `C[i,j] = <c_i† c_j>`. No trace normalization
of the correlators is performed, and `C` and `Q` need not commute.

Evaluates `tr(C(log C-log Q) + (I-C)(log(I-C)-log(I-Q)))` using spectral
matrix logarithms. Target eigenvalues 0 and 1 use the exact `0 log 0 = 0`
limit; reference eigenvalues are clipped to `[eps, 1-eps]` to regularize
singular references. Thus a support mismatch gives an eps-dependent finite
approximation to infinite relative entropy. Non-Hermitian or substantially
unphysical input is rejected. For non-Gaussian target states, this is the
relative entropy of the Gaussian state sharing their normal correlator.
"""
function gaussian_relative_entropy(C::AbstractMatrix, Q::AbstractMatrix;
                                   eps::Real=1e-12, physical_tol::Real=1e-8)
    _gibbs_check_eps(eps)
    size(C) == size(Q) || throw(DimensionMismatch("C and Q must have the same size"))
    target, values = _gibbs_checked_correlator(C; physical_tol)
    reference, _ = _gibbs_checked_correlator(Q; physical_tol)
    entropy = -sum(_gibbs_xlogx(x) + _gibbs_xlogx(1 - x) for x in values)
    cross_entropy = _gibbs_cross_entropy(target, eigen(Hermitian(reference)); eps)
    return max(0.0, cross_entropy - entropy)
end

function _reduced_gibbs_modes(sys::gksl.SystemParams, mu_ref::Real)
    n = sys.n_sites
    n >= 2 && iseven(n) || throw(ArgumentError("The system must have an even number of sites >= 2"))
    isfinite(mu_ref) || throw(ArgumentError("mu_ref must be finite"))
    h = gksl.single_particle_hamiltonian(sys)
    all(isfinite, h) || throw(ArgumentError("The system Hamiltonian must be finite"))
    F = eigen(Hermitian(h))
    inds = (n ÷ 2, n ÷ 2 + 1)
    return F.values .- mu_ref, F.vectors[inds[1]:inds[2], :], inds
end

function _reduced_gibbs_value_derivative(energies, modes, beta)
    # Branching prevents overflow at large positive/negative beta*energy.
    z = exp.(-abs.(beta .* energies))
    occupations = [beta * energies[a] >= 0 ? z[a] / (1 + z[a]) : 1 / (1 + z[a])
                   for a in eachindex(energies)]
    derivatives = -energies .* z ./ (1 .+ z).^2
    # Transpose convention: C_ij=<c_i† c_j>; the current Hamiltonian is real.
    Q = conj.(modes * Diagonal(occupations) * modes')
    Qdot = conj.(modes * Diagonal(derivatives) * modes')
    return Q, Qdot
end

"""
    reduced_gibbs_correlator(sys, beta; mu_ref=0.0)

Return the central 2x2 correlator of the full-system grand-canonical Gibbs
state at inverse temperature `beta` and fixed chemical potential `mu_ref`.
Diagonalizes only the LxL one-particle Hamiltonian. This is neither the Gibbs
state of an isolated dimer nor a canonical state at fixed particle number.
Negative and zero inverse temperatures are supported.
"""
function reduced_gibbs_correlator(sys::gksl.SystemParams, beta::Real; mu_ref::Real=0.0)
    isfinite(beta) || throw(ArgumentError("beta must be finite"))
    energies, modes, _ = _reduced_gibbs_modes(sys, mu_ref)
    Q, _ = _reduced_gibbs_value_derivative(energies, modes, beta)
    return Q
end

function _gibbs_cross_entropy_derivative(C, Q, Qdot; eps)
    F = eigen(Hermitian(Q))
    values = F.values
    q = clamp.(values, eps, 1 - eps)
    W = F.vectors' * C * F.vectors
    Vdot = F.vectors' * Qdot * F.vectors
    logdot = similar(Vdot)
    hole_logdot = similar(Vdot)
    # Frechet derivatives of the spectral matrix logarithms. Divided
    # differences retain off-diagonal contributions for noncommuting states.
    for i in eachindex(values), j in eachindex(values)
        if abs(values[i] - values[j]) <= 1e-12
            x = (values[i] + values[j]) / 2
            active = eps < x < 1 - eps
            logdot[i, j] = active ? Vdot[i, j] / x : 0
            hole_logdot[i, j] = active ? -Vdot[i, j] / (1 - x) : 0
        else
            gap = values[i] - values[j]
            logdot[i, j] = Vdot[i, j] * (log(q[i]) - log(q[j])) / gap
            hole_logdot[i, j] = Vdot[i, j] * (log1p(-q[i]) - log1p(-q[j])) / gap
        end
    end
    value = _gibbs_cross_entropy(C, F; eps)
    derivative = -real(tr(W * logdot + (I - W) * hole_logdot))
    return value, derivative
end

function _gibbs_stationary_point(evaluate, lo, hi; beta_tol, maxiter)
    mid = (lo + hi) / 2
    for iteration in 1:maxiter
        mid = (lo + hi) / 2
        value, derivative = evaluate(mid)
        if iszero(derivative) || hi - lo <= beta_tol * max(1.0, abs(mid))
            return (beta=mid, value=value, converged=true, iterations=iteration)
        elseif derivative < 0
            lo = mid
        else
            hi = mid
        end
    end
    value, _ = evaluate(mid)
    return (beta=mid, value=value, converged=false, iterations=maxiter)
end

"""
    local_temperature_from_reduced_gibbs(C, sys; mu_ref=0.0,
        beta_bounds=(0.0, 100.0), grid_points=201, beta_tol=1e-9,
        maxiter=200, eps=1e-12, physical_tol=1e-8)

Fit the central two-site state to the reduction of the full-system Gibbs
state by minimizing forward quantum relative entropy, with `mu_ref` fixed.
`C` may be the full LxL correlator or its central 2x2 block. No anomalous
pairing correlators are included. Allow negative temperatures explicitly
with, for example, `beta_bounds=(-100.0, 100.0)`.

A uniform grid brackets stationary minima, which are refined using the
analytic derivative of cross entropy. All bracketed minima and both search
endpoints are compared; increase `grid_points` for complicated reference
families. `beta_tol` controls the inverse-temperature bracket tolerance,
not uncertainty in the inferred temperature. The full-system Hamiltonian is
diagonalized once; each objective evaluation diagonalizes only a 2x2 block.

Returns `beta_eff`, `temp_eff` (`Inf` at beta=0), `mu_ref`, `central_indices`,
`Cmid`, `reference_Cmid`, Gaussian `relative_entropy`, `fit_residual`
(Frobenius norm), `beta_bounds`, `at_boundary`, `converged`, `iterations`,
and `reference_regularization` (maximum reference eigenvalue clipping).
Inspect `at_boundary` before interpreting a bound-limited fit as a temperature.
For a non-Gaussian target, the minimizer is still the forward-relative-entropy
minimizer because log(reference) is quadratic, but the reported relative
entropy uses the Gaussian state sharing C rather than the true target entropy.
"""
function local_temperature_from_reduced_gibbs(C::AbstractMatrix, sys::gksl.SystemParams;
                                             mu_ref::Real=0.0,
                                             beta_bounds::Tuple{<:Real,<:Real}=(0.0, 100.0),
                                             grid_points::Integer=201,
                                             beta_tol::Real=1e-9, maxiter::Integer=200,
                                             eps::Real=1e-12, physical_tol::Real=1e-8)
    _gibbs_check_eps(eps)
    lo, hi = Float64.(beta_bounds)
    isfinite(lo) && isfinite(hi) && lo < hi ||
        throw(ArgumentError("beta_bounds must be finite and strictly increasing"))
    grid_points >= 3 || throw(ArgumentError("grid_points must be >= 3"))
    isfinite(beta_tol) && beta_tol > 0 || throw(ArgumentError("beta_tol must be finite and positive"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    energies, modes, inds = _reduced_gibbs_modes(sys, mu_ref)
    all(iszero, energies) && throw(ArgumentError("The Gibbs reference is independent of temperature"))
    size(C) == (sys.n_sites, sys.n_sites) || size(C) == (2, 2) ||
        throw(DimensionMismatch("C must be the full system correlator or the central 2x2 block"))
    block = size(C) == (2, 2) ? C : C[inds[1]:inds[2], inds[1]:inds[2]]
    target, _ = _gibbs_checked_correlator(block; physical_tol)
    evaluate(beta) = begin
        Q, Qdot = _reduced_gibbs_value_derivative(energies, modes, beta)
        _gibbs_cross_entropy_derivative(target, Q, Qdot; eps)
    end

    grid = collect(range(lo, hi; length=grid_points))
    if lo < 0 < hi
        push!(grid, 0.0)
        sort!(unique!(grid))
    end
    samples = evaluate.(grid)
    candidates = [(beta=lo, value=first(samples)[1], converged=true, iterations=0),
                  (beta=hi, value=last(samples)[1], converged=true, iterations=0)]
    for i in eachindex(grid)
        if iszero(samples[i][2]) || iszero(grid[i])
            push!(candidates, (beta=grid[i], value=samples[i][1], converged=true, iterations=0))
        end
        if i < length(grid) && samples[i][2] < 0 < samples[i + 1][2]
            push!(candidates, _gibbs_stationary_point(evaluate, grid[i], grid[i + 1]; beta_tol, maxiter))
        end
    end
    # If the coarse grid missed a stationary bracket, expose that fact.
    sample_best = argmin(first.(samples))
    comparison_tol = 64 * Base.eps(Float64) * max(1.0, abs(samples[sample_best][1]))
    if samples[sample_best][1] < minimum(candidate.value for candidate in candidates) - comparison_tol
        push!(candidates, (beta=grid[sample_best], value=samples[sample_best][1],
                           converged=false, iterations=0))
    end
    minimum_value = minimum(candidate.value for candidate in candidates)
    tied = filter(candidate -> candidate.value <= minimum_value + comparison_tol, candidates)
    # Resolve roundoff-level objective ties by stationarity. In particular,
    # beta=0 must not become a spurious huge negative/positive temperature.
    best = tied[argmin([(abs(evaluate(candidate.beta)[2]), abs(candidate.beta)) for candidate in tied])]
    Q, _ = _reduced_gibbs_value_derivative(energies, modes, best.beta)
    reference_values = eigvals(Hermitian(Q))
    return (
        beta_eff=best.beta,
        temp_eff=iszero(best.beta) ? Inf : 1 / best.beta,
        mu_ref=Float64(mu_ref),
        central_indices=inds,
        Cmid=target,
        reference_Cmid=Q,
        relative_entropy=gaussian_relative_entropy(target, Q; eps, physical_tol),
        fit_residual=norm(target - Q),
        beta_bounds=(lo, hi),
        at_boundary=best.beta == lo || best.beta == hi,
        converged=best.converged,
        iterations=best.iterations,
        reference_regularization=maximum(abs.(reference_values - clamp.(reference_values, eps, 1 - eps))),
    )
end
