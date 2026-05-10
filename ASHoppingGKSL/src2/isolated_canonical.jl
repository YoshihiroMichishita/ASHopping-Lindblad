function canonical_mode_occupations(eps::AbstractVector{<:Real}, beta::Real, N::Integer)
    n = length(eps)
    0 <= N <= n || error("Particle number N must satisfy 0 <= N <= length(eps)")

    occ = zeros(Float64, n)
    Z = 0.0

    maxmask = UInt(1) << n
    for mask in UInt(0):(maxmask - UInt(1))
        count_ones(mask) == N || continue

        energy = 0.0
        for a in 1:n
            if ((mask >> (a - 1)) & UInt(1)) == UInt(1)
                energy += eps[a]
            end
        end

        w = exp(-beta * energy)
        Z += w
        for a in 1:n
            if ((mask >> (a - 1)) & UInt(1)) == UInt(1)
                occ[a] += w
            end
        end
    end

    Z > 0 || error("Partition function vanished in canonical_mode_occupations")
    return occ ./ Z
end

function isolated_canonical_correlator(sys::gksl.SystemParams, beta::Real; N::Integer = sys.n_sites ÷ 2)
    eps, U = gksl.single_particle_eigensystem(sys)
    occ = canonical_mode_occupations(eps, beta, N)
    return ComplexF64.(U * Diagonal(occ) * U')
end

function isolated_local_observables(sys::gksl.SystemParams, beta::Real; N::Integer = sys.n_sites ÷ 2)
    C = isolated_canonical_correlator(sys, beta; N=N)
    obs = local_observables_from_C(C, sys.t)
    return merge((C = C,), obs)
end
