function thermal_rates(eps::AbstractVector{<:Real}, alpha::Real, beta::Real)
    n_modes = length(eps)
    W = zeros(Float64, n_modes, n_modes)
    for from in 1:n_modes, to in 1:n_modes
        if from != to
            W[from, to] = float(alpha) * exp(-0.5 * float(beta) * (eps[to] - eps[from]))
        end
    end
    return W
end

function drive_rates_pbc(p::ModelParams, ks::AbstractVector{<:Real})
    t_a = p.t * sinh(p.J)
    gain = similar(collect(Float64, ks))
    loss = similar(gain)
    for i in eachindex(ks)
        gain[i] = max(2 * t_a * sin(ks[i]), 0.0)
        loss[i] = max(-2 * t_a * sin(ks[i]), 0.0)
    end
    return gain, loss
end

function drive_rates_obc(p::ModelParams, U::AbstractMatrix{<:Complex})
    return drive_rates_site(p, U; periodic = false)
end

function drive_rates_site(p::ModelParams, U::AbstractMatrix{<:Complex}; periodic::Bool)
    n_sites, n_modes = size(U)
    W = zeros(Float64, n_modes, n_modes)
    n_bonds = periodic ? n_sites : (n_sites - 1)
    for from in 1:n_modes, to in 1:n_modes
        if from == to
            continue
        end
        forward = 0.0
        backward = 0.0
        for i in 1:n_bonds
            j = periodic && i == n_sites ? 1 : i + 1
            forward += abs2(conj(U[j, to]) * U[i, from])
            backward += abs2(conj(U[i, to]) * U[j, from])
        end
        W[from, to] = 2 * p.gamma_plus * forward + 2 * p.gamma_minus * backward
    end
    return W
end

function thermal_reference_occupations(eps::AbstractVector{<:Real}, p::ModelParams)
    return [1.0 / (exp(p.beta * (e - p.mu)) + 1.0) for e in eps]
end
