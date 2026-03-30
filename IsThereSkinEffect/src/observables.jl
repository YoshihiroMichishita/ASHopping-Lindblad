function site_density(U::AbstractMatrix{<:Complex}, occupations::AbstractVector{<:Real})
    weights = abs2.(U)
    density = weights * collect(Float64, occupations)
    return real.(density)
end

function coherent_bond_currents(result::SteadyStateResult, p::ModelParams)
    U = result.basis_transform
    occ = result.occupations

    if result.boundary == :pbc
        currents = zeros(Float64, p.L)
        for i in 1:p.L
            j = i == p.L ? 1 : i + 1
            correlator = 0.0 + 0.0im
            for m in eachindex(occ)
                correlator += conj(U[j, m]) * U[i, m] * occ[m]
            end
            currents[i] = 2 * tbar(p) * imag(correlator)
        end
        return currents
    elseif result.boundary == :obc
        currents = zeros(Float64, p.L - 1)
        for i in 1:(p.L - 1)
            correlator = 0.0 + 0.0im
            for m in eachindex(occ)
                correlator += conj(U[i + 1, m]) * U[i, m] * occ[m]
            end
            currents[i] = 2 * tbar(p) * imag(correlator)
        end
        return currents
    else
        error("unsupported boundary = $(result.boundary)")
    end
end

function dissipative_fluxes(result::SteadyStateResult, p::ModelParams)
    if result.boundary == :pbc
        isnothing(result.mode_labels) && error("PBC results must store mode labels")
        ks = result.mode_labels
        gain, loss = drive_rates_pbc(p, ks)
        flux = sum(gain .* (1 .- result.occupations) .+ loss .* result.occupations) / p.L
        return fill(flux, p.L)
    elseif result.boundary == :obc
        return bond_currents(result, p)
    else
        error("unsupported boundary = $(result.boundary)")
    end
end

function dissipative_flux_expectation(result::SteadyStateResult, p::ModelParams)
    fluxes = dissipative_fluxes(result, p)
    return sum(fluxes) / length(fluxes)
end

function bond_currents(result::SteadyStateResult, p::ModelParams)
    if result.boundary == :pbc
        isnothing(result.mode_labels) && error("PBC results must store mode labels")
        ks = result.mode_labels
        avg_current = 2 * tbar(p) * sum(sin.(ks) .* result.occupations) / p.L
        return fill(avg_current, p.L)
    elseif result.boundary == :obc
        U = result.basis_transform
        occ = result.occupations
        currents = zeros(Float64, p.L - 1)
        for i in 1:(p.L - 1)
            forward = 0.0
            backward = 0.0
            for from in eachindex(occ), to in eachindex(occ)
                forward += abs2(conj(U[i + 1, to]) * U[i, from]) * occ[from] * (1 - occ[to])
                backward += abs2(conj(U[i, to]) * U[i + 1, from]) * occ[from] * (1 - occ[to])
            end
            currents[i] = 2 * p.gamma_plus * forward - 2 * p.gamma_minus * backward
        end
        return currents
    else
        error("unsupported boundary = $(result.boundary)")
    end
end

function current_expectation(result::SteadyStateResult, p::ModelParams)
    currents = bond_currents(result, p)
    return sum(currents) / length(currents)
end

function steady_state_residual(result::SteadyStateResult, p::ModelParams)
    W = thermal_rates(result.energies, p.alpha, p.beta)
    injections = zeros(Float64, length(result.occupations))
    losses = zeros(Float64, length(result.occupations))

    if result.boundary == :pbc
        isnothing(result.mode_labels) && error("PBC results must store mode labels")
        ks = result.mode_labels
        gain, loss = drive_rates_pbc(p, ks)
        injections .+= gain
        losses .+= loss
    elseif result.boundary == :obc
        W .+= drive_rates_obc(p, result.basis_transform)
    else
        error("unsupported boundary = $(result.boundary)")
    end

    rhs = occupation_rhs(result.occupations, W; injection = injections, loss = losses)
    return norm(rhs, Inf)
end
