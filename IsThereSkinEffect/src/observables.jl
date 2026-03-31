function site_density(C::AbstractMatrix{<:Complex})
    return real.(diag(C))
end

function bond_currents(C::AbstractMatrix{<:Complex}, p::ModelParams, bc::BoundaryCondition)
    n_bonds = bc isa PBC ? p.L : (p.L - 1)
    currents = zeros(Float64, n_bonds)
    for i in 1:n_bonds
        j = bc isa PBC && i == p.L ? 1 : i + 1
        currents[i] = 2 * tbar(p) * imag(C[i, j])
    end
    return currents
end

function bond_currents(result::SteadyStateResult)
    return result.currents
end

function current_expectation(result::SteadyStateResult)
    return sum(result.currents) / length(result.currents)
end

function dissipative_site_net(
    C::AbstractMatrix{<:Complex},
    gamma_gain::AbstractMatrix{<:Complex},
    gamma_loss::AbstractMatrix{<:Complex},
)
    gamma_total = gamma_gain + gamma_loss
    dC_diss = 2 * gamma_gain - (gamma_total * C + C * gamma_total)
    return real.(diag(dC_diss))
end

function dissipative_site_net(result::SteadyStateResult)
    return dissipative_site_net(result.correlation, result.gamma_gain, result.gamma_loss)
end

function continuity_residuals(net::AbstractVector{<:Real}, currents::AbstractVector{<:Real}, bc::BoundaryCondition)
    L = length(net)
    residuals = zeros(Float64, L)
    if bc isa PBC
        for i in 1:L
            left = i == 1 ? currents[end] : currents[i - 1]
            right = currents[i]
            residuals[i] = net[i] - left + right
        end
    else
        residuals[1] = net[1] + currents[1]
        for i in 2:(L - 1)
            residuals[i] = net[i] - currents[i - 1] + currents[i]
        end
        residuals[L] = net[L] - currents[end]
    end
    return residuals
end

function continuity_residuals(result::SteadyStateResult)
    bc = result.boundary == :pbc ? PBC() : OBC()
    net = dissipative_site_net(result)
    return continuity_residuals(net, result.currents, bc)
end

function steady_state_residual(
    C::AbstractMatrix{<:Complex},
    h::AbstractMatrix{<:Complex},
    gamma_gain::AbstractMatrix{<:Complex},
    gamma_loss::AbstractMatrix{<:Complex},
)
    rhs = covariance_rhs(C, h, gamma_gain, gamma_loss)
    return norm(rhs, Inf)
end

function steady_state_residual(result::SteadyStateResult)
    return result.residual
end
