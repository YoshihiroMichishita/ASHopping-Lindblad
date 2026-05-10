function covariance_rhs(
    C::AbstractMatrix{<:Complex},
    h::AbstractMatrix{<:Complex},
    gamma_gain::AbstractMatrix{<:Complex},
    gamma_loss::AbstractMatrix{<:Complex},
)
    gamma_total = gamma_gain + gamma_loss
    return -im * (h * C - C * h) + 2 * gamma_gain - (gamma_total * C + C * gamma_total)
end

function select_dissipation_kernels(
    p::ModelParams,
    bc::BoundaryCondition,
    dissipation::Symbol,
)
    if dissipation == :legacy
        return dissipation_kernels(p, bc)
    elseif dissipation == :eq15
        return dissipation_kernels_new(p, bc)
    end
    throw(ArgumentError("unknown dissipation selector: $(dissipation)"))
end

function steady_covariance(p::ModelParams, bc::BoundaryCondition; dissipation::Symbol = :legacy)
    h = single_particle_hamiltonian(p, bc)
    gamma_gain, gamma_loss = select_dissipation_kernels(p, bc, dissipation)
    gamma_total = gamma_gain + gamma_loss

    L = p.L
    I_L = Matrix{ComplexF64}(I, L, L)
    A = -im * (kron(I_L, h) - kron(transpose(h), I_L))
    A -= kron(I_L, gamma_total) + kron(transpose(gamma_total), I_L)
    b = -vec(2 * gamma_gain)

    c_vec = A \ b
    C = reshape(c_vec, L, L)
    C = 0.5 * (C + C')
    return C, h, gamma_gain, gamma_loss
end

function steady_state_matrix(p::ModelParams, bc::BoundaryCondition; dissipation::Symbol = :legacy)
    C, h, gamma_gain, gamma_loss = steady_covariance(p, bc; dissipation)
    density = site_density(C)
    currents = bond_currents(C, p, bc)
    residual = steady_state_residual(C, h, gamma_gain, gamma_loss)
    boundary = bc isa PBC ? :pbc : :obc
    return SteadyStateResult(boundary, C, h, gamma_gain, gamma_loss, density, currents, residual)
end

function steady_state(p::ModelParams, bc::BoundaryCondition; dissipation::Symbol = :legacy)
    return steady_state_matrix(p, bc; dissipation)
end

function compare_boundaries(p::ModelParams; dissipation::Symbol = :legacy)
    return (; pbc = steady_state(p, PBC(); dissipation), obc = steady_state(p, OBC(); dissipation))
end
