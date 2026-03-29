#include("ops_space.jl")
#include("bath.jl")

function current_operator(p::SystemParams; ind_leftsite::Int=1)
    _, H = system_hamiltonian(p)
    n_ops, _ = local_number_projectors(p)
    n_left = sum(n_ops[i] for i in 1:ind_leftsite)
    n_right = sum(n_ops[i] for i in (ind_leftsite+1):p.n_sites)
    pos = 0.5 * (n_left - n_right)
    #pos = 0.5 * (n_ops[ind_leftsite] - n_ops[ind_leftsite+1])
    return im * (H * pos - pos * H)
end

function current_operator_LS(p::SystemParams, H_LS::Matrix{C64}; ind_leftsite::Int=1)
    _, H = system_hamiltonian(p)
    n_ops, _ = local_number_projectors(p)
    n_left = sum(n_ops[i] for i in 1:ind_leftsite)
    n_right = sum(n_ops[i] for i in (ind_leftsite+1):p.n_sites)
    pos = 0.5 * (n_left - n_right)
    #pos = 0.5 * (n_ops[ind_leftsite] - n_ops[ind_leftsite+1])
    H_S = H + H_LS
    return im * (H_S * pos - pos * H_S)
end

function current_expectation(rho::Matrix{C64}, p::SystemParams; ind_leftsite::Int=1)
    Jop = current_operator(p; ind_leftsite=ind_leftsite)
    return real(tr(Jop * rho))
end

function current_expectation_sites(rho::Matrix{C64}, p::SystemParams)
    return [current_expectation(rho, p; ind_leftsite=ind_leftsite) for ind_leftsite in 1:(p.n_sites-1)]
end

function current_expectation(rho::Matrix{C64}, p::SystemParams, H_LS::Matrix{C64}; ind_leftsite::Int=1)
    Jop = current_operator_LS(p, H_LS; ind_leftsite=ind_leftsite)
    return real(tr(Jop * rho))
end

function current_expectation_sites(rho::Matrix{C64}, p::SystemParams, H_LS::Matrix{C64})
    return [current_expectation(rho, p, H_LS; ind_leftsite=ind_leftsite) for ind_leftsite in 1:(p.n_sites-1)]
end