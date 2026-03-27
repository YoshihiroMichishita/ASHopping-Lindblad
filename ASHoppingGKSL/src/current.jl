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
#=
function current_operator(p::SystemParams)
    c1, c2, cd1, cd2 = fermion_ops_2site()
    return im * p.t * (cd1 * c2 - cd2 * c1)
end=#

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
#=
function current_operator_LS(p::SystemParams, H_LS::Matrix{C64})
    c1, c2, cd1, cd2 = fermion_ops_2site()
    r = 0.5*(cd1*c1 - cd2*c2)
    _, H = system_hamiltonian(p)
    H_S = H + H_LS
    j = im*(H_S * r - r * H_S)
    return j
end=#

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