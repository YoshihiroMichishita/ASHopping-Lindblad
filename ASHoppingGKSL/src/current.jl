#include("ops_space.jl")
#include("bath.jl")

function current_operator(p::SystemParams)
    c1, c2, cd1, cd2 = fermion_ops_2site()
    return im * p.t * (cd1 * c2 - cd2 * c1)
end

function current_operator_LS(p::SystemParams, H_LS::Matrix{C64})
    c1, c2, cd1, cd2 = fermion_ops_2site()
    r = 0.5*(cd1*c1 - cd2*c2)
    H = system_hamiltonian(p)
    H_S = H + H_LS
    j = im*(H_S * r - r * H_S)
    return j
end

function current_expectation(rho::Matrix{C64}, p::SystemParams)
    Jop = current_operator(p)
    return real(tr(Jop * rho))
end

function current_expectation(rho::Matrix{C64}, p::SystemParams, H_LS::Matrix{C64})
    Jop = current_operator_LS(p, H_LS)
    return real(tr(Jop * rho))
end