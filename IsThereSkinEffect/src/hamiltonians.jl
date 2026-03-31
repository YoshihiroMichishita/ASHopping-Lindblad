function tbar(p::ModelParams)
    return p.t * cosh(p.J)
end

function ta(p::ModelParams)
    return p.t * sinh(p.J)
end

function single_particle_hamiltonian(p::ModelParams, ::PBC)
    h = single_particle_hamiltonian(p, OBC())
    h[1, p.L] = -tbar(p)
    h[p.L, 1] = -tbar(p)
    return h
end

function single_particle_hamiltonian(p::ModelParams, ::OBC)
    H = zeros(ComplexF64, p.L, p.L)
    hopping = -tbar(p)
    for i in 1:(p.L - 1)
        H[i + 1, i] += hopping
        H[i, i + 1] += hopping
    end
    return H
end
