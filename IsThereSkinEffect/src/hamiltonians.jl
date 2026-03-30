function tbar(p::ModelParams)
    return p.t * cosh(p.J)
end

function pbc_momenta(L::Int)
    ks = [2pi * (n - 1) / L for n in 1:L]
    for i in eachindex(ks)
        if ks[i] > pi
            ks[i] -= 2pi
        end
    end
    return sort(ks)
end

function single_particle_hamiltonian(p::ModelParams, ::PBC)
    H = zeros(ComplexF64, p.L, p.L)
    hopping = -tbar(p)
    for i in 1:p.L
        j = i == p.L ? 1 : i + 1
        H[j, i] += hopping
        H[i, j] += hopping
    end
    return H
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

function mode_basis(p::ModelParams, ::PBC)
    ks = pbc_momenta(p.L)
    U = Matrix{ComplexF64}(undef, p.L, p.L)
    for i in 1:p.L
        for a in 1:p.L
            U[i, a] = exp(im * ks[a] * (i - 1)) / sqrt(p.L)
        end
    end
    eps = [-2 * tbar(p) * cos(k) for k in ks]
    return eps, U, ks
end

function mode_basis(p::ModelParams, ::OBC)
    H = single_particle_hamiltonian(p, OBC())
    eig = eigen(Hermitian(H))
    perm = sortperm(real.(eig.values))
    eps = real.(eig.values[perm])
    U = eig.vectors[:, perm]
    return eps, U, nothing
end
