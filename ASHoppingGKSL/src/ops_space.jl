using LinearAlgebra
using Random

const C64 = ComplexF64

struct SystemParams
    n_sites::Int   # number of sites (2 or 3)
    t::Float64       # nearest-neighbor hopping
    omega::Float64   # potential drop between neighboring sites
end



"""
Two-site spinless fermion operators in the explicit Fock basis order
[|00>, |01>, |10>, |11>].
"""
function fermion_ops_2site()
    I2 = Matrix{C64}(I, 2, 2)
    c = C64[0 1; 0 0]
    sigma_z = C64[1 0; 0 -1]

    # basis ordering: |n2 n1>
    c1 = kron(I2, c)
    c2 = kron(c, sigma_z)

    return c1, c2, c1', c2'
end

function fermion_ops_3site()
    I2 = Matrix{C64}(I, 2, 2)
    c = C64[0 1; 0 0]
    sigma_z = C64[1 0; 0 -1]

    # basis ordering: |n3 n2 n1>
    c1 = kron(kron(I2, I2), c)
    c2 = kron(kron(I2, c), sigma_z)
    c3 = kron(kron(c, sigma_z), sigma_z)

    return c1, c2, c3, c1', c2', c3'
end

function system_hamiltonian(p::SystemParams)
    if p.n_sites == 2
        c1, c2, cd1, cd2 = fermion_ops_2site()
        H_matrix = -p.t * [0 1; 1 0] - 0.5 * p.omega * [1 0; 0 -1]
        H_op = [c1, c2]' * H_matrix * [c1, c2]
        return H_matrix, H_op
        # Match main.tex convention: H_S = -t(c1†c2 + c2†c1) - (omega/2)(n1 - n2)
        #return -p.t * (cd1 * c2 + cd2 * c1) - 0.5 * p.omega * (n1 - n2)
    elseif p.n_sites == 3
        c1, c2, c3, cd1, cd2, cd3 = fermion_ops_3site()

        H_matrix = -p.t * [0 1 0; 1 0 1; 0 1 0] - p.omega * diagm([1.0, 0.0, -1.0])
        c_vec = [c1, c2, c3]
        H_op = c_vec' * H_matrix * c_vec
        return H_matrix, H_op
        #=
        n1 = cd1 * c1
        n2 = cd2 * c2
        n3 = cd3 * c3

        H_hop = -p.t * (
            cd1 * c2 + cd2 * c1 +
            cd2 * c3 + cd3 * c2
        )

        H_pot = -p.omega * n1 + 0.0 * n2 + p.omega * n3

        return H_hop + H_pot=#
    else
        error("n_sites must be 2 or 3")
    end
end


function site_ops(p::SystemParams)
    # 3サイト版なら fermion_ops_3site() を呼ぶ
    # 将来は generic fermion_ops_chain(nsites(p)) に差し替えたい
    if p.n_sites == 2
        c1, c2, _, _ = fermion_ops_2site()
        return [c1, c2], [c1', c2']
    elseif p.n_sites == 3
        c1, c2, c3, cd1, cd2, cd3 = fermion_ops_3site()
        c_ops  = [c1, c2, c3]
        cd_ops = [cd1, cd2, cd3]
        return c_ops, cd_ops
    else
        error("n_sites must be 2 or 3")
    end
end

function one_particle_basis(p::SystemParams)
    _, cds = site_ops(p)
    d = size(cds[1], 1)
    vac = zeros(ComplexF64, d)
    vac[1] = 1.0
    return [cd * vac for cd in cds]
end

function one_particle_block(Op, p)
    basis = one_particle_basis(p)
    n = length(basis)
    h = zeros(ComplexF64, n, n)
    for i in 1:n, j in 1:n
        h[i, j] = basis[i]' * Op * basis[j]
    end
    return h
end

# Diagonalize the single-particle Hamiltonian.
# U[:,a] is the a-th eigenmode in the site basis.
function single_particle_eigensystem(p::SystemParams)
    if p.n_sites == 2
        c1, c2, _, _ = fermion_ops_2site()
        h_matrix, h_op = system_hamiltonian(p)
    elseif p.n_sites == 3
        c1, c2, c3, _, _, _ = fermion_ops_3site()
        h_matrix, h_op = system_hamiltonian(p)
    else
        error("n_sites must be 2 or 3")
    end
    eig = eigen(Hermitian(h_matrix))

    eps = real.(eig.values)
    U = eig.vectors

    # optional: sort by eigenvalue
    perm = sortperm(eps)
    eps = eps[perm]
    U = U[:, perm]

    return eps, U
end

# Build annihilation operators d_a for the single-particle normal modes:
# c_i = sum_a U[i,a] d_a
#
# Returns:
#   eps      :: Vector{Float64}
#   U        :: Matrix{ComplexF64}
#   d_modes  :: Vector{Matrix{C64}}
#   dd_modes :: Vector{Matrix{C64}}
function normal_modes(p::SystemParams)
    eps, U = single_particle_eigensystem(p)

    # For future N-site extension, this should eventually call
    # a generic fermion_ops_chain(N), but for now we assume 3 sites.
    if p.n_sites == 2
        c1, c2, _, _ = fermion_ops_2site()
        c_sites = [c1, c2]
    elseif p.n_sites == 3
        c1, c2, c3, cd1, cd2, cd3 = fermion_ops_3site()
        c_sites = [c1, c2, c3]
    else
        error("n_sites must be 2 or 3")
    end

    d_modes = Matrix{C64}[]
    dd_modes = Matrix{C64}[]

    for a in eachindex(eps)
        d = zero(c1)
        for i in 1:length(c_sites)
            # c_i = sum_a U[i,a] d_a  =>  d_a = sum_i conj(U[i,a]) c_i
            d += conj(U[i, a]) * c_sites[i]
        end
        push!(d_modes, d)
        push!(dd_modes, d')
    end

    return eps, U, d_modes, dd_modes
end

# Group nearby frequencies to avoid raw floating-point Dict keys.
function bucket_frequency(w::Float64; tol::Float64 = 1e-10)
    wb = round(w / tol) * tol
    return abs(wb) < tol ? 0.0 : wb
end

# Decompose the local Hermitian bath-coupling operators at site i
# into Bohr-frequency components.
#
# Local operators:
#   c_alpha(i) =  i (c_i - c_i†) / sqrt(2)
#   c_beta(i)  =    (c_i + c_i†) / sqrt(2)
#
# Return type:
#   Dict((channel, omega) => operator)
# where channel in (:alpha, :beta), omega is the Bohr frequency bucket.
function coupling_ops_by_frequency(p::SystemParams, site::Int; tol::Float64 = 1e-10)
    eps, U, d_modes, dd_modes = normal_modes(p)

    ops = Dict{Tuple{Symbol, Float64}, Matrix{C64}}()

    # initialize all frequencies that can appear
    freqs = Float64[]
    for e in eps
        push!(freqs, bucket_frequency(-e; tol=tol))  # annihilation part
        push!(freqs, bucket_frequency(+e; tol=tol))  # creation part
    end
    freqs = unique(sort(freqs))

    d0 = zero(d_modes[1])
    for ch in (:alpha, :beta), w in freqs
        ops[(ch, w)] = zero(d0)
    end

    # c_site = sum_a U[site,a] d_a
    # c_site† = sum_a conj(U[site,a]) d_a†
    for a in eachindex(eps)
        u = U[site, a]
        e = eps[a]
        d = d_modes[a]
        dd = dd_modes[a]

        w_ann = bucket_frequency(+e; tol=tol)
        w_cre = bucket_frequency(-e; tol=tol)

        ops[(:alpha, w_ann)] += (im / sqrt(2)) * u * d
        ops[(:alpha, w_cre)] += (-im / sqrt(2)) * conj(u) * dd

        ops[(:beta,  w_ann)] += (1.0 / sqrt(2)) * u * d
        ops[(:beta,  w_cre)] += (1.0 / sqrt(2)) * conj(u) * dd
    end

    return ops
end


# Vectorization helpers: vec(A ρ B) = (B^T ⊗ A) vec(ρ)
function spre(A::AbstractMatrix{C64}, d::Int)
    return kron(Matrix{C64}(I, d, d), A)
end

function spost(B::AbstractMatrix{C64}, d::Int)
    return kron(transpose(B), Matrix{C64}(I, d, d))
end

function commutator_super(H::Matrix{C64})
    d = size(H, 1)
    return -im * (spre(H, d) - spost(H, d))
end

function lindblad_super(J::Matrix{C64})
    d = size(J, 1)
    JdJ = J' * J
    return 2*(spost(J', d) * spre(J, d) - 0.5 * spre(JdJ, d) - 0.5 * spost(JdJ, d))
end