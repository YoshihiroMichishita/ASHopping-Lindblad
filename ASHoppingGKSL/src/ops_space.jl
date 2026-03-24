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
        n1 = cd1 * c1
        n2 = cd2 * c2
        # Match main.tex convention: H_S = -t(c1†c2 + c2†c1) - (omega/2)(n1 - n2)
        return -p.t * (cd1 * c2 + cd2 * c1) - 0.5 * p.omega * (n1 - n2)
    elseif p.n_sites == 3
        c1, c2, c3, cd1, cd2, cd3 = fermion_ops_3site()

        n1 = cd1 * c1
        n2 = cd2 * c2
        n3 = cd3 * c3

        H_hop = -p.t * (
            cd1 * c2 + cd2 * c1 +
            cd2 * c3 + cd3 * c2
        )

        H_pot = -p.omega * n1 + 0.0 * n2 + p.omega * n3

        return H_hop + H_pot
    else
        error("n_sites must be 2 or 3")
    end
end

function system_hamiltonian_3site(p::SystemParams)
    c1, c2, c3, cd1, cd2, cd3 = fermion_ops_3site()

    n1 = cd1 * c1
    n2 = cd2 * c2
    n3 = cd3 * c3

    H_hop = -p.t * (
        cd1 * c2 + cd2 * c1 +
        cd2 * c3 + cd3 * c2
    )

    H_pot = -p.omega * n1 + 0.0 * n2 + p.omega * n3

    return H_hop + H_pot
end

# Diagonalize the single-particle Hamiltonian.
# U[:,a] is the a-th eigenmode in the site basis.
function single_particle_eigensystem(p::SystemParams)
    if p.n_sites == 2
        c1, c2, _, _ = fermion_ops_2site()
        h = system_hamiltonian(p)
    elseif p.n_sites == 3
        c1, c2, c3, _, _, _ = fermion_ops_3site()
        h = system_hamiltonian_3site(p)
    else
        error("n_sites must be 2 or 3")
    end
    eig = eigen(Hermitian(h))

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
    return round(w / tol) * tol
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

        w_ann = bucket_frequency(-e; tol=tol)  # [H, d] = -e d
        w_cre = bucket_frequency(+e; tol=tol)  # [H, d†] = +e d†

        ops[(:alpha, w_ann)] += (im / sqrt(2)) * u * d
        ops[(:alpha, w_cre)] += (-im / sqrt(2)) * conj(u) * dd

        ops[(:beta,  w_ann)] += (1.0 / sqrt(2)) * u * d
        ops[(:beta,  w_cre)] += (1.0 / sqrt(2)) * conj(u) * dd
    end

    return ops
end

#=
function mixing_angles(p::SystemParams)
    r = sqrt(p.t^2 + (p.omega / 2)^2)
    den = sqrt((r + p.omega / 2)^2 + p.t^2)
    c = (r + p.omega / 2) / den
    s = p.t / den
    return r, c, s
end

function normal_modes(p::SystemParams)
    c1, c2, _, _ = fermion_ops_2site()
    _, c, s = mixing_angles(p)
    # Match main.tex rotation: [c+; c-] = [c -s; -s -c] [c1; c2]
    cplus = c * c1 - s * c2
    cminus = -s * c1 - c * c2
    return cplus, cminus
end

function local_to_eigen_coeff(site::Int, c::Float64, s::Float64)
    if site == 1
        return c, -s
    elseif site == 2
        return -s, -c
    else
        error("site must be 1 or 2")
    end
end

"""
Decompose Hermitian system operators (eqs. 62-64, 87-90) into Bohr frequencies ±r.
Returns Dict((p, sign) => c_p[sign*r]), p in (:alpha, :beta), sign in (+1, -1).
"""
function coupling_ops_by_frequency(p::SystemParams, site::Int)
    cplus, cminus = normal_modes(p)
    _, c, s = mixing_angles(p)
    a, b = local_to_eigen_coeff(site, c, s)

    # For ci = a c+ + b c-, positive-frequency part carries c+ and c-†.
    # c_alpha = i(ci - ci†)/√2, c_beta = (ci† + ci)/√2.
    c_alpha_plus = im * (a * cplus - b * cminus') / sqrt(2)
    c_beta_plus = (a * cplus + b * cminus') / sqrt(2)

    ops = Dict{Tuple{Symbol, Int}, Matrix{C64}}()
    ops[(:alpha, +1)] = c_alpha_plus
    ops[(:alpha, -1)] = c_alpha_plus'
    ops[(:beta, +1)] = c_beta_plus
    ops[(:beta, -1)] = c_beta_plus'
    return ops
end=#

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