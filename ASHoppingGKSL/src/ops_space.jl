using LinearAlgebra
using Random

const C64 = ComplexF64

struct SystemParams
    t::Float64
    omega::Float64
end



"""
Two-site spinless fermion operators in the explicit Fock basis order
[|00>, |01>, |10>, |11>].
"""
function fermion_ops_2site()
    I2 = Matrix{C64}(I, 2, 2)
    # Local basis is [|0>, |1>].
    c = C64[0 1; 0 0]
    sigma_z = C64[1 0; 0 -1]
    #c1 = kron(c, I2)
    c1 = kron(I2, c)
    #c2 = kron(sigma_z, c) # Jordan-Wigner string on site 1
    c2 = kron(c, sigma_z)
    return c1, c2, c1', c2'
end

function system_hamiltonian(p::SystemParams)
    c1, c2, cd1, cd2 = fermion_ops_2site()
    n1 = cd1 * c1
    n2 = cd2 * c2
    # Match main.tex convention: H_S = -t(c1†c2 + c2†c1) - (omega/2)(n1 - n2)
    return -p.t * (cd1 * c2 + cd2 * c1) - 0.5 * p.omega * (n1 - n2)
end

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