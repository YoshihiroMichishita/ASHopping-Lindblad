using LinearAlgebra
using Random

const C64 = ComplexF64

struct SystemParams
    t::Float64
    omega::Float64
end

struct BathSpectrum
    Sigma::Function      # Σ(ω)
    SigmaTilde::Function # \tilde{Σ}(ω)
end

struct TwoBathModel
    sys::SystemParams
    bath1::BathSpectrum
    bath2::BathSpectrum
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

"""
Bath correlation Φ_pq(ω) from eqs. (72)-(73).
Sign convention for off-diagonal part follows Φ_{βα} = -Φ_{αβ}.
"""
function Phi_from_sigma(b::BathSpectrum, p::Symbol, q::Symbol, w::Float64)
    s1 = b.Sigma(w)
    s2 = conj(b.SigmaTilde(-w))
    if p == q
        # eqs. (70),(72): Φ_{αα}=Φ_{ββ}= (i/2)(Σ - \tilde{Σ}^\dagger)
        return 0.5im * (s1 - s2)
    elseif p == :alpha && q == :beta
        # eqs. (66),(73): Φ_{αβ}= (1/2)(Σ + \tilde{Σ}^\dagger)
        return 0.5 * (s1 + s2)
    elseif p == :beta && q == :alpha
        return -0.5 * (s1 + s2)
    else
        error("p,q must be :alpha or :beta")
    end
end

"""
Compute gamma_pq(ω) and eta_pq(ω) from Φ_pq(ω).
Uses relations: γ_pq = Φ_pq + Φ_qp* , η_pq = -(i/2)(Φ_pq - Φ_qp*).
"""
function gamma_eta_from_phi(b::BathSpectrum, p::Symbol, q::Symbol, w::Float64)
    phi_pq = Phi_from_sigma(b, p, q, w)
    phi_qp = Phi_from_sigma(b, q, p, w)
    gamma = phi_pq + conj(phi_qp)
    eta = -0.5im * (phi_pq - conj(phi_qp))
    return gamma, eta
end

"""
Redfield Liouvillian pieces (eq. 95): dissipator superoperator and Lamb shift H_LS.
Returns (L_diss, H_LS).
"""
function redfield_parts(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    pq = (:alpha, :beta)
    L_diss = zeros(C64, d^2, d^2)
    H_LS = zeros(C64, d, d)

    for (site, bath) in ((1, m.bath1), (2, m.bath2))
        ops = coupling_ops_by_frequency(m.sys, site)
        for p in pq, q in pq, sgn_w in (+1, -1), sgn_wp in (+1, -1)
            w = sgn_w * r
            wp = sgn_wp * r
            Ap = ops[(p, sgn_w)]   # c_p[w]
            Aq = ops[(q, sgn_wp)]  # c_q[w']
            pref = (p == q) ? -1.0 : 1.0  # -(1 - 2δ_pq)

            gamma_w, eta_w = gamma_eta_from_phi(bath, p, q, w)
            gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)

            Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)
            coeff_ls = 0.5 * (eta_w + eta_wp) + 0.25im * (gamma_w - gamma_wp)

            # dissipator term: Aq ρ Ap† - 1/2 {Ap† Aq, ρ}
            #L_diss += pref * Gamma *
            L_diss += -2pref * Gamma *
                    (spost(Ap', d) * spre(Aq, d) -
                       0.5 * spre(Ap' * Aq, d) - 0.5 * spost(Ap' * Aq, d))

            # Lamb shift term
            H_LS += pref * coeff_ls * (Ap' * Aq)
        end
    end

    H_LS = 0.5 * (H_LS + H_LS')
    return L_diss, H_LS
end

"""
Redfield Liouvillian pieces (eq. 95): dissipator superoperator and Lamb shift H_LS.
Returns (L_diss, H_LS).
"""
function redfield_parts(m::TwoBathModel, ind_bath::Int)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    pq = (:alpha, :beta)
    L_diss = zeros(C64, d^2, d^2)
    H_LS = zeros(C64, d, d)
    bath = (ind_bath == 1) ? m.bath1 : m.bath2
    site = ind_bath
    #for (site, bath) in ((1, m.bath1), (2, m.bath2))
    ops = coupling_ops_by_frequency(m.sys, site)
    for p in pq, q in pq, sgn_w in (+1, -1), sgn_wp in (+1, -1)
        w = sgn_w * r
        wp = sgn_wp * r
        Ap = ops[(p, sgn_w)]   # c_p[w]
        Aq = ops[(q, sgn_wp)]  # c_q[w']
        pref = (p == q) ? -1.0 : 1.0  # (1 - 2δ_pq)

        gamma_w, eta_w = gamma_eta_from_phi(bath, p, q, w)
        gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)

        Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)
        coeff_ls = 0.5 * (eta_w + eta_wp) + 0.25im * (gamma_w - gamma_wp)

        # dissipator term: Aq ρ Ap† - 1/2 {Ap† Aq, ρ}
        #L_diss += pref * Gamma *
        L_diss += -2pref * Gamma *
                    (spost(Ap', d) * spre(Aq, d) -
                    0.5 * spre(Ap' * Aq, d) - 0.5 * spost(Ap' * Aq, d))

        # Lamb shift term
        H_LS += pref * coeff_ls * (Ap' * Aq)
    end
    #end

    H_LS = 0.5 * (H_LS + H_LS')
    return L_diss, H_LS
end

function normalgksl_parts(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    pq = (:alpha, :beta)
    L_diss = zeros(C64, d^2, d^2)
    H_LS = zeros(C64, d, d)

    for (site, bath) in ((1, m.bath1), (2, m.bath2))
        ops = coupling_ops_by_frequency(m.sys, site)
        for p in pq, q in pq, sgn_w in (+1, -1)#, sgn_wp in (+1, -1)
            sgn_wp = sgn_w
            w = sgn_w * r
            wp = sgn_wp * r
            Ap = ops[(p, sgn_w)]   # c_p[w]
            Aq = ops[(q, sgn_wp)]  # c_q[w']
            pref = (p == q) ? -1.0 : 1.0  # (1 - 2δ_pq)

            gamma_w, eta_w = gamma_eta_from_phi(bath, p, q, w)
            gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)

            Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)
            coeff_ls = 0.5 * (eta_w + eta_wp) + 0.25im * (gamma_w - gamma_wp)

            # dissipator term: Aq ρ Ap† - 1/2 {Ap† Aq, ρ}
            #L_diss += pref * Gamma *
            L_diss += -2pref * Gamma *
                    (spost(Ap', d) * spre(Aq, d) -
                       0.5 * spre(Ap' * Aq, d) - 0.5 * spost(Ap' * Aq, d))

            # Lamb shift term
            H_LS += pref * coeff_ls * (Ap' * Aq)
        end
    end

    H_LS = 0.5 * (H_LS + H_LS')
    return L_diss, H_LS
end

function normalgksl_parts(m::TwoBathModel, ind_bath::Int)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    pq = (:alpha, :beta)
    L_diss = zeros(C64, d^2, d^2)
    H_LS = zeros(C64, d, d)

    bath = (ind_bath == 1) ? m.bath1 : m.bath2
    site = ind_bath
    ops = coupling_ops_by_frequency(m.sys, site)
    for p in pq, q in pq, sgn_w in (+1, -1)#, sgn_wp in (+1, -1)
        sgn_wp = sgn_w
        w = sgn_w * r
        wp = sgn_wp * r
        Ap = ops[(p, sgn_w)]   # c_p[w]
        Aq = ops[(q, sgn_wp)]  # c_q[w']
        pref = (p == q) ? -1.0 : 1.0  # (1 - 2δ_pq)

        gamma_w, eta_w = gamma_eta_from_phi(bath, p, q, w)
        gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)

        Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)
        coeff_ls = 0.5 * (eta_w + eta_wp) + 0.25im * (gamma_w - gamma_wp)

        # dissipator term: Aq ρ Ap† - 1/2 {Ap† Aq, ρ}
        #L_diss += pref * Gamma *
        L_diss += -2pref * Gamma *
                (spost(Ap', d) * spre(Aq, d) -
                    0.5 * spre(Ap' * Aq, d) - 0.5 * spost(Ap' * Aq, d))

        # Lamb shift term
        H_LS += pref * coeff_ls * (Ap' * Aq)
    end

    H_LS = 0.5 * (H_LS + H_LS')
    return L_diss, H_LS
end

"""
Redfield Liouvillian (eq. 95) for the two-bath model.
"""
function redfield_liouvillian(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    L_diss, H_LS = redfield_parts(m)
    return commutator_super(H + H_LS) + L_diss
end

function normalgksl_liouvillian(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    L_diss, H_LS = normalgksl_parts(m)
    return commutator_super(H + H_LS) + L_diss
end

function fermi(e::Float64, mu::Float64, beta::Float64)
    return 1.0 / (exp(beta * (e - mu)) + 1.0)
end

"""
Example finite-correlation bath self-energy model.
"""
# mu is chemical potential
function make_bath(; Gamma::Float64, mu::Float64, beta::Float64, tau_c::Float64)
    Sigma = (w::Float64) -> begin
        f = fermi(w, mu, beta)
        -im * Gamma * (1 - f) / (1 - im * w * tau_c)
    end
    SigmaTilde = (w::Float64) -> begin
        f = fermi(w, mu, beta)
        -im * Gamma * f / (1 - im * w * tau_c)
    end
    return BathSpectrum(Sigma, SigmaTilde)
end

# mu is energy potential
function make_bath2(; Gamma::Float64, mu::Float64, beta::Float64, tau_c::Float64)
    Sigma = (w::Float64) -> begin
        f = fermi(w, 0.0, beta)
        -im * Gamma * (1 - f) / (1 - im * (w - mu) * tau_c)
    end
    SigmaTilde = (w::Float64) -> begin
        f = fermi(w, 0.0, beta)
        -im * Gamma * f / (1 - im * (w - mu) * tau_c)
    end
    return BathSpectrum(Sigma, SigmaTilde)
end


"""
γ-matrix square root (eqs. 84-86, 100) at frequency w.
"""
function gammahat_sqrt(b::BathSpectrum, w::Float64)
    sw = b.Sigma(w)
    #st = conj(b.SigmaTilde(-w))
    st = b.SigmaTilde(-w)

    gamma = imag(sw + st)     # eq. (82)
    gamma_p = imag(sw - st)   # eq. (83)

    #gamma_m = gamma + gamma_p
    #gamma_s = gamma - gamma_p

    g = [gamma 1.0im*gamma_p; -1.0im*gamma_p gamma]
    eigvals, eigvecs = eigen(Hermitian(g))
    sqrt_eigvals = sqrt.(complex.(eigvals))
    sqrt_g = eigvecs * Diagonal(sqrt_eigvals) * eigvecs'
    return sqrt_g

    #gm = sqrt(complex(gamma_m))
    #gm = sqrt(gamma_m)
    #gs = sqrt(complex(gamma_s))
    #gs = sqrt(gamma_s)

    #return (1 / sqrt(2)) * C64[
    #    (gm + gs)         im * (gm - gs)
    #    -im * (gm - gs)   (gm + gs)
    #]
end

"""
Nathan-Rudner type GKSL Liouvillian (eqs. 100-109).
"""
function nr_gksl_liouvillian(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    Hls_total = zeros(C64, d, d)
    pset = (:alpha, :beta)
    pidx = Dict(:alpha => 1, :beta => 2)
    lfac = Dict(:alpha => -im, :beta => im) # l_{α/β} = ∓ i

    # Accumulate Lamb shift from each bath
    for (site, bath) in ((1, m.bath1), (2, m.bath2))
        ops = coupling_ops_by_frequency(m.sys, site)
        Hls = zeros(C64, d, d)
        for p in pset, q in pset, sgn_w in (+1, -1), sgn_wp in (+1, -1)
            w = sgn_w * r
            wp = sgn_wp * r
            Ap = ops[(p, sgn_w)]   # c_p[w]
            Aq = ops[(q, sgn_wp)]  # c_q[w']
            pref = (p == q) ? -1.0 : 1.0  # (1 - 2δ_pq)

            gamma_w, eta_w = gamma_eta_from_phi(bath, p, q, w)
            gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)

            coeff = 0.5 * (eta_w + eta_wp) + 0.25im * (gamma_w - gamma_wp)
            Hls += pref * coeff * (Ap' * Aq)
        end
        Hls = 0.5 * (Hls + Hls')
        Hls_total += Hls
    end

    L = commutator_super(H + Hls_total)

    # Dissipator (Nathan-Rudner GKSL)
    for (site, bath) in ((1, m.bath1), (2, m.bath2))
        ops = coupling_ops_by_frequency(m.sys, site)
        jumps = Matrix{C64}[]
        for p in pset
            J = zeros(C64, d, d)
            for sgn in (+1, -1)
                sqg = gammahat_sqrt(bath, sgn * r)
                for q in pset
                    J += sqg[pidx[p], pidx[q]] * lfac[q] * ops[(q, sgn)]
                end
            end
            push!(jumps, J)
        end
        for J in jumps
            L += lindblad_super(J)
        end
    end
    return L
end

function nr_gksl_liouvillian_diss(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    Hls_total = zeros(C64, d, d)
    pset = (:alpha, :beta)
    pidx = Dict(:alpha => 1, :beta => 2)
    lfac = Dict(:alpha => -im, :beta => im) # l_{α/β} = ∓ i


    L = [zeros(C64, size(commutator_super(H))...), zeros(C64, size(commutator_super(H))...)]
    
    # Dissipator (Nathan-Rudner GKSL)
    for (site, bath) in ((1, m.bath1), (2, m.bath2))
        ops = coupling_ops_by_frequency(m.sys, site)
        jumps = Matrix{C64}[]
        for p in pset
            J = zeros(C64, d, d)
            for sgn in (+1, -1)
                sqg = gammahat_sqrt(bath, sgn * r)
                for q in pset
                    J += sqg[pidx[p], pidx[q]] * lfac[q] * ops[(q, sgn)]
                end
            end
            push!(jumps, J)
        end
        for J in jumps
            L[site] += lindblad_super(J)
        end
    end
    return L
end

function steady_state_from_liouvillian(L::Matrix{C64}; d::Int)
    I_d = Matrix{C64}(I, d, d)
    tr_row = reshape(vec(transpose(I_d)), 1, :)
    A = [L; tr_row]
    b = zeros(C64, size(L, 1) + 1)
    b[end] = 1 + 0im
    rho_vec = A \ b
    rho = reshape(rho_vec, d, d)
    rho = 0.5 * (rho + rho')
    rho ./= real(tr(rho))
    return rho
end

function steady_states(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    Lr = redfield_liouvillian(m)
    Lg = nr_gksl_liouvillian(m)
    return steady_state_from_liouvillian(Lr; d), steady_state_from_liouvillian(Lg; d), H
end

function steady_states3(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    Lr = redfield_liouvillian(m)
    Lng = normalgksl_liouvillian(m)
    Lg = nr_gksl_liouvillian(m)
    return steady_state_from_liouvillian(Lr; d), steady_state_from_liouvillian(Lng; d), steady_state_from_liouvillian(Lg; d), H
end


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

function nr_jumps_by_site(m::TwoBathModel)
    r, _, _ = mixing_angles(m.sys)
    pset = (:alpha, :beta)
    pidx = Dict(:alpha => 1, :beta => 2)
    lfac = Dict(:alpha => -im, :beta => im) # l_{α/β} = ∓ i

    jumps_by_site = Dict{Int, Vector{Matrix{C64}}}()
    for (site, bath) in ((1, m.bath1), (2, m.bath2))
        ops = coupling_ops_by_frequency(m.sys, site)
        jumps = Matrix{C64}[]
        for p in pset
            J = zeros(C64, size(ops[(p, +1)]))
            for sgn in (+1, -1)
                sqg = gammahat_sqrt(bath, sgn * r)
                for q in pset
                    J += sqg[pidx[p], pidx[q]] * lfac[q] * ops[(q, sgn)]
                end
            end
            push!(jumps, J)
        end
        jumps_by_site[site] = jumps
    end
    return jumps_by_site
end

function redfield_gain_loss(m::TwoBathModel, bathind::Int, rho::Matrix{C64})
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    pq = (:alpha, :beta)
    Loss_site = [0.0, 0.0]
    Gain_site = [0.0, 0.0]

    c1, c2, cd1, cd2 = fermion_ops_2site()
    loss_ops = Dict(1 => c1, 2 => c2)
    gain_ops = Dict(1 => cd1, 2 => cd2)

    mybath = (bathind == 1) ? m.bath1 : m.bath2

    ops = coupling_ops_by_frequency(m.sys, bathind)
    for p in pq, q in pq, sgn_w in (+1, -1), sgn_wp in (+1, -1)
        w = sgn_w * r
        wp = sgn_wp * r
        Ap = ops[(p, sgn_w)]   # c_p[w]
        Aq = ops[(q, sgn_wp)]  # c_q[w']
        pref = (p == q) ? 1.0 : -1.0  # -(1 - 2δ_pq)

        gamma_w, eta_w = gamma_eta_from_phi(mybath, p, q, w)
        gamma_wp, eta_wp = gamma_eta_from_phi(mybath, p, q, wp)

        Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)

        # dissipator term: Aq ρ Ap† - 1/2 {Ap† Aq, ρ}
        for site in (1, 2)
            loss_op = loss_ops[site]
            gain_op = gain_ops[site]
            Aq_gain = Aq .* abs.(gain_op)
            Ap_gain = Ap .* abs.(gain_op)
            Aq_loss = Aq .* abs.(loss_op)
            Ap_loss = Ap .* abs.(loss_op)
            Gain_site[site] += 2real(pref * Gamma * tr(Aq_gain * rho * Ap_gain'))
            Loss_site[site] += 2real(pref * Gamma * tr(Aq_loss * rho * Ap_loss'))
        end
    end
    return Gain_site, Loss_site
end

function site_gain_loss_bath(m::TwoBathModel, rho::Matrix{C64}; method::Symbol = :gksl)
    c1, c2, cd1, cd2 = fermion_ops_2site()
    loss_ops = Dict(1 => c1, 2 => c2)
    gain_ops = Dict(1 => cd1, 2 => cd2)
    results = Dict{Tuple{Int, Int}, Tuple{Float64, Float64, Float64}}()

    # Nathan--Rudner GKSL: sum over jumps for each site and bath
    if method == :gksl
        jumps_by_site = nr_jumps_by_site(m)
        for bath in (1, 2), site in (1, 2)
            gain = 0.0
            loss = 0.0
            for J in jumps_by_site[bath]
                loss_J = J .* abs.(loss_ops[site])
                gain_J = J .* abs.(gain_ops[site])
                gain += 2real(tr(gain_J * rho * gain_J'))
                loss += 2real(tr(loss_J * rho * loss_J'))
            end
            results[(bath, site)] = (gain, loss, gain - loss)
        end
        return results
    elseif method == :redfield
        for bath in (1, 2)
            gain_site, loss_site = redfield_gain_loss(m, bath, rho)
            for site in (1, 2)
                results[(bath, site)] = (gain_site[site], loss_site[site], gain_site[site] - loss_site[site])
            end
        end
        return results
    else
        error("method must be :gksl or :redfield")
    end
end

function test_gain_loss_zero_equilibrium(;
    beta::Float64 = 8.0,
    Gamma::Float64 = 0.15,
    tau_c::Float64 = 0.2,
    mu::Float64 = 0.0,
    omega::Float64 = 0.0,
    tol::Float64 = 1e-10,
    verbose::Bool = true
)
    sys = SystemParams(1.0, omega)
    bathL = make_bath(Gamma = Gamma, mu = -mu, beta = beta, tau_c = tau_c)
    bathR = make_bath(Gamma = Gamma, mu = mu, beta = beta, tau_c = tau_c)
    model = TwoBathModel(sys, bathL, bathR)

    rho_red, rho_gksl, _ = steady_states(model)

    gl_red = site_gain_loss_bath(model, rho_red; method = :redfield)
    gl_gksl = site_gain_loss_bath(model, rho_gksl; method = :gksl)

    ok_red = all(abs(gl_red[(bath,site)][3]) < tol for site in (1, 2), bath in (1,2))
    ok_gksl = all(abs(gl_gksl[(bath,site)][3]) < tol for site in (1, 2), bath in (1,2))
    ok = ok_red && ok_gksl

    if verbose
        println("Gain-Loss net flow test (muL=$( -mu ), muR=$(mu), omega=$(omega))")
        println("tolerance = $(tol)")
        println("Redfield gain: (bath, site); (1,1)=$(gl_red[(1,1)][1]), (1,2)=$(gl_red[(1,2)][1]), (2,1)=$(gl_red[(2,1)][1]), (2,2)=$(gl_red[(2,2)][1])")
        println("GKSL gain:    (bath, site); (1,1)=$(gl_gksl[(1,1)][1]), (1,2)=$(gl_gksl[(1,2)][1]), (2,1)=$(gl_gksl[(2,1)][1]), (2,2)=$(gl_gksl[(2,2)][1])")
        println("Redfield loss: (bath, site); (1,1)=$(gl_red[(1,1)][2]), (1,2)=$(gl_red[(1,2)][2]), (2,1)=$(gl_red[(2,1)][2]), (2,2)=$(gl_red[(2,2)][2])")
        println("GKSL loss:    (bath, site); (1,1)=$(gl_gksl[(1,1)][2]), (1,2)=$(gl_gksl[(1,2)][2]), (2,1)=$(gl_gksl[(2,1)][2]), (2,2)=$(gl_gksl[(2,2)][2])")
        println("Redfield net: (bath, site); (1,1)=$(gl_red[(1,1)][3]), (1,2)=$(gl_red[(1,2)][3]), (2,1)=$(gl_red[(2,1)][3]), (2,2)=$(gl_red[(2,2)][3]) => pass=$(ok_red)")
        println("GKSL net:    (bath, site); (1,1)=$(gl_gksl[(1,1)][3]), (1,2)=$(gl_gksl[(1,2)][3]), (2,1)=$(gl_gksl[(2,1)][3]), (2,2)=$(gl_gksl[(2,2)][3]) => pass=$(ok_gksl)")
        println("all pass = $(ok)")
    end

    return ok, gl_red, gl_gksl
end

function test_current_zero_and_finite2(;
    beta::Float64 = 8.0,
    omega::Float64 = 0.0, 
    Gamma::Float64 = 0.15,
    tau_c::Float64 = 0.2,
    mu_eq::Float64 = 0.0,
    muL_bias::Float64 = 0.6,
    muR_bias::Float64 = -0.6,
    zero_tol::Float64 = 1e-10,
    finite_tol::Float64 = 1e-4,
    verbose::Bool = true
)
    # Case 1: equilibrium (omega=0, muL=muR=0) -> current should vanish.
    sys_eq = SystemParams(1.0, omega)
    bathL_eq = make_bath(Gamma = Gamma, mu = mu_eq, beta = beta, tau_c = tau_c)
    bathR_eq = make_bath(Gamma = Gamma, mu = mu_eq, beta = beta, tau_c = tau_c)
    model_eq = TwoBathModel(sys_eq, bathL_eq, bathR_eq)
    rho_red_eq, rho_ngksl_eq, rho_gksl_eq, _ = steady_states3(model_eq)
    #rho_red_eq, rho_gksl_eq, _ = steady_states(model_eq)
    L_diss_eq, H_LS_eq = redfield_parts(model_eq)
    _, H_LS_gksl_eq = normalgksl_parts(model_eq)
    j_eq = current_expectation(rho_red_eq, sys_eq, H_LS_eq)
    j_eq_ngksl = current_expectation(rho_ngksl_eq, sys_eq, H_LS_gksl_eq)
    j_eq_gksl = current_expectation(rho_gksl_eq, sys_eq, H_LS_eq)
    #j_eq_gksl = current_expectation(rho_gksl_eq, sys_eq, H_LS_gksl_eq)
    #j_eq = current_expectation(rho_red_eq, sys_eq)
    #j_eq_gksl = current_expectation(rho_gksl_eq, sys_eq)
    ok_zero = abs(j_eq) < zero_tol
    ok_zero_gksl = abs(j_eq_gksl) < zero_tol

    # Case 2: biased chemical potentials (Redfield) -> finite current.
    sys_bias = SystemParams(1.0, omega)
    bathL_bias = make_bath(Gamma = Gamma, mu = muL_bias, beta = beta, tau_c = tau_c)
    bathR_bias = make_bath(Gamma = Gamma, mu = muR_bias, beta = beta, tau_c = tau_c)
    model_bias = TwoBathModel(sys_bias, bathL_bias, bathR_bias)
    #rho_red_bias, rho_gksl_bias, _ = steady_states(model_bias)
    rho_red_bias, rho_ngksl_bias, rho_gksl_bias, _ = steady_states3(model_bias)
    L_diss_bias, H_LS_bias = redfield_parts(model_bias)
    _, H_LS_gksl_bias = normalgksl_parts(model_bias)
    j_bias = current_expectation(rho_red_bias, sys_bias, H_LS_bias)
    j_bias_ngksl = current_expectation(rho_ngksl_bias, sys_bias, H_LS_gksl_bias)
    j_bias_gksl = current_expectation(rho_gksl_bias, sys_bias, H_LS_bias)
    #j_bias_gksl = current_expectation(rho_gksl_bias, sys_bias, H_LS_gksl_bias)
    #j_bias = current_expectation(rho_red_bias, sys_bias)
    #j_bias_gksl = current_expectation(rho_gksl_bias, sys_bias)
    ok_finite = abs(j_bias) > finite_tol
    ok_finite_gksl = abs(j_bias_gksl) > finite_tol

    ok = ok_zero && ok_finite && ok_zero_gksl
    if verbose
        println("Current test with J = i t (c1†c2 - c2†c1)")
        println("equilibrium case: omega=$(omega), muL=muR=$(mu_eq)")
        println("  Redfield current = $(j_eq), |J|<$(zero_tol) => $(ok_zero)")
        println("  Conventional GKSL current = $(j_eq_ngksl), |J|<$(zero_tol) => $(ok_zero)")
        println("  NR GKSL current = $(j_eq_gksl), |J|<$(zero_tol) => $(ok_zero)")
        println("biased case: omega=0, muL=$(muL_bias), muR=$(muR_bias)")
        println("  Redfield current = $(j_bias), |J|>$(finite_tol) => $(ok_finite)")
        println("  Conventional GKSL current = $(j_bias_ngksl), |J|>$(finite_tol) => $(ok_finite)")
        println("  NR GKSL current = $(j_bias_gksl), |J|>$(finite_tol) => $(ok_finite)")
        println("all pass = $(ok)")
    end
    return ok, j_eq, j_bias
end

function main()
    sys = SystemParams(1.0, 0.8)
    bathL = make_bath(Gamma = 0.15, mu = 0.6, beta = 8.0, tau_c = 0.2)
    bathR = make_bath(Gamma = 0.15, mu = -0.6, beta = 8.0, tau_c = 0.2)
    model = TwoBathModel(sys, bathL, bathR)

    rho_red, rho_gksl, H = steady_states(model)

    c1, c2, cd1, cd2 = fermion_ops_2site()
    n1 = real(tr((cd1 * c1) * rho_red))
    n2 = real(tr((cd2 * c2) * rho_red))
    n1g = real(tr((cd1 * c1) * rho_gksl))
    n2g = real(tr((cd2 * c2) * rho_gksl))

    println("=== Steady state (Redfield) ===")
    println(rho_red)
    println("occupations: n1 = $(n1), n2 = $(n2)")
    println()
    println("=== Steady state (Nathan-Rudner GKSL) ===")
    println(rho_gksl)
    println("occupations: n1 = $(n1g), n2 = $(n2g)")
    println()
    println("trace check: Redfield=$(tr(rho_red)), GKSL=$(tr(rho_gksl))")
    println("Hermiticity check: Redfield=$(norm(rho_red-rho_red')), GKSL=$(norm(rho_gksl-rho_gksl'))")
    _ = H
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
