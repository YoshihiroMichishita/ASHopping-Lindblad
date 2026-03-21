#include("ops_space.jl")
#include("bath.jl")

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


function Phi_from_sigma_TwistedBoundary(b1::BathSpectrum, b2::BathSpectrum, p::Symbol, q::Symbol, w::Float64, t::Float64)
    Δ = b2.μ - b1.μ
    s1 = b2.Sigma(w)
    s2 = conj(b2.SigmaTilde(-w))
    phase = exp(1im * Δ * t)
    if p == q
        # eqs. (70),(72): Φ_{αα}=Φ_{ββ}= (i/2)(Σ - \tilde{Σ}^\dagger)
        return 0.5im * (s1 * phase - s2 * conj(phase))
    elseif p == :alpha && q == :beta
        # eqs. (66),(73): Φ_{αβ}= (1/2)(Σ + \tilde{Σ}^\dagger)
        return 0.5 * (s1 * phase + s2 * conj(phase))
    elseif p == :beta && q == :alpha
        return -0.5 * (s1 * phase + s2 * conj(phase))
    else
        error("p,q must be :alpha or :beta")
    end
end

function Phi_from_sigma_TwistedBoundary(b1::BathSpectrum, b2::BathSpectrum, p::Symbol, q::Symbol, w::Float64)
    Δ = b2.μ - b1.μ
    s1 = b2.Sigma(w+Δ)
    s2 = conj(b2.SigmaTilde(-w+Δ))
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

function gamma_eta_from_phi_TwistedBoundary(b1::BathSpectrum, b2::BathSpectrum, p::Symbol, q::Symbol, w::Float64, t::Float64)
    phi_pq = Phi_from_sigma_TwistedBoundary(b1, b2, p, q, w, t)
    #phi_qp = Phi_from_sigma_TwistedBoundary(b2, b1, q, p, w, t) #TO BE CHECKED
    phi_qp = Phi_from_sigma_TwistedBoundary(b1, b2, q, p, w, t) #TO BE CHECKED
    gamma = phi_pq + conj(phi_qp)
    eta = -0.5im * (phi_pq - conj(phi_qp))
    return gamma, eta
end

function gamma_eta_from_phi_TwistedBoundary(b1::BathSpectrum, b2::BathSpectrum, p::Symbol, q::Symbol, w::Float64)
    phi_pq = Phi_from_sigma_TwistedBoundary(b1, b2, p, q, w)
    #phi_qp = Phi_from_sigma_TwistedBoundary(b2, b1, q, p, w) #TO BE CHECKED
    phi_qp = Phi_from_sigma_TwistedBoundary(b1, b2, q, p, w) #TO BE CHECKED
    gamma = phi_pq + conj(phi_qp)
    eta = -0.5im * (phi_pq - conj(phi_qp))
    return gamma, eta
end

"""
Redfield Liouvillian pieces (eq. 95): dissipator superoperator and Lamb shift H_LS.
Returns (L_diss, H_LS).
"""
function redfield_parts_old(m::TwoBathModel)
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
            L_diss += 2pref * Gamma *
            #L_diss += -2pref * Gamma *
                    (spost(Ap', d) * spre(Aq, d) -
                       0.5 * spre(Ap' * Aq, d) - 0.5 * spost(Ap' * Aq, d))

            # Lamb shift term
            #H_LS += pref * coeff_ls * (Ap' * Aq)
            H_LS += -pref * coeff_ls * (Ap' * Aq)
        end
    end

    H_LS = 0.5 * (H_LS + H_LS')
    return L_diss, H_LS
end

function redfield_parts(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    pq = (:alpha, :beta)
    L_diss = zeros(C64, d^2, d^2)
    H_LS = zeros(C64, d, d)

    ops = [coupling_ops_by_frequency(m.sys, 1), coupling_ops_by_frequency(m.sys, 2)]

    for (site, bath) in ((1, m.bath1), (2, m.bath2))
        #ops = coupling_ops_by_frequency(m.sys, site)
        for p in pq, q in pq, sgn_w in (+1, -1), sgn_wp in (+1, -1)
            w = sgn_w * r
            wp = sgn_wp * r
            Ap = ops[site][(p, sgn_w)]   # c_p[w]
            Aq = ops[site][(q, sgn_wp)]  # c_q[w']
            pref = (p == q) ? -1.0 : 1.0  # -(1 - 2δ_pq)

            gamma_w, eta_w = gamma_eta_from_phi(bath, p, q, w)
            gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)

            Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)
            coeff_ls = 0.5 * (eta_w + eta_wp) + 0.25im * (gamma_w - gamma_wp)

            # dissipator term: Aq ρ Ap† - 1/2 {Ap† Aq, ρ}
            L_diss += 2pref * Gamma *
            #L_diss += -2pref * Gamma *
                    (spost(Ap', d) * spre(Aq, d) -
                       0.5 * spre(Ap' * Aq, d) - 0.5 * spost(Ap' * Aq, d))

            # Lamb shift term
            #H_LS += pref * coeff_ls * (Ap' * Aq)
            H_LS += -pref * coeff_ls * (Ap' * Aq)
        end
    end

    H_LS = 0.5 * (H_LS + H_LS')
    return L_diss, H_LS
end

function redfield_Heff(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    pq = (:alpha, :beta)
    L_diss = zeros(C64, d^2, d^2)
    H_LS = zeros(C64, d, d)
    H_g = zeros(C64, d, d)

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
            H_LS += pref * coeff_ls  * (Ap' * Aq)
            H_g += pref * 1.0im*Gamma * (Ap' * Aq)
        end
    end

    H_LS = 0.5 * (H_LS + H_LS')
    return H_LS, H_g
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
        #L_diss += -2pref * Gamma *
        L_diss += 2pref * Gamma *
                    (spost(Ap', d) * spre(Aq, d) -
                    0.5 * spre(Ap' * Aq, d) - 0.5 * spost(Ap' * Aq, d))

        # Lamb shift term
        #H_LS += pref * coeff_ls * (Ap' * Aq)
        H_LS += -pref * coeff_ls * (Ap' * Aq)
    end

    H_LS = 0.5 * (H_LS + H_LS')
    return L_diss, H_LS
end

function redfield_parts_TwistedBoundary(m::TwoBathModel, t::Float64)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    pq = (:alpha, :beta)
    L_diss = zeros(C64, d^2, d^2)
    H_LS = zeros(C64, d, d)

    ops = [coupling_ops_by_frequency(m.sys, 1), coupling_ops_by_frequency(m.sys, 2)]

    #1,2
    for p in pq, q in pq, sgn_w in (+1, -1), sgn_wp in (+1, -1)
        w = sgn_w * r
        wp = sgn_wp * r
        Ap = ops[1][(p, sgn_w)]   # c_p[w]
        Aq = ops[2][(q, sgn_wp)]  # c_q[w']
        pref = (p == q) ? -1.0 : 1.0  # -(1 - 2δ_pq)

        gamma_w, eta_w = gamma_eta_from_phi_TwistedBoundary(m.bath1, m.bath2, p, q, w, t)
        gamma_wp, eta_wp = gamma_eta_from_phi_TwistedBoundary(m.bath2, m.bath1, p, q, wp, t)

        
        Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)
        coeff_ls = 0.5 * (eta_w + eta_wp) + 0.25im * (gamma_w - gamma_wp)

        # dissipator term: Aq ρ Ap† - 1/2 {Ap† Aq, ρ}
        #L_diss += pref * Gamma *
        L_diss += 2pref * Gamma *
                (spost(Ap', d) * spre(Aq, d) -
                    0.5 * spre(Ap' * Aq, d) - 0.5 * spost(Ap' * Aq, d))

        # Lamb shift term
        H_LS += -pref * coeff_ls * (Ap' * Aq)
    end
    #2,1
    for p in pq, q in pq, sgn_w in (+1, -1), sgn_wp in (+1, -1)
        w = sgn_w * r
        wp = sgn_wp * r
        Ap = ops[2][(p, sgn_w)]   # c_p[w]
        Aq = ops[1][(q, sgn_wp)]  # c_q[w']
        pref = (p == q) ? -1.0 : 1.0  # -(1 - 2δ_pq)

        gamma_w, eta_w = gamma_eta_from_phi_TwistedBoundary(m.bath2, m.bath1, p, q, w, t)
        gamma_wp, eta_wp = gamma_eta_from_phi_TwistedBoundary(m.bath1, m.bath2, p, q, wp, t)

        Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)
        coeff_ls = 0.5 * (eta_w + eta_wp) + 0.25im * (gamma_w - gamma_wp)

        # dissipator term: Aq ρ Ap† - 1/2 {Ap† Aq, ρ}
        #L_diss += pref * Gamma *
        L_diss += 2pref * Gamma *
                (spost(Ap', d) * spre(Aq, d) -
                    0.5 * spre(Ap' * Aq, d) - 0.5 * spost(Ap' * Aq, d))

        # Lamb shift term
        H_LS += -pref * coeff_ls * (Ap' * Aq)
    end

    H_LS = 0.5 * (H_LS + H_LS')
    return L_diss, H_LS
end


function redfield_parts_TwistedBoundary(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    pq = (:alpha, :beta)
    L_diss = zeros(C64, d^2, d^2)
    H_LS = zeros(C64, d, d)

    ops = [coupling_ops_by_frequency(m.sys, 1), coupling_ops_by_frequency(m.sys, 2)]

    #1,2
    for p in pq, q in pq, sgn_w in (+1, -1), sgn_wp in (+1, -1)
        w = sgn_w * r
        wp = sgn_wp * r
        Ap = ops[1][(p, sgn_w)]   # c_p[w]
        Aq = ops[2][(q, sgn_wp)]  # c_q[w']
        pref = (p == q) ? -1.0 : 1.0  # -(1 - 2δ_pq)

        #gamma_w, eta_w = gamma_eta_from_phi_TwistedBoundary(m.bath1, m.bath2, p, q, w)
        gamma_w, eta_w = gamma_eta_from_phi_TwistedBoundary(m.bath1, m.bath2, q, p, w)
        gamma_wp, eta_wp = gamma_eta_from_phi_TwistedBoundary(m.bath2, m.bath1, p, q, wp)

        
        #Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)
        #coeff_ls = 0.5 * (eta_w + eta_wp) + 0.25im * (gamma_w - gamma_wp)

        Gamma = 0.5 * (gamma_w' + gamma_wp) - 1im * (eta_w' - eta_wp)
        coeff_ls = 0.5 * (eta_w' + eta_wp) + 0.25im * (gamma_w' - gamma_wp)

        # dissipator term: Aq ρ Ap† - 1/2 {Ap† Aq, ρ}
        #L_diss += pref * Gamma *
        L_diss += 2pref * Gamma *
                (spost(Ap', d) * spre(Aq, d) -
                    0.5 * spre(Ap' * Aq, d) - 0.5 * spost(Ap' * Aq, d))

        # Lamb shift term
        H_LS += -pref * coeff_ls * (Ap' * Aq)
    end
    #2,1
    for p in pq, q in pq, sgn_w in (+1, -1), sgn_wp in (+1, -1)
        w = sgn_w * r
        wp = sgn_wp * r
        Ap = ops[2][(p, sgn_w)]   # c_p[w]
        Aq = ops[1][(q, sgn_wp)]  # c_q[w']
        pref = (p == q) ? -1.0 : 1.0  # -(1 - 2δ_pq)

        #gamma_w, eta_w = gamma_eta_from_phi_TwistedBoundary(m.bath2, m.bath1, p, q, w)
        gamma_w, eta_w = gamma_eta_from_phi_TwistedBoundary(m.bath2, m.bath1, q, p, w)
        gamma_wp, eta_wp = gamma_eta_from_phi_TwistedBoundary(m.bath1, m.bath2, p, q, wp)

        #Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)
        #coeff_ls = 0.5 * (eta_w + eta_wp) + 0.25im * (gamma_w - gamma_wp)

        Gamma = 0.5 * (gamma_w' + gamma_wp) - 1im * (eta_w' - eta_wp)
        coeff_ls = 0.5 * (eta_w' + eta_wp) + 0.25im * (gamma_w' - gamma_wp)

        # dissipator term: Aq ρ Ap† - 1/2 {Ap† Aq, ρ}
        #L_diss += pref * Gamma *
        L_diss += 2pref * Gamma *
                (spost(Ap', d) * spre(Aq, d) -
                    0.5 * spre(Ap' * Aq, d) - 0.5 * spost(Ap' * Aq, d))

        # Lamb shift term
        H_LS += -pref * coeff_ls * (Ap' * Aq)
    end

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
            L_diss += 2pref * Gamma *
                    (spost(Ap', d) * spre(Aq, d) -
                       0.5 * spre(Ap' * Aq, d) - 0.5 * spost(Ap' * Aq, d))

            # Lamb shift term
            #H_LS += pref * coeff_ls * (Ap' * Aq)
            H_LS += -pref * coeff_ls * (Ap' * Aq)
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
        L_diss += 2pref * Gamma *
        #L_diss += -2pref * Gamma *
                (spost(Ap', d) * spre(Aq, d) -
                    0.5 * spre(Ap' * Aq, d) - 0.5 * spost(Ap' * Aq, d))

        # Lamb shift term
        #H_LS += pref * coeff_ls * (Ap' * Aq)
        H_LS += -pref * coeff_ls * (Ap' * Aq)
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

"""
Redfield Liouvillian (eq. 95) for the two-bath model.
"""

function redfield_liouvillian_TwistedBoundary(m::TwoBathModel, ts::Vector{Float64} = collect(0.0:0.02:2.0))
    H = system_hamiltonian(m.sys)
    #ts = collect(0.0:0.02:2.0)  # Example time points for the twist
    L_diss = Matrix{C64}[]
    H_LS = Matrix{C64}[]
    for t in ts
        L_diss_t, H_LS_t = redfield_parts_TwistedBoundary(m, t)
        push!(L_diss, L_diss_t)
        push!(H_LS, H_LS_t)
    end
    return H_LS, L_diss
end


function normalgksl_liouvillian(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    L_diss, H_LS = normalgksl_parts(m)
    return commutator_super(H + H_LS) + L_diss
end
