#include("ops_space.jl")
#include("bath.jl")
#include("redfield.jl")

"""
γ-matrix square root (eqs. 84-86, 100) at frequency w.
"""
function gammahat_sqrt(b::BathSpectrum, w::Float64)
    gamma = Phi_from_sigma(b, :alpha, :alpha, w) + Phi_from_sigma(b, :alpha, :alpha, w)'  # eq. (72)
    gamma_p = Phi_from_sigma(b, :alpha, :beta, w) + Phi_from_sigma(b, :beta, :alpha, w)'  # eq. (73)
    g = C64[gamma gamma_p; gamma_p' gamma]
    eigvals, eigvecs = eigen(Hermitian(g))
    sqrt_eigvals = sqrt.(complex.(eigvals))
    sqrt_g = eigvecs * Diagonal(sqrt_eigvals) * eigvecs'
    return sqrt_g
end

"""
Nathan-Rudner type GKSL Liouvillian (eqs. 100-109).
"""
function nr_parts_from_redfield(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    pq = (:alpha, :beta)
    L_diss = zeros(C64, d^2, d^2)
    H_LS = zeros(C64, d, d)

    pidx = Dict(:alpha => 1, :beta => 2)
    ops = [coupling_ops_by_frequency(m.sys, 1), coupling_ops_by_frequency(m.sys, 2)]
    for (site, bath) in ((1, m.bath1), (2, m.bath2))
        op = coupling_ops_by_frequency(m.sys, site)
        for sgn_w in (+1, -1), sgn_wp in (+1, -1)
            w = sgn_w * r
            wp = sgn_wp * r
            gamma_sqrt = gammahat_sqrt(bath, w)
            gamma_sqrt_wp = gammahat_sqrt(bath, wp)
            Gamma = gamma_sqrt * gamma_sqrt_wp'  # eq. (101)
            for p in pq, q in pq
                
                Ap = op[(p, sgn_w)]   # c_p[w]
                Aq = op[(q, sgn_wp)]  # c_q[w']
                pref = (p == q) ? -1.0 : 1.0  # -(1 - 2δ_pq)

                gamma_w, eta_w = gamma_eta_from_phi(bath, p, q, w)
                gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)

                
                #Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)
                coeff_ls = 0.5 * (eta_w + eta_wp)

                # dissipator term: Aq ρ Ap† - 1/2 {Ap† Aq, ρ}
                #L_diss += pref * Gamma *
                L_diss += -2pref * Gamma[pidx[p], pidx[q]] *
                        (spost(Ap', d) * spre(Aq, d) -
                        0.5 * spre(Ap' * Aq, d) - 0.5 * spost(Ap' * Aq, d))

                # Lamb shift term
                H_LS += pref * coeff_ls * (Ap' * Aq)
            end
        end
    end

    H_LS = 0.5 * (H_LS + H_LS')
    return L_diss, H_LS
end

function nr_Heff_from_redfield(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    pq = (:alpha, :beta)
    L_diss = zeros(C64, d, d)
    H_LS = zeros(C64, d, d)

    pidx = Dict(:alpha => 1, :beta => 2)

    for (site, bath) in ((1, m.bath1), (2, m.bath2))
        ops = coupling_ops_by_frequency(m.sys, site)
        for sgn_w in (+1, -1), sgn_wp in (+1, -1)
            w = sgn_w * r
            wp = sgn_wp * r
            gamma_sqrt = gammahat_sqrt(bath, w)
            gamma_sqrt_wp = gammahat_sqrt(bath, wp)
            Gamma = gamma_sqrt * gamma_sqrt_wp'  # eq. (101)
            for p in pq, q in pq
                
                Ap = ops[(p, sgn_w)]   # c_p[w]
                Aq = ops[(q, sgn_wp)]  # c_q[w']
                pref = (p == q) ? -1.0 : 1.0  # -(1 - 2δ_pq)

                gamma_w, eta_w = gamma_eta_from_phi(bath, p, q, w)
                gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)

                
                #Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)
                coeff_ls = 0.5 * (eta_w + eta_wp)

                # dissipator term: Aq ρ Ap† - 1/2 {Ap† Aq, ρ}
                #L_diss += pref * Gamma *
                L_diss += -1.0im * pref * Gamma[pidx[p], pidx[q]] * (Ap' * Aq)

                # Lamb shift term
                H_LS += pref * coeff_ls * (Ap' * Aq)
            end
        end
    end

    H_LS = 0.5 * (H_LS + H_LS')
    return H_LS, L_diss
end

function nr_liouvillian_from_redfield(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    L_diss, H_LS = nr_parts_from_redfield(m)
    return commutator_super(H + H_LS) + L_diss
end

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

            #coeff = 0.5 * (eta_w + eta_wp) + 0.25im * (gamma_w - gamma_wp)
            coeff = 0.5 * (eta_w + eta_wp)
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