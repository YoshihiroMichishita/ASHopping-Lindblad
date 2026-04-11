#include("ops_space.jl")
#include("bath.jl")
#include("redfield.jl")


function majorana_channel_index()
    return Dict(:alpha => 1, :beta => 2)
end

function nr_lfac()
    # same convention as current code
    return Dict(:alpha => -im, :beta => im)
end

function gammahat_matrix(b::BathSpectrum, w::Float64)
    phi_aa = Phi_from_sigma(b, :alpha, :alpha, w)
    phi_ab = Phi_from_sigma(b, :alpha, :beta,  w)
    phi_ba = Phi_from_sigma(b, :beta,  :alpha, w)
    phi_bb = Phi_from_sigma(b, :beta,  :beta,  w)

    g11 = phi_aa + conj(phi_aa)
    g12 = phi_ab + conj(phi_ba)
    g21 = phi_ba + conj(phi_ab)
    g22 = phi_bb + conj(phi_bb)

    return C64[
        g11 g12
        g21 g22
    ]
end
"""
γ-matrix square root (eqs. 84-86, 100) at frequency w.
"""

function gammahat_sqrt(b::BathSpectrum, w::Float64)
    g = gammahat_matrix(b, w)
    eigvals, eigvecs = eigen(Hermitian(g))
    sqrt_g = eigvecs * Diagonal(sqrt.(complex.(eigvals))) * eigvecs'
    return sqrt_g
end

#TODO:Dynamic版も作る
function gammahat_sqrt_TwistedBoundary(m::TwoBathModel, w::Float64; scale::Float64=1.0)
    b1 = m.bath1
    b2 = m.bath2
    gamma_11 = Phi_from_sigma(b1, :alpha, :alpha, w) + Phi_from_sigma(b1, :alpha, :alpha, w)'  # eq. (72)
    gamma_p_11 = Phi_from_sigma(b1, :alpha, :beta, w) + Phi_from_sigma(b1, :beta, :alpha, w)'  # eq. (73)
    gamma_22 = Phi_from_sigma(b2, :alpha, :alpha, w) + Phi_from_sigma(b2, :alpha, :alpha, w)'  # eq. (72)
    gamma_p_22 = Phi_from_sigma(b2, :alpha, :beta, w) + Phi_from_sigma(b2, :beta, :alpha, w)'  # eq.(73)
    gamma_12 = scale* (Phi_from_sigma_TwistedBoundary(b1, b2, :alpha, :alpha, w) + Phi_from_sigma_TwistedBoundary(b1, b2, :alpha, :alpha, w)')  # eq. (72)
    gamma_p_12 = scale* (Phi_from_sigma_TwistedBoundary(b1, b2, :alpha, :beta, w) + Phi_from_sigma_TwistedBoundary(b1, b2, :beta, :alpha, w)')  # eq. (73)
    gamma_21 = scale* (Phi_from_sigma_TwistedBoundary(b2, b1, :alpha, :alpha, w) + Phi_from_sigma_TwistedBoundary(b2, b1, :alpha, :alpha, w)')  # eq. (72)
    gamma_p_21 = scale* (Phi_from_sigma_TwistedBoundary(b2, b1, :alpha, :beta, w) + Phi_from_sigma_TwistedBoundary(b2, b1, :beta, :alpha, w)')  # eq. (73)

    g = C64[gamma_11 gamma_p_11 gamma_12 gamma_p_12;
            gamma_p_11' gamma_11 gamma_p_12' gamma_12;
            gamma_21 gamma_p_21 gamma_22 gamma_p_22;
            gamma_p_21' gamma_21 gamma_p_22' gamma_22]
    eigvals, eigvecs = eigen(Hermitian(g))
    sqrt_eigvals = sqrt.(complex.(eigvals))
    sqrt_g = eigvecs * Diagonal(sqrt_eigvals) * eigvecs'
    sqrt_g_dict = Dict((1, 1) => sqrt_g[1:2, 1:2], (1, 2) => sqrt_g[1:2, 3:4],
                     (2, 1) => sqrt_g[3:4, 1:2], (2, 2) => sqrt_g[3:4, 3:4])
    return sqrt_g_dict
end
#=
function gammahat_sqrt_TwistedBoundary(m::TwoBathModel, w::Float64; scale::Float64=1.0)
    b1 = m.bath1
    b2 = m.bath2
    gamma_11 = Phi_from_sigma(b1, :alpha, :alpha, w) + Phi_from_sigma(b1, :alpha, :alpha, w)'  # eq. (72)
    gamma_p_11 = Phi_from_sigma(b1, :alpha, :beta, w) + Phi_from_sigma(b1, :beta, :alpha, w)'  # eq. (73)
    gamma_22 = Phi_from_sigma(b2, :alpha, :alpha, w) + Phi_from_sigma(b2, :alpha, :alpha, w)'  # eq. (72)
    gamma_p_22 = Phi_from_sigma(b2, :alpha, :beta, w) + Phi_from_sigma(b2, :beta, :alpha, w)'  # eq.(73)
    gamma_12 = scale* (Phi_from_sigma_TwistedBoundary(b1, b2, :alpha, :alpha, w) + Phi_from_sigma_TwistedBoundary(b1, b2, :alpha, :alpha, w)')  # eq. (72)
    gamma_p_12 = scale* (Phi_from_sigma_TwistedBoundary(b1, b2, :alpha, :beta, w) + Phi_from_sigma_TwistedBoundary(b1, b2, :beta, :alpha, w)')  # eq. (73)
    gamma_21 = scale* (Phi_from_sigma_TwistedBoundary(b2, b1, :alpha, :alpha, w) + Phi_from_sigma_TwistedBoundary(b2, b1, :alpha, :alpha, w)')  # eq. (72)
    gamma_p_21 = scale* (Phi_from_sigma_TwistedBoundary(b2, b1, :alpha, :beta, w) + Phi_from_sigma_TwistedBoundary(b2, b1, :beta, :alpha, w)')  # eq. (73)

    g1 = C64[gamma_11 gamma_p_11; gamma_p_11' gamma_11]
    g2 = C64[gamma_22 gamma_p_22; gamma_p_22' gamma_22]
    g12 = C64[gamma_12 gamma_p_12; gamma_p_12' gamma_12]
    g21 = C64[gamma_21 gamma_p_21; gamma_p_21' gamma_21]
    eigvals, eigvecs = eigen(Hermitian(g1))
    sqrt_eigvals = sqrt.(complex.(eigvals))
    sqrt_g_11 = eigvecs * Diagonal(sqrt_eigvals) * eigvecs'
    eigvals, eigvecs = eigen(Hermitian(g2))
    sqrt_eigvals = sqrt.(complex.(eigvals))
    sqrt_g_22 = eigvecs * Diagonal(sqrt_eigvals) * eigvecs'
    eigvals, eigvecs = eigen(Hermitian(g12))
    sqrt_eigvals = sqrt.(complex.(eigvals))
    sqrt_g_12 = eigvecs * Diagonal(sqrt_eigvals) * eigvecs'
    eigvals, eigvecs = eigen(Hermitian(g21))
    sqrt_eigvals = sqrt.(complex.(eigvals))
    sqrt_g_21 = eigvecs * Diagonal(sqrt_eigvals) * eigvecs'
    sqrt_g_dict = Dict((1, 1) => sqrt_g_11, (1, 2) => sqrt_g_12,
                     (2, 1) => sqrt_g_21, (2, 2) => sqrt_g_22)
    return sqrt_g_dict
end=#

"""
Nathan-Rudner type GKSL Liouvillian (eqs. 100-109).
"""

function nr_parts_from_redfield(m::TwoBathModel)
    _, H = system_hamiltonian(m.sys)
    d = size(H, 1)

    L_diss = zeros(C64, d^2, d^2)
    H_LS   = zeros(C64, d, d)

    pset = majorana_channels()
    pidx = majorana_channel_index()

    for (site, bath) in bath_site_pairs(m)
        ops = coupling_ops_by_frequency(m.sys, site)
        freqs = frequency_list(ops)

        for w in freqs, wp in freqs
            sqrt_g_w  = gammahat_sqrt(bath, w)
            sqrt_g_wp = gammahat_sqrt(bath, wp)

            # Nathan-Rudner coefficient matrix
            Gamma = sqrt_g_w * sqrt_g_wp'

            for p in pset, q in pset
                Ap = ops[(p, w)]
                Aq = ops[(q, wp)]

                pref = (p == q) ? -1.0 : 1.0

                gamma_w,  eta_w  = gamma_eta_from_phi(bath, p, q, w)
                gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)

                coeff_ls = 0.5 * (eta_w + eta_wp)

                L_diss += -2pref * Gamma[pidx[p], pidx[q]] * (
                    spost(Ap', d) * spre(Aq, d)
                    - 0.5 * spre(Ap' * Aq, d)
                    - 0.5 * spost(Ap' * Aq, d)
                )

                H_LS += pref * coeff_ls * (Ap' * Aq)
            end
        end
    end

    H_LS = 0.5 * (H_LS + H_LS')
    return L_diss, H_LS
end

function nr_Heff_from_redfield(m::TwoBathModel)
    _, H = system_hamiltonian(m.sys)
    d = size(H, 1)

    H_LS = zeros(C64, d, d)
    H_g  = zeros(C64, d, d)

    pset = majorana_channels()
    pidx = majorana_channel_index()

    for (site, bath) in bath_site_pairs(m)
        ops = coupling_ops_by_frequency(m.sys, site)
        freqs = frequency_list(ops)

        for w in freqs, wp in freqs
            sqrt_g_w  = gammahat_sqrt(bath, w)
            sqrt_g_wp = gammahat_sqrt(bath, wp)
            Gamma = sqrt_g_w * sqrt_g_wp'

            for p in pset, q in pset
                Ap = ops[(p, w)]
                Aq = ops[(q, wp)]

                pref = (p == q) ? -1.0 : 1.0

                gamma_w,  eta_w  = gamma_eta_from_phi(bath, p, q, w)
                gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)

                coeff_ls = 0.5 * (eta_w + eta_wp)

                H_g  += 1im * pref * Gamma[pidx[p], pidx[q]] * (Ap' * Aq)
                H_LS +=       pref * coeff_ls              * (Ap' * Aq)
            end
        end
    end

    H_LS = 0.5 * (H_LS + H_LS')
    return H_LS, H_g
end

#=
function nr_parts_from_redfield(m::TwoBathModel)
    _, H = system_hamiltonian(m.sys)
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
    _, H = system_hamiltonian(m.sys)
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
end=#

function nr_liouvillian_from_redfield(m::TwoBathModel)
    _, H = system_hamiltonian(m.sys)
    L_diss, H_LS = nr_parts_from_redfield(m)
    return commutator_super(H + H_LS) + L_diss
end

function nr_gksl_liouvillian(m::TwoBathModel)
    _, H = system_hamiltonian(m.sys)
    d = size(H, 1)

    pset = majorana_channels()
    pidx = majorana_channel_index()
    lfac = nr_lfac()

    Hls_total = zeros(C64, d, d)

    # Lamb shift part
    for (site, bath) in bath_site_pairs(m)
        ops = coupling_ops_by_frequency(m.sys, site)
        freqs = frequency_list(ops)

        Hls = zeros(C64, d, d)

        for p in pset, q in pset, w in freqs, wp in freqs
            Ap = ops[(p, w)]
            Aq = ops[(q, wp)]

            pref = (p == q) ? -1.0 : 1.0

            gamma_w,  eta_w  = gamma_eta_from_phi(bath, p, q, w)
            gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)

            coeff_ls = 0.5 * (eta_w + eta_wp)
            Hls += pref * coeff_ls * (Ap' * Aq)
        end

        Hls = 0.5 * (Hls + Hls')
        Hls_total += Hls
    end

    L = commutator_super(H + Hls_total)

    # Dissipator part
    for (site, bath) in bath_site_pairs(m)
        ops = coupling_ops_by_frequency(m.sys, site)
        freqs = frequency_list(ops)

        jumps = Matrix{C64}[]

        for p in pset
            J = zeros(C64, d, d)

            for w in freqs
                sqrt_g = gammahat_sqrt(bath, w)
                for q in pset
                    J += sqrt_g[pidx[p], pidx[q]] * lfac[q] * ops[(q, w)]
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

#=
function nr_gksl_liouvillian(m::TwoBathModel)
    _, H = system_hamiltonian(m.sys)
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

function nr_gksl_liouvillian_TwistedBoundary(m::TwoBathModel; scale::Float64 = 1.0)
    _, H = system_hamiltonian(m.sys)
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
    _, H_LS_TB = redfield_parts_TwistedBoundary(m; scale=scale)
    Hls_total += H_LS_TB

    L = commutator_super(H + Hls_total)

    # Dissipator (Nathan-Rudner GKSL)
    ops = [coupling_ops_by_frequency(m.sys, 1), coupling_ops_by_frequency(m.sys, 2)]
    #=
    jumps = Matrix{C64}[]
    for sgn in (+1, -1)
        sqg = gammahat_sqrt_TwistedBoundary(m, sgn * r; scale=scale)
        J = zeros(C64, d, d)
        for p in pset, site1 in (1, 2)
            for q in pset, site2 in (1, 2)
                J += sqg[(site1, site2)][pidx[p], pidx[q]] * lfac[q] * ops[site2][(q, sgn)]
            end
            push!(jumps, J)
        end
        for J in jumps
            L += lindblad_super(J)
        end
    end=#
    jumps = Matrix{C64}[]
    for p in pset, site1 in (1, 2)
        J = zeros(C64, d, d)
        for sgn in (+1, -1)
            sqg = gammahat_sqrt_TwistedBoundary(m, sgn * r; scale=scale)
            for q in pset, site2 in (1, 2)
                J += sqg[(site1, site2)][pidx[p], pidx[q]] * lfac[q] * ops[site2][(q, sgn)]
            end
        end
        push!(jumps, J)
    end
    for J in jumps
        L += lindblad_super(J)
    end
    return L
end=#

function nr_gksl_liouvillian_TwistedBoundary(m::TwoBathModel; scale::Float64 = 1.0)
    _, H = system_hamiltonian(m.sys)
    d = size(H, 1)

    pset = majorana_channels()
    pidx = majorana_channel_index()
    lfac = nr_lfac()

    ops_by_bath = tb_ops_by_bath(m)
    freqs = tb_frequency_list(m)

    Hls_total = zeros(C64, d, d)

    # local Lamb shift
    for bathind in 1:2
        bath = bath_by_index(m, bathind)
        ops = ops_by_bath[bathind]

        Hls = zeros(C64, d, d)
        for p in pset, q in pset, w in freqs, wp in freqs
            Ap = ops[(p, w)]
            Aq = ops[(q, wp)]
            pref = (p == q) ? -1.0 : 1.0

            gamma_w, eta_w   = gamma_eta_from_phi(bath, p, q, w)
            gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)

            Hls += pref * 0.5 * (eta_w + eta_wp) * (Ap' * Aq)
        end

        Hls_total += 0.5 * (Hls + Hls')
    end

    # cross Lamb shift
    _, Hls_tb = redfield_parts_TwistedBoundary(m; scale=scale)
    Hls_total += Hls_tb

    L = commutator_super(H + Hls_total)

    # TBC dissipator
    jumps = Matrix{C64}[]
    for p in pset, out_bath in 1:2
        J = zeros(C64, d, d)
        for w in freqs
            sqg = gammahat_sqrt_TwistedBoundary(m, w; scale=scale)
            for q in pset, in_bath in 1:2
                J += sqg[(out_bath, in_bath)][pidx[p], pidx[q]] *
                     lfac[q] *
                     ops_by_bath[in_bath][(q, w)]
            end
        end
        push!(jumps, J)
    end

    for J in jumps
        L += lindblad_super(J)
    end

    return L
end


function nr_gksl_liouvillian_diss(m::TwoBathModel)
    _, H = system_hamiltonian(m.sys)
    d = size(H, 1)

    pset = majorana_channels()
    pidx = majorana_channel_index()
    lfac = nr_lfac()

    Ls = [zeros(C64, d^2, d^2), zeros(C64, d^2, d^2)]

    for (bathind, (site, bath)) in enumerate(bath_site_pairs(m))
        ops = coupling_ops_by_frequency(m.sys, site)
        freqs = frequency_list(ops)

        jumps = Matrix{C64}[]

        for p in pset
            J = zeros(C64, d, d)

            for w in freqs
                sqrt_g = gammahat_sqrt(bath, w)
                for q in pset
                    J += sqrt_g[pidx[p], pidx[q]] * lfac[q] * ops[(q, w)]
                end
            end

            push!(jumps, J)
        end

        for J in jumps
            Ls[bathind] += lindblad_super(J)
        end
    end

    return Ls
end

#=
function nr_gksl_liouvillian_diss(m::TwoBathModel)
    _, H = system_hamiltonian(m.sys)
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
end=#