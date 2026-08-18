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