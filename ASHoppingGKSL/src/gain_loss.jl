#include("ops_space.jl")
#include("bath.jl")
#include("redfield.jl")
#include("NathanRudner.jl")


function local_number_projectors(p::SystemParams)
    c_ops, cd_ops = site_ops(p)

    n_ops = Matrix{C64}[]
    nbar_ops = Matrix{C64}[]

    for i in 1:p.n_sites
        n = cd_ops[i] * c_ops[i]
        nbar = c_ops[i] * cd_ops[i]
        push!(n_ops, n)
        push!(nbar_ops, nbar)
    end

    return n_ops, nbar_ops
end

function nr_jumps_by_bath(m::TwoBathModel)
    pset = majorana_channels()
    pidx = majorana_channel_index()
    lfac = nr_lfac()

    jumps_by_site = Dict{Int, Vector{Matrix{C64}}}()

    for (bathind, site) in enumerate((1, m.sys.n_sites))
        ops = coupling_ops_by_frequency(m.sys, site)
        freqs = frequency_list(ops)
        bath = bath_by_index(m, bathind)
        jumps = Matrix{C64}[]

        for p in pset
            J = zero(ops[(pset[1], freqs[1])])

            for w in freqs
                sqg = gammahat_sqrt(bath, w)
                for q in pset
                    J += sqg[pidx[p], pidx[q]] * lfac[q] * ops[(q, w)]
                end
            end

            push!(jumps, J)
        end
        jumps_by_site[bathind] = jumps
    end

    return jumps_by_site
end

function redfield_gain_loss(m::TwoBathModel, bathind::Int, rho::Matrix{C64})
    pset = majorana_channels()

    site = bath_site(m, bathind)
    bath = bath_by_index(m, bathind)

    ops = coupling_ops_by_frequency(m.sys, site)
    freqs = frequency_list(ops)

    n_ops, nbar_ops = local_number_projectors(m.sys)
    nsites_sys = m.sys.n_sites

    gain_site = zeros(Float64, nsites_sys)
    loss_site = zeros(Float64, nsites_sys)

    for p in pset, q in pset, w in freqs, wp in freqs
        Ap = ops[(p, w)]
        Aq = ops[(q, wp)]

        pref = (p == q) ? 1.0 : -1.0

        gamma_w,  eta_w  = gamma_eta_from_phi(bath, p, q, w)
        gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)

        Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)

        for i in 1:nsites_sys
            n_i = n_ops[i]
            nbar_i = nbar_ops[i]

            Aq_gain = n_i    * Aq * nbar_i
            Ap_gain = n_i    * Ap * nbar_i
            Aq_loss = nbar_i * Aq * n_i
            Ap_loss = nbar_i * Ap * n_i

            gain_site[i] += 2real(pref * Gamma * tr(Aq_gain * rho * Ap_gain'))
            loss_site[i] += 2real(pref * Gamma * tr(Aq_loss * rho * Ap_loss'))
        end
    end

    return gain_site, loss_site
end

function site_gain_loss_bath(m::TwoBathModel, rho::Matrix{C64}; method::Symbol = :nr)
    nsites_sys = m.sys.n_sites
    results = Dict{Tuple{Int, Int}, Tuple{Float64, Float64, Float64}}()

    if method == :nr
        n_ops, nbar_ops = local_number_projectors(m.sys)
        jumps_by_bath = nr_jumps_by_bath(m)

        for bathind in 1:2
            for site in 1:nsites_sys
                gain = 0.0
                loss = 0.0

                n_i = n_ops[site]
                nbar_i = nbar_ops[site]

                for J in jumps_by_bath[bathind]
                    gain_J = n_i    * J * nbar_i
                    loss_J = nbar_i * J * n_i

                    gain += 2real(tr(gain_J * rho * gain_J'))
                    loss += 2real(tr(loss_J * rho * loss_J'))
                end

                results[(bathind, site)] = (gain, loss, gain - loss)
            end
        end

        return results

    elseif method == :redfield
        for bathind in 1:2
            gain_site, loss_site = redfield_gain_loss(m, bathind, rho)

            for site in 1:m.sys.n_sites
                results[(bathind, site)] = (
                    gain_site[site],
                    loss_site[site],
                    gain_site[site] - loss_site[site],
                )
            end
        end

        return results

    else
        error("method must be :nr or :redfield")
    end
end

function site_net_bath(m::TwoBathModel, rho::Matrix{C64}; method::Symbol = :redfield)
    results = Dict{Tuple{Int, Int}, Float64}()

    if method == :redfield
        for bathind in 1:2
            L_diss, _ = redfield_parts(m, bathind)
            net_site = dissipative_site_net_from_liouvillian(L_diss, rho, m.sys)

            for site in 1:m.sys.n_sites
                results[(bathind, site)] = net_site[site]
            end
        end

        return results

    elseif method == :nr
        Ls = nr_gksl_liouvillian_diss(m)

        for bathind in 1:2
            net_site = dissipative_site_net_from_liouvillian(Ls[bathind], rho, m.sys)

            for site in 1:m.sys.n_sites
                results[(bathind, site)] = net_site[site]
            end
        end

        return results

    else
        error("method must be :redfield or :nr")
    end
end

function total_site_gain_loss(results::Dict, nsites_sys::Int)
    gain = zeros(Float64, nsites_sys)
    loss = zeros(Float64, nsites_sys)
    net  = zeros(Float64, nsites_sys)

    for site in 1:nsites_sys
        for bathind in 1:2
            gain[site] += results[(bathind, site)][1]
            loss[site] += results[(bathind, site)][2]
            net[site]  += results[(bathind, site)][3]
        end
    end

    return gain, loss, net
end

function dissipative_site_net_from_liouvillian(L_diss::Matrix{C64}, rho::Matrix{C64}, p)
    n_ops, _ = local_number_projectors(p)
    d = size(rho, 1)

    net = zeros(Float64, length(n_ops))
    drho = reshape(L_diss * vec(rho), d, d)

    for i in 1:length(n_ops)
        net[i] = real(tr(n_ops[i] * drho))
    end

    return net
end

function total_site_net(results::Dict, nsites_sys::Int)
    net = zeros(Float64, nsites_sys)

    for site in 1:nsites_sys
        for bathind in 1:2
            net[site] += results[(bathind, site)]
        end
    end

    return net
end


function operator_linear_coeffs_in_site_basis(Op::AbstractMatrix{<:Complex}, sys::SystemParams)
    c_ops, cd_ops = site_ops(sys)
    n = sys.n_sites

    u = zeros(ComplexF64, n)  # coeffs of c_i
    v = zeros(ComplexF64, n)  # coeffs of c_i†

    # ここは厳密には Hilbert-Schmidt projection
    for i in 1:n
        u[i] = tr(c_ops[i]' * Op) / tr(c_ops[i]' * c_ops[i])
        v[i] = tr(cd_ops[i]' * Op) / tr(cd_ops[i]' * cd_ops[i])
    end

    return u, v
end

function nr_local_gain_loss_coeffs(m::TwoBathModel)
    jumps_by_bath = nr_jumps_by_bath(m)
    n = m.sys.n_sites

    gain_coeff = zeros(Float64, n)  # coeff of c_j† ρ c_j
    loss_coeff = zeros(Float64, n)  # coeff of c_j ρ c_j†

    for bathind in 1:2
        for J in jumps_by_bath[bathind]
            u, v = operator_linear_coeffs_in_site_basis(J, m.sys)

            for j in 1:n
                loss_coeff[j] += 2 * real(u[j] * conj(u[j]))
                gain_coeff[j] += 2 * real(v[j] * conj(v[j]))
            end
        end
    end

    return gain_coeff, loss_coeff
end

function nr_local_gain_loss_coeffs_bath(m::TwoBathModel)
    jumps_by_bath = nr_jumps_by_bath(m)
    n = m.sys.n_sites

    gains = []
    losses = []

    for bathind in 1:2
        gain_coeff = zeros(Float64, n)  # coeff of c_j† ρ c_j
        loss_coeff = zeros(Float64, n)  # coeff of c_j ρ c_j†
        for J in jumps_by_bath[bathind]
            u, v = operator_linear_coeffs_in_site_basis(J, m.sys)

            for j in 1:n
                loss_coeff[j] += 2 * real(u[j] * conj(u[j]))
                gain_coeff[j] += 2 * real(v[j] * conj(v[j]))
            end
        end
        push!(gains, gain_coeff)
        push!(losses, loss_coeff)
    end

    return gains, losses
end

function redfield_local_gain_loss_coeffs(m::TwoBathModel)
    n = m.sys.n_sites

    gain_j = zeros(ComplexF64, n)  # coeff of c_j† ρ c_j
    loss_j = zeros(ComplexF64, n)  # coeff of c_j ρ c_j†

    for bathind in 1:2
        site = bath_site(m, bathind)
        bath = bath_by_index(m, bathind)

        ops = coupling_ops_by_frequency(m.sys, site)
        freqs = frequency_list(ops)

        for p in majorana_channels(), q in majorana_channels(), w in freqs, wp in freqs
            Ap = ops[(p, w)]
            Aq = ops[(q, wp)]

            pref = (p == q) ? 1.0 : -1.0

            gamma_w, eta_w = gamma_eta_from_phi(bath, p, q, w)
            gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)
            Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)

            u_p, v_p = operator_linear_coeffs_in_site_basis(Ap, m.sys)
            u_q, v_q = operator_linear_coeffs_in_site_basis(Aq, m.sys)

            for j in 1:n
                # 2 pref Gamma Aq ρ Ap† の c_j ρ c_j† 成分
                loss_j[j] += 2 * pref * Gamma * u_q[j] * conj(u_p[j])

                # 2 pref Gamma Aq ρ Ap† の c_j† ρ c_j 成分
                gain_j[j] += 2 * pref * Gamma * v_q[j] * conj(v_p[j])
            end
        end
    end

    return gain_j, loss_j
end