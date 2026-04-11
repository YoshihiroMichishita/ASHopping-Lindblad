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

#=
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
            n_op = gain_op * loss_op
            nbar_op = loss_op * gain_op
            Aq_gain = n_op * Aq * nbar_op
            Ap_gain = n_op * Ap * nbar_op
            Aq_loss = nbar_op * Aq * n_op
            Ap_loss = nbar_op * Ap * n_op
            #=
            Aq_gain = Aq .* abs.(gain_op)
            Ap_gain = Ap .* abs.(gain_op)
            Aq_loss = Aq .* abs.(loss_op)
            Ap_loss = Ap .* abs.(loss_op)=#
            Gain_site[site] += 2real(pref * Gamma * tr(Aq_gain * rho * Ap_gain'))
            Loss_site[site] += 2real(pref * Gamma * tr(Aq_loss * rho * Ap_loss'))
        end
    end
    return Gain_site, Loss_site
end=#

function redfield_TB_net(m::TwoBathModel, rho::Matrix{C64}; scale::Float64 = 1.0)
    L_diss_TB, _ = redfield_parts_TwistedBoundary(m; scale=scale)
    return dissipative_site_net_from_liouvillian(L_diss_TB, rho, m.sys)
end
#=
function redfield_TB_net(m::TwoBathModel, rho::Matrix{C64}; scale::Float64=1.0)
    _, H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    pq = (:alpha, :beta)
    net_site = [0.0, 0.0]
    #Gain_site = [0.0, 0.0]

    c1, c2, cd1, cd2 = fermion_ops_2site()
    loss_ops = Dict(1 => c1, 2 => c2)
    gain_ops = Dict(1 => cd1, 2 => cd2)

    #mybath = (bathind == 1) ? m.bath1 : m.bath2
    ops = [coupling_ops_by_frequency(m.sys, 1), coupling_ops_by_frequency(m.sys, 2)]

    L_diss_TB, H_LS_TB = redfield_parts_TwistedBoundary(m; scale=scale)
    for site in (1, 2)
        loss_op = loss_ops[site]
        gain_op = gain_ops[site]
        n_op = gain_op * loss_op
        net_site[site] += real(tr(n_op * reshape(L_diss_TB * vec(rho), d, d))) # TB dissipator contribution to net gain-loss
    end
    return net_site
end=#

#=
function site_gain_loss_bath(m::TwoBathModel, rho::Matrix{C64}; method::Symbol = :gksl, scale::Float64 = 1.0)
    c1, c2, cd1, cd2 = fermion_ops_2site()
    loss_ops = Dict(1 => c1, 2 => c2)
    gain_ops = Dict(1 => cd1, 2 => cd2)
    results = Dict{Tuple{Int, Int}, Tuple{Float64, Float64, Float64}}()

    # Nathan--Rudner GKSL: sum over jumps for each site and bath
    if method == :nr
        jumps_by_site = nr_jumps_by_site(m)
        for bath in (1, 2), site in (1, 2)
            gain = 0.0
            loss = 0.0
            loss_op = loss_ops[site]
            gain_op = gain_ops[site]
            n_op = gain_op * loss_op
            nbar_op = loss_op * gain_op
            for J in jumps_by_site[bath]
                loss_J = nbar_op * J * n_op
                gain_J = n_op * J * nbar_op
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
    elseif method == :redfield_tbc
        for bath in (1, 2)
            gain_site, loss_site = redfield_gain_loss(m, bath, rho)
            
            for site in (1, 2)
                results[(bath, site)] = (gain_site[site], loss_site[site], (gain_site[site]) - (loss_site[site]))
            end
        end
        net_tb_site = redfield_TB_net(m, rho; scale=scale)
        for site in (1, 2)
            net_site = results[(1, site)][3] + net_tb_site[site]
            results[(1, site)] = (results[(1, site)][1], results[(1, site)][2], net_site)
        end
        return results
    else
        error("method must be :gksl or :redfield")
    end
end=#

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