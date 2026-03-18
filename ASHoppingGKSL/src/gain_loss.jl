#include("ops_space.jl")
#include("bath.jl")
#include("redfield.jl")
#include("NathanRudner.jl")

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