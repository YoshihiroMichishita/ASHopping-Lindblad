function empty_jump_block_dict(n::Int)
    return Dict(
        :loss_coeff => zeros(ComplexF64, n, n),
        :gain_coeff => zeros(ComplexF64, n, n),
    )
end

function accumulate_jump_coefficients!(blocks::Dict{Symbol, Matrix{ComplexF64}},
                                       coeff::ComplexF64,
                                       u_left::AbstractVector{<:Complex},
                                       v_left::AbstractVector{<:Complex},
                                       u_right::AbstractVector{<:Complex},
                                       v_right::AbstractVector{<:Complex})
    blocks[:loss_coeff] .+= 2 .* coeff .* (u_left * u_right')
    blocks[:gain_coeff] .+= 2 .* coeff .* (v_left * v_right')
    return blocks
end

function nr_jump_blocks_by_bath_1p(m::gksl.TwoBathModel)
    n = m.sys.n_sites
    jumps_by_bath = gksl.nr_jumps_by_bath(m)
    out = Dict{Int, Dict{Symbol, Matrix{ComplexF64}}}()

    for bath in 1:2
        blocks = empty_jump_block_dict(n)
        for J in jumps_by_bath[bath]
            u, v = gksl.operator_linear_coeffs_in_site_basis(J, m.sys)
            accumulate_jump_coefficients!(blocks, 1.0 + 0.0im, u, v, u, v)
        end
        out[bath] = blocks
    end

    return out
end

function redfield_jump_blocks_by_bath_1p(m::gksl.TwoBathModel)
    n = m.sys.n_sites
    out = Dict{Int, Dict{Symbol, Matrix{ComplexF64}}}()

    for bath in 1:2
        blocks = empty_jump_block_dict(n)
        site = gksl.bath_site(m, bath)
        spec = gksl.bath_by_index(m, bath)
        ops = gksl.coupling_ops_by_frequency(m.sys, site)
        freqs = gksl.frequency_list(ops)

        for p in gksl.majorana_channels(), q in gksl.majorana_channels(), w in freqs, wp in freqs
            Ap = ops[(p, w)]
            Aq = ops[(q, wp)]

            pref = (p == q) ? 1.0 : -1.0
            gamma_w, eta_w = gksl.gamma_eta_from_phi(spec, p, q, w)
            gamma_wp, eta_wp = gksl.gamma_eta_from_phi(spec, p, q, wp)
            Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)

            u_p, v_p = gksl.operator_linear_coeffs_in_site_basis(Ap, m.sys)
            u_q, v_q = gksl.operator_linear_coeffs_in_site_basis(Aq, m.sys)
            accumulate_jump_coefficients!(blocks, ComplexF64(pref * Gamma), u_q, v_q, u_p, v_p)
        end

        out[bath] = blocks
    end

    return out
end

function attach_weighted_from_C(blocks::Dict{Symbol, Matrix{ComplexF64}},
                                C::AbstractMatrix{<:Complex})
    weighted = loss_gain_weighted_from_C(blocks[:loss_coeff], blocks[:gain_coeff], C)
    out = copy(blocks)
    out[:loss_weighted] = weighted.loss_weighted
    out[:gain_weighted] = weighted.gain_weighted
    return out
end

function jump_blocks_by_bath_1p(m::gksl.TwoBathModel, method::Symbol)
    if method == :nr
        return nr_jump_blocks_by_bath_1p(m)
    elseif method == :redfield
        return redfield_jump_blocks_by_bath_1p(m)
    else
        error("method must be :nr or :redfield")
    end
end

function one_particle_correlator_for_method(m::gksl.TwoBathModel, method::Symbol)
    if method == :nr
        return steady_correlator_nr(m)
    elseif method == :redfield
        return steady_correlator_redfield(m)
    else
        error("method must be :nr or :redfield")
    end
end

function jump_blocks_with_weighted_1p(m::gksl.TwoBathModel, method::Symbol)
    blocks_by_bath = jump_blocks_by_bath_1p(m, method)
    C = one_particle_correlator_for_method(m, method)
    return Dict(bath => attach_weighted_from_C(blocks_by_bath[bath], C) for bath in keys(blocks_by_bath))
end
