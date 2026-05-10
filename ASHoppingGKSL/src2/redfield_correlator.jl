function redfield_terms_coeffs(m::gksl.TwoBathModel)
    terms = Tuple{ComplexF64, Vector{ComplexF64}, Vector{ComplexF64}, Vector{ComplexF64}, Vector{ComplexF64}}[]

    for bathind in 1:2
        site = gksl.bath_site(m, bathind)
        bath = gksl.bath_by_index(m, bathind)
        ops = gksl.coupling_ops_by_frequency(m.sys, site)
        freqs = gksl.frequency_list(ops)

        for p in gksl.majorana_channels(), q in gksl.majorana_channels(), w in freqs, wp in freqs
            Ap = ops[(p, w)]
            Aq = ops[(q, wp)]

            pref = (p == q) ? -1.0 : 1.0
            Gamma, _ = gksl.redfield_coefficients_from_bath(bath, p, q, w, wp)
            coeff = ComplexF64(-2 * pref * Gamma)
            u_p, v_p = gksl.operator_linear_coeffs_in_site_basis(Ap, m.sys)
            u_q, v_q = gksl.operator_linear_coeffs_in_site_basis(Aq, m.sys)
            push!(terms, (coeff, u_q, v_q, u_p, v_p))
        end
    end

    return terms
end

struct ReducedRedfieldTerm
    coeff::ComplexF64
    Aq::Matrix{ComplexF64}
    Ap::Matrix{ComplexF64}
end

struct ReducedRedfieldCache
    hred::Matrix{ComplexF64}
    terms::Vector{ReducedRedfieldTerm}
end

const REDUCED_REDFIELD_CACHE = Dict{Any, ReducedRedfieldCache}()

function get_reduced_redfield_cache(m::gksl.TwoBathModel)
    return get!(REDUCED_REDFIELD_CACHE, m) do
        n = m.sys.n_sites
        cache = get_reduced_space_cache(n)
        hsys, _ = gksl.system_hamiltonian(m.sys)
        Hls_full = gksl.redfield_parts(m)[2]
        hls = gksl.one_particle_block(Hls_full, m.sys)
        hred = reduced_number_conserving_operator(ComplexF64.(hsys) + hls, cache)
        terms = ReducedRedfieldTerm[
            ReducedRedfieldTerm(
                coeff,
                reduced_linear_operator(uA, vA, cache.c_ops, cache.cd_ops),
                reduced_linear_operator(uB, vB, cache.c_ops, cache.cd_ops),
            )
            for (coeff, uA, vA, uB, vB) in redfield_terms_coeffs(m)
        ]
        ReducedRedfieldCache(hred, terms)
    end
end

function one_particle_generator_redfield(m::gksl.TwoBathModel, basis::BilinearBasis)
    n = m.sys.n_sites
    reduced_cache = get_reduced_space_cache(n)
    model_cache = get_reduced_redfield_cache(m)

    n2 = n^2
    M = zeros(ComplexF64, n2, n2)
    b = zeros(ComplexF64, n2)

    for a in 1:n2
        Ored = reduced_cache.bilinear_ops[a]
        LOred = 1im .* (model_cache.hred * Ored - Ored * model_cache.hred)
        for term in model_cache.terms
            K = term.Ap' * term.Aq
            LOred .+= term.coeff .* (term.Ap' * Ored * term.Aq .- 0.5 .* (K * Ored .+ Ored * K))
        end
        s, X = source_and_bilinear_from_reduced(LOred)
        b[a] = s
        M[a, :] .= flatten_bilinear_block(X)
    end

    return M, b
end

function steady_correlator_redfield(m::gksl.TwoBathModel)
    basis = bilinear_basis(m.sys)
    M, b = one_particle_generator_redfield(m, basis)
    x = M \ (-b)
    return bilinear_matrix_from_coeff_vector(x, basis)
end
