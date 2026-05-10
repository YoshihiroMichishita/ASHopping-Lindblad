function nr_jump_coeffs(m::gksl.TwoBathModel)
    jumps_by_bath = gksl.nr_jumps_by_bath(m)
    jumps = Tuple{Vector{ComplexF64}, Vector{ComplexF64}}[]
    for bath in 1:2
        for J in jumps_by_bath[bath]
            u, v = gksl.operator_linear_coeffs_in_site_basis(J, m.sys)
            push!(jumps, (u, v))
        end
    end
    return jumps
end

struct ReducedNRCache
    hred::Matrix{ComplexF64}
    jumps::Vector{Matrix{ComplexF64}}
end

const REDUCED_NR_CACHE = Dict{Any, ReducedNRCache}()

function get_reduced_nr_cache(m::gksl.TwoBathModel)
    return get!(REDUCED_NR_CACHE, m) do
        n = m.sys.n_sites
        cache = get_reduced_space_cache(n)
        hsys, _ = gksl.system_hamiltonian(m.sys)
        _, Hls_full = gksl.nr_parts_from_redfield(m)
        hls = gksl.one_particle_block(Hls_full, m.sys)
        hred = reduced_number_conserving_operator(ComplexF64.(hsys) + hls, cache)
        jumps = [
            reduced_linear_operator(u, v, cache.c_ops, cache.cd_ops)
            for (u, v) in nr_jump_coeffs(m)
        ]
        ReducedNRCache(hred, jumps)
    end
end

function one_particle_generator_nr(m::gksl.TwoBathModel, basis::BilinearBasis)
    n = m.sys.n_sites
    reduced_cache = get_reduced_space_cache(n)
    model_cache = get_reduced_nr_cache(m)

    n2 = n^2
    M = zeros(ComplexF64, n2, n2)
    b = zeros(ComplexF64, n2)

    for a in 1:n2
        Ored = reduced_cache.bilinear_ops[a]
        LOred = 1im .* (model_cache.hred * Ored - Ored * model_cache.hred)
        for J in model_cache.jumps
            JdJ = J' * J
            LOred .+= 2 .* (J' * Ored * J) .- JdJ * Ored .- Ored * JdJ
        end
        s, X = source_and_bilinear_from_reduced(LOred)
        b[a] = s
        M[a, :] .= flatten_bilinear_block(X)
    end

    return M, b
end

function steady_correlator_nr(m::gksl.TwoBathModel)
    basis = bilinear_basis(m.sys)
    M, b = one_particle_generator_nr(m, basis)
    x = M \ (-b)
    return bilinear_matrix_from_coeff_vector(x, basis)
end
