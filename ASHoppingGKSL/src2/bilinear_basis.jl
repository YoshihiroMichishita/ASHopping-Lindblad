struct BilinearBasis
    sys::gksl.SystemParams
    ids::Vector{Union{Symbol, Tuple{Int, Int}}}
    ops::Vector{Matrix{ComplexF64}}
    gram::Matrix{ComplexF64}
    gram_factor
end

function bilinear_basis(sys::gksl.SystemParams)
    c_ops, cd_ops = gksl.site_ops(sys)
    d = size(c_ops[1], 1)

    ids = Union{Symbol, Tuple{Int, Int}}[:I]
    ops = Matrix{ComplexF64}[Matrix{ComplexF64}(I, d, d)]

    for i in 1:sys.n_sites, j in 1:sys.n_sites
        push!(ids, (i, j))
        push!(ops, cd_ops[i] * c_ops[j])
    end

    gram = zeros(ComplexF64, length(ops), length(ops))
    for a in eachindex(ops), b in eachindex(ops)
        gram[a, b] = tr(ops[a]' * ops[b])
    end

    return BilinearBasis(sys, ids, ops, gram, lu(gram))
end

function project_to_basis(Op::AbstractMatrix{<:Complex}, basis::BilinearBasis)
    p = zeros(ComplexF64, length(basis.ops))
    for a in eachindex(basis.ops)
        p[a] = tr(basis.ops[a]' * Op)
    end
    return basis.gram_factor \ p
end

function reconstruct_from_basis(coeffs::AbstractVector{<:Complex}, basis::BilinearBasis)
    Op = zero(basis.ops[1])
    for a in eachindex(coeffs)
        Op += coeffs[a] * basis.ops[a]
    end
    return Op
end

function bilinear_matrix_from_coeff_vector(x::AbstractVector{<:Complex}, basis::BilinearBasis)
    n = basis.sys.n_sites
    C = zeros(ComplexF64, n, n)
    idx = 1
    for i in 1:n, j in 1:n
        C[i, j] = x[idx]
        idx += 1
    end
    return C
end

function flatten_bilinear_block(X::AbstractMatrix{<:Complex})
    n = size(X, 1)
    x = zeros(ComplexF64, n * n)
    idx = 1
    for i in 1:n, j in 1:n
        x[idx] = X[i, j]
        idx += 1
    end
    return x
end

function reduced_sector_states(n::Int)
    states = Tuple{Vararg{Int}}[()]
    append!(states, Tuple{Vararg{Int}}[(i,) for i in 1:n])
    append!(states, Tuple{Vararg{Int}}[(i, j) for i in 1:n for j in (i + 1):n])

    index = Dict{Tuple{Vararg{Int}}, Int}()
    for (idx, state) in enumerate(states)
        index[state] = idx
    end

    return states, index
end

function reduced_annihilation_ops(n::Int)
    return get_reduced_space_cache(n).c_ops
end

function reduced_creation_ops(n::Int)
    return get_reduced_space_cache(n).cd_ops
end

function reduced_bilinear_basis(n::Int)
    cache = get_reduced_space_cache(n)
    ids = Union{Symbol, Tuple{Int, Int}}[:I]
    ops = Matrix{ComplexF64}[cache.identity_op]
    for i in 1:n, j in 1:n
        push!(ids, (i, j))
        push!(ops, cache.bilinear_ops[(i - 1) * n + j])
    end
    return ids, ops
end

function reduced_number_conserving_operator(h::AbstractMatrix{<:Complex}, cache::ReducedSpaceCache)
    O = zeros(ComplexF64, size(cache.identity_op)...)
    for i in 1:cache.n, j in 1:cache.n
        idx = (i - 1) * cache.n + j
        O .+= ComplexF64(h[i, j]) .* cache.bilinear_ops[idx]
    end
    return O
end

function reduced_number_conserving_operator(h::AbstractMatrix{<:Complex},
                                            c_ops_red::Vector{<:AbstractMatrix{<:Complex}},
                                            cd_ops_red::Vector{<:AbstractMatrix{<:Complex}})
    O = zeros(ComplexF64, size(c_ops_red[1])...)
    for i in axes(h, 1), j in axes(h, 2)
        O .+= ComplexF64(h[i, j]) .* cd_ops_red[i] * c_ops_red[j]
    end
    return O
end

function reduced_linear_operator(u::AbstractVector{<:Complex},
                                 v::AbstractVector{<:Complex},
                                 c_ops_red::Vector{<:AbstractMatrix{<:Complex}},
                                 cd_ops_red::Vector{<:AbstractMatrix{<:Complex}})
    J = zero(c_ops_red[1])
    for i in eachindex(u)
        J .+= u[i] .* c_ops_red[i] .+ v[i] .* cd_ops_red[i]
    end
    return J
end

function source_and_bilinear_from_reduced(Ored::AbstractMatrix{<:Complex})
    n = Int(round((-1 + sqrt(1 + 8 * (size(Ored, 1) - 1))) / 2))
    s = ComplexF64(Ored[1, 1])
    one_particle = Matrix{ComplexF64}(Ored[2:1 + n, 2:1 + n])
    X = one_particle - s .* Matrix{ComplexF64}(I, n, n)
    return s, X
end

function projected_linear_product(u_left::AbstractVector{<:Complex},
                                  v_left::AbstractVector{<:Complex},
                                  u_right::AbstractVector{<:Complex},
                                  v_right::AbstractVector{<:Complex})
    s = ComplexF64(sum(u_left .* v_right))
    X = v_left * transpose(u_right) - v_right * transpose(u_left)
    return s, Matrix{ComplexF64}(X)
end

function projected_pair_adjoint_on_bilinear(X::AbstractMatrix{<:Complex},
                                            uA::AbstractVector{<:Complex},
                                            vA::AbstractVector{<:Complex},
                                            uB::AbstractVector{<:Complex},
                                            vB::AbstractVector{<:Complex},
                                            alpha::ComplexF64)
    u_left = conj.(vB)
    v_left = conj.(uB)

    u_ox_a = -(transpose(X) * uA)
    v_ox_a = X * vA
    s1, X1 = projected_linear_product(u_left, v_left, u_ox_a, v_ox_a)

    sK, XK = projected_linear_product(u_left, v_left, uA, vA)
    Xanti = X * XK + XK * X + 2 * sK * X

    return alpha * s1, alpha .* (X1 - 0.5 .* Xanti)
end
