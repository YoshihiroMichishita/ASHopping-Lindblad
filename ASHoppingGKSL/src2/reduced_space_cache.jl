struct ReducedSpaceCache
    n::Int
    states::Vector{Tuple{Vararg{Int}}}
    state_index::Dict{Tuple{Vararg{Int}}, Int}
    c_ops::Vector{Matrix{ComplexF64}}
    cd_ops::Vector{Adjoint{ComplexF64, Matrix{ComplexF64}}}
    bilinear_ops::Vector{Matrix{ComplexF64}}
    identity_op::Matrix{ComplexF64}
end

const REDUCED_SPACE_CACHE = Dict{Int, ReducedSpaceCache}()

function build_reduced_annihilation_ops(states::Vector{Tuple{Vararg{Int}}},
                                        state_index::Dict{Tuple{Vararg{Int}}, Int},
                                        n::Int)
    dim = length(states)
    ops = Matrix{ComplexF64}[]
    for i in 1:n
        A = zeros(ComplexF64, dim, dim)
        for (ket_index, state) in enumerate(states)
            if i in state
                phase = isodd(count(site -> site < i, state)) ? -1.0 : 1.0
                bra_state = Tuple(site for site in state if site != i)
                bra_index = state_index[bra_state]
                A[bra_index, ket_index] = phase
            end
        end
        push!(ops, A)
    end
    return ops
end

function build_reduced_space_cache(n::Int)
    states, state_index = reduced_sector_states(n)
    c_ops = build_reduced_annihilation_ops(states, state_index, n)
    cd_ops = [A' for A in c_ops]
    bilinear_ops = Matrix{ComplexF64}[]
    for i in 1:n, j in 1:n
        push!(bilinear_ops, cd_ops[i] * c_ops[j])
    end
    identity_op = Matrix{ComplexF64}(I, length(states), length(states))
    return ReducedSpaceCache(n, states, state_index, c_ops, cd_ops, bilinear_ops, identity_op)
end

function get_reduced_space_cache(n::Int)
    return get!(REDUCED_SPACE_CACHE, n) do
        build_reduced_space_cache(n)
    end
end
