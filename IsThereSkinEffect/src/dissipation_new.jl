function active_bonds(L::Int, ::PBC)
    return [(j, j == L ? 1 : j + 1) for j in 1:L]
end

function active_bonds(L::Int, ::OBC)
    return [(j, j + 1) for j in 1:(L - 1)]
end

function loss_jump_vector(L::Int, j::Int, jp1::Int)
    v = zeros(ComplexF64, L)
    v[j] = 1.0
    v[jp1] = -im
    return v
end

function gain_jump_vector(L::Int, j::Int, jp1::Int)
    v = zeros(ComplexF64, L)
    v[j] = 1.0
    v[jp1] = im
    return v
end

function dissipation_kernels_eq15(p::ModelParams, bc::BoundaryCondition)
    gamma_gain = zeros(ComplexF64, p.L, p.L)
    gamma_loss = zeros(ComplexF64, p.L, p.L)

    for (j, jp1) in active_bonds(p.L, bc)
        v_loss = loss_jump_vector(p.L, j, jp1)
        v_gain = gain_jump_vector(p.L, j, jp1)
        gamma_loss .+= p.J .* (v_loss * v_loss')
        gamma_gain .+= p.J .* (v_gain * v_gain')
    end

    return gamma_gain, gamma_loss
end

function dissipation_kernels_new(p::ModelParams, bc::BoundaryCondition)
    return dissipation_kernels_eq15(p, bc)
end
