function shift_matrix(L::Int, bc::BoundaryCondition)
    T = zeros(ComplexF64, L, L)
    for i in 1:(L - 1)
        T[i + 1, i] = 1.0
    end
    if bc isa PBC
        T[1, L] = 1.0
    end
    return T
end

function dissipation_kernels(p::ModelParams, bc::BoundaryCondition)
    T = shift_matrix(p.L, bc)
    A = (T - T') / (2im)
    I_L = Matrix{ComplexF64}(I, p.L, p.L)
    ta_val = ta(p)
    gamma_gain = ta_val * (I_L + A)
    gamma_loss = ta_val * (I_L - A)
    return gamma_gain, gamma_loss
end
