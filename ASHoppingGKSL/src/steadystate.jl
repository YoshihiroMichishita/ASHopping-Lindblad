function thermal_grand_canonical_state(H::Matrix{C64}, beta::Float64)
    rho = exp(-beta * (H))
    rho ./= tr(rho)
    return rho
end


function steady_state_from_liouvillian(L::Matrix{C64}; d::Int)
    n = d^2
    size(L) == (n, n) || error("L must be a $(n)×$(n) matrix for d = $d")
    I_d = Matrix{C64}(I, d, d)
    tr_row = reshape(vec(transpose(I_d)), 1, :)
    A = [L; tr_row]
    b = zeros(C64, size(L, 1) + 1)
    b[end] = 1 + 0im
    rho_vec = A \ b
    #println("eigenval check: ", norm(L * rho_vec))
    rho = reshape(rho_vec, d, d)
    rho = 0.5 * (rho + rho')
    rho ./= real(tr(rho))
    return rho
end


function steady_state_from_time_dependent_liouvillian(staticL::Matrix{C64}, H_LSs::Vector{Matrix{C64}}, L_disss::Vector{Matrix{C64}}; dt::Float64 = 0.02, verbose::Bool = true)
    Nt = length(H_LSs)
    d = size(H_LSs[1], 1)
    rho = Matrix{C64}(I, d, d) / d  # Start with the maximally mixed state
    for t in 1:Nt
        L_t = staticL + commutator_super(H_LSs[t]) + L_disss[t]
        drho_dt = L_t * vec(rho)
        diff = norm(drho_dt)
        if verbose
            println("Time step $t, norm of drho/dt: $diff")
        end
        rho += reshape(drho_dt, d, d) * dt  # Simple Euler step with a small time step (dt)
        rho ./= real(tr(rho))
    end
    return rho
end


function steady_states(m::TwoBathModel)
    _, H = system_hamiltonian(m.sys)
    d = size(H, 1)
    Lr = redfield_liouvillian(m)
    #Lg = nr_liouvillian_from_redfield(m)
    Lg = nr_gksl_liouvillian(m)
    return steady_state_from_liouvillian(Lr; d), steady_state_from_liouvillian(Lg; d), H
end

function steady_states3(m::TwoBathModel)
    _, H = system_hamiltonian(m.sys)
    d = size(H, 1)
    Lr = redfield_liouvillian(m)
    Lng = normalgksl_liouvillian(m)
    Lg = nr_gksl_liouvillian(m)
    #Lg = nr_liouvillian_from_redfield(m)
    return steady_state_from_liouvillian(Lr; d), steady_state_from_liouvillian(Lng; d), steady_state_from_liouvillian(Lg; d), H
end

function steady_state_td(m::TwoBathModel, ts::Vector{Float64};  verbose::Bool = true)
    _, H = system_hamiltonian(m.sys)
    d = size(H, 1)
    static_Lr = redfield_liouvillian(m)
    dynamic_H_LSs, dynamic_L_disss = redfield_liouvillian_TwistedBoundary(m, ts)
    rho0 = zeros(C64, d, d)
    rho0[1, 1] = 1.0
    rho_ss = steady_state_from_time_dependent_liouvillian(static_Lr, dynamic_H_LSs, dynamic_L_disss; dt = ts[2] - ts[1], verbose = verbose)
    return rho_ss
end

function steady_state_tb(m::TwoBathModel; scale::Float64 = 1.0)
    _, H = system_hamiltonian(m.sys)
    d = size(H, 1)
    static_Lr = redfield_liouvillian(m)
    dynamic_L_diss, dynamic_H_LS = redfield_parts_TwistedBoundary(m; scale = scale)
    #rho_ss = steady_state_from_liouvillian(static_Lr + commutator_super(dynamic_H_LS) + dynamic_L_diss; d)
    rho_ss = steady_state_from_liouvillian(static_Lr + commutator_super(dynamic_H_LS) + dynamic_L_diss; d=d)
    return rho_ss
end

function steady_state_tb_nr(m::TwoBathModel; scale::Float64 = 1.0)
    L = nr_gksl_liouvillian_TwistedBoundary(m; scale = scale)
    d = size(system_hamiltonian(m.sys)[2], 1)
    rho_ss = steady_state_from_liouvillian(L; d = d)
    return rho_ss
end


function steady_state_from_liouvillian_ncondition(L::Matrix{C64}, Nexp::Float64=1.0; d::Int)
    n = d^2
    size(L) == (n, n) || error("L must be a $(n)×$(n) matrix for d = $d")
    I_d = Matrix{C64}(I, d, d)
    tr_row = reshape(vec(transpose(I_d)), 1, :)
    #c1, c2, cd1, cd2 = fermion_ops_2site()
    #Nop = cd1 * c1 + cd2 * c2
    Nop = zeros(C64, d, d)
    Nop[2, 2] = 1.0
    Nop[3, 3] = 1.0
    Nop[4, 4] = 2.0
    n_row = reshape(vec(transpose(Nop)), 1, :)
    A = [L; tr_row; n_row]
    b = zeros(C64, size(L, 1) + 2)
    b[end-1] = 1 + 0im
    b[end] = Nexp + 0im
    rho_vec = A \ b
    println("eigenval check: ", norm(L * rho_vec))
    rho = reshape(rho_vec, d, d)
    rho = 0.5 * (rho + rho')
    rho ./= real(tr(rho))
    return rho
end

function steady_state_tb_ncondition(m::TwoBathModel, Nexp::Float64 = 1.0)
    _, H = system_hamiltonian(m.sys)
    d = size(H, 1)
    static_Lr = redfield_liouvillian(m)
    dynamic_L_diss, dynamic_H_LS = redfield_parts_TwistedBoundary(m)
    rho_ss = steady_state_from_liouvillian_ncondition(static_Lr + commutator_super(dynamic_H_LS) + dynamic_L_diss, Nexp; d = d)
    return rho_ss
end