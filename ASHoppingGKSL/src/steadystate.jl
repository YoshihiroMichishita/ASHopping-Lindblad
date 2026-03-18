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


function steady_state_from_time_dependent_liouvillian(staticL::Matrix{C64}, H_LSs::Vector{Matrix{C64}}, L_disss::Vector{Matrix{C64}}; d::Int, dt::Float64 = 0.02, verbose::Bool = true)
    Nt = length(H_LSs)
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
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    Lr = redfield_liouvillian(m)
    Lg = nr_liouvillian_from_redfield(m)
    #Lg = nr_gksl_liouvillian(m)
    return steady_state_from_liouvillian(Lr; d), steady_state_from_liouvillian(Lg; d), H
end

function steady_states3(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    Lr = redfield_liouvillian(m)
    Lng = normalgksl_liouvillian(m)
    #Lg = nr_gksl_liouvillian(m)
    Lg = nr_liouvillian_from_redfield(m)
    return steady_state_from_liouvillian(Lr; d), steady_state_from_liouvillian(Lng; d), steady_state_from_liouvillian(Lg; d), H
end

function steady_state_td(m::TwoBathModel, ts::Vector{Float64})
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    Lr = redfield_liouvillian(m)
    Lg = nr_liouvillian_from_redfield(m)
    #Lg = nr_gksl_liouvillian(m)
    rho0 = zeros(C64, d, d)
    rho0[1, 1] = 1.0
    rhors = time_evolve(Lr, rho0, ts)
    rhongs = time_evolve(Lg, rho0, ts)
    return rhors, rhongs
end