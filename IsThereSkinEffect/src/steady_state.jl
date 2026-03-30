function occupation_rhs(
    p::AbstractVector{<:Real},
    W::AbstractMatrix{<:Real};
    injection::AbstractVector{<:Real} = zeros(eltype(p), length(p)),
    loss::AbstractVector{<:Real} = zeros(eltype(p), length(p)),
)
    n_modes = length(p)
    rhs = zeros(Float64, n_modes)

    for m in 1:n_modes
        rhs[m] += injection[m] * (1 - p[m]) - loss[m] * p[m]
    end

    for from in 1:n_modes, to in 1:n_modes
        if from == to
            continue
        end
        rhs[to] += W[from, to] * p[from] * (1 - p[to])
        rhs[to] -= W[to, from] * p[to] * (1 - p[from])
    end

    return rhs
end

function stable_timestep(
    W::AbstractMatrix{<:Real},
    injection::AbstractVector{<:Real},
    loss::AbstractVector{<:Real},
)
    col_sum = vec(sum(W; dims = 1))
    row_sum = vec(sum(W; dims = 2))
    scale = maximum(vcat(injection .+ loss, col_sum, row_sum, [1.0]))
    return 0.1 / scale
end

function solve_steady_mode_occupations(
    W::AbstractMatrix{<:Real};
    injection::Union{Nothing, AbstractVector{<:Real}} = nothing,
    loss::Union{Nothing, AbstractVector{<:Real}} = nothing,
    initial::Union{Nothing, AbstractVector{<:Real}} = nothing,
    dt::Union{Nothing, Real} = nothing,
    tol::Real = 1e-10,
    max_steps::Int = 2_000_000,
)
    n_modes = size(W, 1)
    size(W, 2) == n_modes || error("W must be square")

    inj = injection === nothing ? zeros(Float64, n_modes) : collect(Float64, injection)
    los = loss === nothing ? zeros(Float64, n_modes) : collect(Float64, loss)
    occ = initial === nothing ? fill(0.5, n_modes) : clamp.(collect(Float64, initial), 0.0, 1.0)
    step = dt === nothing ? stable_timestep(W, inj, los) : float(dt)

    for _ in 1:max_steps
        rhs = occupation_rhs(occ, W; injection = inj, loss = los)
        if norm(rhs, Inf) < tol
            return occ
        end
        occ .= clamp.(occ .+ step .* rhs, 0.0, 1.0)
    end

    error("steady-state solver did not converge within max_steps = $(max_steps)")
end

function steady_state(
    p::ModelParams,
    bc::BoundaryCondition;
    initial::Union{Nothing, AbstractVector{<:Real}} = nothing,
    dt::Union{Nothing, Real} = nothing,
    tol::Real = 1e-10,
    max_steps::Int = 2_000_000,
)
    eps, U, labels = mode_basis(p, bc)
    W = thermal_rates(eps, p.alpha, p.beta)
    injections = zeros(Float64, length(eps))
    losses = zeros(Float64, length(eps))

    if bc isa PBC
        gain, loss = drive_rates_pbc(p, labels)
        injections .+= gain
        losses .+= loss
    elseif bc isa OBC
        W .+= drive_rates_obc(p, U)
    else
        error("unsupported boundary condition")
    end

    init = initial === nothing ? thermal_reference_occupations(eps, p) : initial
    occupations = solve_steady_mode_occupations(
        W;
        injection = injections,
        loss = losses,
        initial = init,
        dt = dt,
        tol = tol,
        max_steps = max_steps,
    )

    density = site_density(U, occupations)
    boundary = bc isa PBC ? :pbc : :obc
    drive_model = bc isa PBC ? :mode : :site
    return SteadyStateResult(boundary, eps, occupations, density, labels, U, drive_model)
end

function steady_state_site_drive(
    p::ModelParams,
    bc::BoundaryCondition;
    initial::Union{Nothing, AbstractVector{<:Real}} = nothing,
    dt::Union{Nothing, Real} = nothing,
    tol::Real = 1e-10,
    max_steps::Int = 2_000_000,
)
    eps, U, labels = mode_basis(p, bc)
    periodic = bc isa PBC
    W = thermal_rates(eps, p.alpha, p.beta) + drive_rates_site(p, U; periodic = periodic)
    init = initial === nothing ? thermal_reference_occupations(eps, p) : initial
    occupations = solve_steady_mode_occupations(
        W;
        initial = init,
        dt = dt,
        tol = tol,
        max_steps = max_steps,
    )
    density = site_density(U, occupations)
    boundary = bc isa PBC ? :pbc : :obc
    return SteadyStateResult(boundary, eps, occupations, density, labels, U, :site)
end

function compare_boundaries(
    p::ModelParams;
    initial_obc::Union{Nothing, AbstractVector{<:Real}} = nothing,
    initial_pbc::Union{Nothing, AbstractVector{<:Real}} = nothing,
    dt::Union{Nothing, Real} = nothing,
    tol::Real = 1e-10,
    max_steps::Int = 2_000_000,
)
    pbc = steady_state(
        p,
        PBC();
        initial = initial_pbc,
        dt = dt,
        tol = tol,
        max_steps = max_steps,
    )
    obc = steady_state(
        p,
        OBC();
        initial = initial_obc,
        dt = dt,
        tol = tol,
        max_steps = max_steps,
    )
    return (; pbc, obc)
end

function compare_boundaries_site_drive(
    p::ModelParams;
    initial_obc::Union{Nothing, AbstractVector{<:Real}} = nothing,
    initial_pbc::Union{Nothing, AbstractVector{<:Real}} = nothing,
    dt::Union{Nothing, Real} = nothing,
    tol::Real = 1e-10,
    max_steps::Int = 2_000_000,
)
    pbc = steady_state_site_drive(
        p,
        PBC();
        initial = initial_pbc,
        dt = dt,
        tol = tol,
        max_steps = max_steps,
    )
    obc = steady_state_site_drive(
        p,
        OBC();
        initial = initial_obc,
        dt = dt,
        tol = tol,
        max_steps = max_steps,
    )
    return (; pbc, obc)
end
