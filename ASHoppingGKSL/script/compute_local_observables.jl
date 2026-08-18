push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..")))
import ASHoppingGKSL as gksl

include(joinpath(@__DIR__, "..", "src2", "ASHoppingGKSL1P.jl"))
using .ASHoppingGKSL1P

function parse_args(args)
    opts = Dict{String, String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        startswith(arg, "--") || error("Unexpected argument: $arg")
        i == length(args) && error("Missing value for $arg")
        opts[arg[3:end]] = args[i + 1]
        i += 2
    end
    return opts
end

get_opt(opts, key, default) = get(opts, key, string(default))

function build_model(nsites::Int, t::Float64, omega::Float64, mu::Float64,
                     beta::Float64, Gamma::Float64, tau_c::Float64,
                     bathkind::Symbol)
    sys = gksl.SystemParams(nsites, t, omega)
    if bathkind == :bath
        b1 = gksl.make_bath(Gamma=Gamma, mu=-mu, beta=beta, tau_c=tau_c)
        b2 = gksl.make_bath(Gamma=Gamma, mu=mu, beta=beta, tau_c=tau_c)
    elseif bathkind == :bath2
        b1 = gksl.make_bath2(Gamma=Gamma, mu=-mu, beta=beta, tau_c=tau_c)
        b2 = gksl.make_bath2(Gamma=Gamma, mu=mu, beta=beta, tau_c=tau_c)
    else
        error("bathkind must be :bath or :bath2")
    end
    return sys, gksl.TwoBathModel(sys, b1, b2)
end

function compute_or_load_correlator(opts)
    input_path = get(opts, "input", "")
    t = parse(Float64, get_opt(opts, "t", 1.0))

    if !isempty(input_path)
        format = Symbol(get_opt(opts, "format", "auto"))
        C = ASHoppingGKSL1P.load_correlator(input_path; format=format)
        @show real.(C[2:3, 2:3])
        return C, t
    end

    nsites = parse(Int, get_opt(opts, "nsites", 4))
    omega = parse(Float64, get_opt(opts, "omega", 0.0))
    mu = parse(Float64, get_opt(opts, "mu", 0.8))
    beta = parse(Float64, get_opt(opts, "beta", 8.0))
    Gamma = parse(Float64, get_opt(opts, "Gamma", 1.0))
    tau_c = parse(Float64, get_opt(opts, "tau-c", 0.2))
    bathkind = Symbol(get_opt(opts, "bathkind", "bath2"))
    method = Symbol(get_opt(opts, "method", "nr"))

    sys, model = build_model(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind)
    C = ASHoppingGKSL1P.one_particle_correlator_for_method(model, method)
    return C, sys.t
end

function main(args)
    opts = parse_args(args)
    C, t = compute_or_load_correlator(opts)
    obs = ASHoppingGKSL1P.local_observables_from_C(C, t)

    println("central_indices = ", obs.central_indices)
    println("beta_eff = ", obs.beta_eff)
    println("temp_eff = ", obs.temp_eff)
    println("mu_eff = ", obs.mu_eff)
    println("n_minus = ", obs.n_minus)
    println("n_plus = ", obs.n_plus)
    println("residual_diag = ", obs.residual_diag)
    println("residual_imag = ", obs.residual_imag)
    println("beta_eff_energy = ", obs.beta_eff_energy)
    println("temp_eff_energy = ", obs.temp_eff_energy)
    println("mu_eff_energy = ", obs.mu_eff_energy)
    println("energy_loc = ", obs.energy_loc)
    println("number_loc = ", obs.number_loc)
    println("n_minus_energy = ", obs.n_minus_energy)
    println("n_plus_energy = ", obs.n_plus_energy)
    println("occupation_clamp_error_energy = ", obs.occupation_clamp_error_energy)
    println("current = ", obs.current)
end

main(ARGS)
