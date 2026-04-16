push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..")))
import ASHoppingGKSL as gksl
using LinearAlgebra

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

function get_opt(opts, key, default)
    return get(opts, key, string(default))
end

function build_model(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind)
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

function main(args)
    opts = parse_args(args)

    nsites = parse(Int, get_opt(opts, "nsites", 3))
    t = parse(Float64, get_opt(opts, "t", 1.0))
    omega = parse(Float64, get_opt(opts, "omega", 0.0))
    mu = parse(Float64, get_opt(opts, "mu", 0.8))
    beta = parse(Float64, get_opt(opts, "beta", 8.0))
    Gamma = parse(Float64, get_opt(opts, "Gamma", 1.0))
    tau_c = parse(Float64, get_opt(opts, "tau-c", 0.2))
    bathkind = Symbol(get_opt(opts, "bathkind", "bath2"))
    mode = Symbol(get_opt(opts, "mode", "local"))
    scale = parse(Float64, get_opt(opts, "scale", 1.0))
    time = parse(Float64, get_opt(opts, "time", 0.0))
    Nt = parse(Int, get_opt(opts, "Nt", 400))

    sys, m = build_model(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind)
    H0, Hsys = gksl.system_hamiltonian(sys)
    hsys = gksl.one_particle_block(Hsys, sys)
    Hls_loc, Hg_loc = gksl.redfield_Heff(m)
    hloc = gksl.one_particle_block(Hsys + Hls_loc + Hg_loc, sys)

    println("Parameters:")
    println("  nsites   = ", nsites)
    println("  t        = ", t)
    println("  omega    = ", omega)
    println("  mu       = ", mu)
    println("  beta     = ", beta)
    println("  Gamma    = ", Gamma)
    println("  tau_c    = ", tau_c)
    println("  bathkind = ", bathkind)
    println("  mode     = ", mode)
    if mode == :tb_time
        println("  time     = ", time)
    elseif mode == :tb_floquet
        println("  Nt       = ", Nt)
    end
    if mode != :local
        println("  scale    = ", scale)
    end
    println()

    println("H0 =")
    display(H0)
    println()
    println("one_particle_block(Hsys) =")
    display(hsys)
    println()
    println("local one-particle Heff =")
    display(hloc)
    println()

    if mode == :red
        println("selected Heff =")
        display(hloc)
        println()
        return
    elseif mode == :nr
        Hls_loc, Hg_loc = gksl.nr_Heff_from_redfield(m)
        hloc_nr = gksl.one_particle_block(Hsys + Hls_loc + Hg_loc, sys)
        println("NR one-particle contribution =")
        display(hloc_nr - hsys)
        println()
        println("selected Heff =")
        display(hloc_nr)
        println()
    elseif mode == :tb_static
        Hls_tb, Hg_tb = gksl.redfield_Heff_TwistedBoundary(m; scale=scale)
        htb = gksl.one_particle_block(Hls_tb + Hg_tb, sys)
        htotal = gksl.one_particle_block(Hsys + Hls_loc + Hg_loc + Hls_tb + Hg_tb, sys)
        println("TB one-particle contribution =")
        display(htb)
        println()
        println("selected Heff =")
        display(htotal)
        println()
    elseif mode == :tb_time
        Hls_tb, Hg_tb = gksl.redfield_Heff_TwistedBoundary(m, time; scale=scale)
        htb = gksl.one_particle_block(Hls_tb + Hg_tb, sys)
        htotal = gksl.one_particle_block(Hsys + Hls_loc + Hg_loc + Hls_tb + Hg_tb, sys)
        println("TB(t) one-particle contribution =")
        display(htb)
        println()
        println("selected Heff =")
        display(htotal)
        println()
    elseif mode == :tb_floquet
        hF, U, ts, extras = gksl.redfield_Heff_TwistedBoundary_floquet(m; Nt=Nt, scale=scale)
        println("selected Heff (Floquet) =")
        display(hF)
        println()
        println("Floquet propagator U =")
        display(U)
        println()
        println("period = ", ts[end] - ts[1])
        println("extras = ", extras)
    else
        error("mode must be :local, :tb_static, :tb_time, or :tb_floquet")
    end
end

main(ARGS)
