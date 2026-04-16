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

function compute_one_particle_heff(m, sys, mode; scale::Float64 = 1.0, time::Float64 = 0.0, Nt::Int = 400)
    _, Hsys = gksl.system_hamiltonian(sys)

    if mode == :red
        Hls, Hg = gksl.redfield_Heff(m)
        return gksl.one_particle_block(Hsys + Hls + Hg, sys)
    elseif mode == :nr
        Hls, Hg = gksl.nr_Heff_from_redfield(m)
        return gksl.one_particle_block(Hsys + Hls + Hg, sys)
    elseif mode == :tb_static
        Hls_loc, Hg_loc = gksl.redfield_Heff(m)
        Hls_tb, Hg_tb = gksl.redfield_Heff_TwistedBoundary(m; scale=scale)
        return gksl.one_particle_block(Hsys + Hls_loc + Hg_loc + Hls_tb + Hg_tb, sys)
    elseif mode == :tb_time
        Hls_loc, Hg_loc = gksl.redfield_Heff(m)
        Hls_tb, Hg_tb = gksl.redfield_Heff_TwistedBoundary(m, time; scale=scale)
        return gksl.one_particle_block(Hsys + Hls_loc + Hg_loc + Hls_tb + Hg_tb, sys)
    elseif mode == :tb_floquet
        hF, _, _, _ = gksl.redfield_Heff_TwistedBoundary_floquet(m; Nt=Nt, scale=scale)
        return hF
    else
        error("mode must be :red, :nr, :tb_static, :tb_time, or :tb_floquet")
    end
end

function dissipative_diag_and_heff(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind;
                                   mode::Symbol = :red, scale::Float64 = 1.0,
                                   time::Float64 = 0.0, Nt::Int = 400)
    sys, m = build_model(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind)
    h = compute_one_particle_heff(m, sys, mode; scale=scale, time=time, Nt=Nt)
    h_diss = 0.5 * (h - h')
    γ = imag.(diag(h_diss))
    return γ, h
end

function target_diagonal_segment(γ)
    nsites = length(γ)
    if nsites > 3
        return γ[2:(end - 1)], collect(2:(nsites - 1))
    end
    return γ, collect(1:nsites)
end

function arithmetic_metrics(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind;
                            mode::Symbol = :red, scale::Float64 = 1.0,
                            time::Float64 = 0.0, Nt::Int = 400,
                            slope_floor::Float64 = 1e-8)
    γ_full, _ = dissipative_diag_and_heff(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind;
                                          mode=mode, scale=scale, time=time, Nt=Nt)
    γ_target, target_indices = target_diagonal_segment(γ_full)
    length(γ_target) >= 2 || error("Need at least 2 target sites to test arithmetic progression")
    d1 = diff(γ_target)
    d2 = diff(d1)
    eq_score = norm(d2)
    slope_score = norm(d1)
    if length(d2) == 0
        score = -slope_score
    else
        score = eq_score / (slope_score + slope_floor)
    end
    return score, eq_score, slope_score, γ_full, γ_target, target_indices, d1, d2
end

function choose_index_two_stage(eq_scores, slope_scores, tol_eq)
    candidates = findall(<(tol_eq), eq_scores)
    if !isempty(candidates)
        local_best = candidates[argmax(slope_scores[candidates])]
        return local_best, :constraint_satisfied
    end
    return argmin(eq_scores), :fallback_min_eq
end

function find_minimum_ratio(f; omega_min::Float64 = -2.0, omega_max::Float64 = 2.0,
                            ngrid::Int = 401, nrefine::Int = 6)
    lo, hi = omega_min, omega_max
    best_omega = (lo + hi) / 2
    best_score = f(best_omega)

    for _ in 1:nrefine
        ws = collect(range(lo, hi; length=ngrid))
        vals = f.(ws)
        idx = argmin(vals)
        best_omega = ws[idx]
        best_score = vals[idx]

        if idx == 1
            lo, hi = ws[1], ws[2]
        elseif idx == length(ws)
            lo, hi = ws[end - 1], ws[end]
        else
            lo, hi = ws[idx - 1], ws[idx + 1]
        end
    end

    return best_omega, best_score
end

function find_minimum_two_stage(metrics_f; omega_min::Float64 = -2.0, omega_max::Float64 = 2.0,
                                ngrid::Int = 401, nrefine::Int = 6, tol_eq::Float64 = 1e-6)
    lo, hi = omega_min, omega_max
    best_omega = (lo + hi) / 2
    best_eq = Inf
    best_slope = 0.0
    stage_status = :fallback_min_eq

    for _ in 1:nrefine
        ws = collect(range(lo, hi; length=ngrid))
        eq_scores = zeros(Float64, length(ws))
        slope_scores = zeros(Float64, length(ws))

        for (i, w) in enumerate(ws)
            _, eq_score, slope_score, _, _, _, _, _ = metrics_f(w)
            eq_scores[i] = eq_score
            slope_scores[i] = slope_score
        end

        idx, stage_status = choose_index_two_stage(eq_scores, slope_scores, tol_eq)
        best_omega = ws[idx]
        best_eq = eq_scores[idx]
        best_slope = slope_scores[idx]

        if idx == 1
            lo, hi = ws[1], ws[2]
        elseif idx == length(ws)
            lo, hi = ws[end - 1], ws[end]
        else
            lo, hi = ws[idx - 1], ws[idx + 1]
        end
    end

    return best_omega, best_eq, best_slope, stage_status
end

function main(args)
    opts = parse_args(args)

    nsites = parse(Int, get_opt(opts, "nsites", 3))
    t = parse(Float64, get_opt(opts, "t", 1.0))
    mu = parse(Float64, get_opt(opts, "mu", 0.8))
    beta = parse(Float64, get_opt(opts, "beta", 8.0))
    Gamma = parse(Float64, get_opt(opts, "Gamma", 1.0))
    tau_c = parse(Float64, get_opt(opts, "tau-c", 0.2))
    bathkind = Symbol(get_opt(opts, "bathkind", "bath2"))
    mode = Symbol(get_opt(opts, "mode", "red"))
    scale = parse(Float64, get_opt(opts, "scale", 1.0))
    time = parse(Float64, get_opt(opts, "time", 0.0))
    Nt = parse(Int, get_opt(opts, "Nt", 400))
    search_method = Symbol(get_opt(opts, "search-method", "ratio"))
    tol_eq = parse(Float64, get_opt(opts, "tol-eq", 1e-6))
    slope_floor = parse(Float64, get_opt(opts, "slope-floor", 1e-8))
    omega_min = parse(Float64, get_opt(opts, "omega-min", -2.0))
    omega_max = parse(Float64, get_opt(opts, "omega-max", 2.0))
    ngrid = parse(Int, get_opt(opts, "ngrid", 401))
    nrefine = parse(Int, get_opt(opts, "nrefine", 6))

    nsites >= 2 || error("Need nsites >= 2 to check arithmetic progression")

    metrics_f(omega) = arithmetic_metrics(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind;
                                          mode=mode, scale=scale, time=time, Nt=Nt,
                                          slope_floor=slope_floor)

    stage_status = :ratio
    if search_method == :ratio
        f(omega) = first(metrics_f(omega))
        omega_star, _ = find_minimum_ratio(f; omega_min=omega_min, omega_max=omega_max,
                                           ngrid=ngrid, nrefine=nrefine)
    elseif search_method == :two_stage
        omega_star, _, _, stage_status = find_minimum_two_stage(metrics_f;
                                                                omega_min=omega_min,
                                                                omega_max=omega_max,
                                                                ngrid=ngrid,
                                                                nrefine=nrefine,
                                                                tol_eq=tol_eq)
    else
        error("search-method must be :ratio or :two_stage")
    end

    score_star, eq_score_star, slope_score_star, gamma_full_star, gamma_target_star,
    target_indices_star, diff1_star, diff2_star =
        arithmetic_metrics(nsites, t, omega_star, mu, beta, Gamma, tau_c, bathkind;
                           mode=mode, scale=scale, time=time, Nt=Nt, slope_floor=slope_floor)
    _, h_star = dissipative_diag_and_heff(nsites, t, omega_star, mu, beta, Gamma, tau_c, bathkind;
                                          mode=mode, scale=scale, time=time, Nt=Nt)

    println("Parameters:")
    println("  nsites   = ", nsites)
    println("  t        = ", t)
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
    if mode != :red && mode != :nr
        println("  scale    = ", scale)
    end
    println("  search_method = ", search_method)
    if search_method == :two_stage
        println("  tol_eq   = ", tol_eq)
        println("  stage_status = ", stage_status)
    else
        println("  slope_floor = ", slope_floor)
    end
    println()
    println("Search:")
    println("  omega range = [", omega_min, ", ", omega_max, "]")
    println("  ngrid       = ", ngrid)
    println("  nrefine     = ", nrefine)
    println()
    println("Solution:")
    println("  omega*      = ", omega_star)
    println("  score       = ", score_star)
    println("  eq_score    = ", eq_score_star)
    println("  slope_score = ", slope_score_star)
    println("  target_sites = ", target_indices_star)
    println("  gamma_full   = ", gamma_full_star)
    println("  gamma_target = ", gamma_target_star)
    println("  diff1_target = ", diff1_star)
    println("  diff2_target = ", diff2_star)
    println("  h_eff =")
    display(h_star)
end

main(ARGS)
