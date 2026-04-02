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

function heff_block(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind)
    sys, m = build_model(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind)
    _, Hsys = gksl.system_hamiltonian(sys)
    Hls, Hg = gksl.redfield_Heff(m)
    return sys, gksl.one_particle_block(Hsys + Hls + Hg, sys)
end

function re_diagdiff(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind, site_i, site_j)
    _, h = heff_block(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind)
    return real(h[site_i, site_i] - h[site_j, site_j])
end

function bisect_root(f, a, b; tol=1e-12, maxiter=200)
    fa = f(a)
    fb = f(b)
    fa == 0 && return a
    fb == 0 && return b
    signbit(fa) == signbit(fb) && error("Root is not bracketed: f($a)=$fa, f($b)=$fb")

    left, right = a, b
    fleft = fa
    for _ in 1:maxiter
        mid = (left + right) / 2
        fmid = f(mid)
        if abs(fmid) < tol || abs(right - left) < tol
            return mid
        end
        if signbit(fleft) == signbit(fmid)
            left = mid
            fleft = fmid
        else
            right = mid
        end
    end
    return (left + right) / 2
end

function find_root(nsites, t, mu, beta, Gamma, tau_c, bathkind, site_i, site_j; omega_min=-2.0, omega_max=2.0, ngrid=4001)
    f(omega) = re_diagdiff(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind, site_i, site_j)
    xs = collect(range(omega_min, omega_max; length=ngrid))
    vals = f.(xs)

    for i in 1:(length(xs) - 1)
        if vals[i] == 0
            return xs[i], vals[i]
        elseif signbit(vals[i]) != signbit(vals[i + 1])
            omega = bisect_root(f, xs[i], xs[i + 1])
            return omega, f(omega)
        end
    end

    idx = argmin(abs.(vals))
    error("No sign change found in [$omega_min, $omega_max]. Nearest point: omega=$(xs[idx]), residual=$(vals[idx])")
end

function main(args)
    opts = parse_args(args)

    nsites = parse(Int, get_opt(opts, "nsites", 2))
    t = parse(Float64, get_opt(opts, "t", 1.0))
    mu = parse(Float64, get_opt(opts, "mu", 0.8))
    beta = parse(Float64, get_opt(opts, "beta", 8.0))
    Gamma = parse(Float64, get_opt(opts, "Gamma", 1.0))
    tau_c = parse(Float64, get_opt(opts, "tau-c", 0.2))
    bathkind = Symbol(get_opt(opts, "bathkind", "bath2"))
    omega_min = parse(Float64, get_opt(opts, "omega-min", -2.0))
    omega_max = parse(Float64, get_opt(opts, "omega-max", 2.0))
    ngrid = parse(Int, get_opt(opts, "ngrid", 4001))

    default_site_j = nsites == 2 ? 2 : nsites
    site_i = parse(Int, get_opt(opts, "site-i", 1))
    site_j = parse(Int, get_opt(opts, "site-j", default_site_j))

    1 <= site_i <= nsites || error("site-i must be between 1 and nsites")
    1 <= site_j <= nsites || error("site-j must be between 1 and nsites")
    site_i != site_j || error("site-i and site-j must be different")

    omega_star, residual = find_root(
        nsites, t, mu, beta, Gamma, tau_c, bathkind, site_i, site_j;
        omega_min=omega_min, omega_max=omega_max, ngrid=ngrid,
    )

    sys, h = heff_block(nsites, t, omega_star, mu, beta, Gamma, tau_c, bathkind)
    diagdiff = h[site_i, site_i] - h[site_j, site_j]

    println("Parameters:")
    println("  nsites   = ", nsites)
    println("  t        = ", t)
    println("  mu       = ", mu)
    println("  beta     = ", beta)
    println("  Gamma    = ", Gamma)
    println("  tau_c    = ", tau_c)
    println("  bathkind = ", bathkind)
    println("  target   = Re[h[$site_i,$site_i] - h[$site_j,$site_j]] = 0")
    println()
    println("Solution:")
    println("  omega*   = ", omega_star)
    println("  residual = ", residual)
    println("  diagdiff = ", diagdiff)
    println("  h =")
    display(h)
    println()
    println("Check:")
    println("  Re[h[$site_i,$site_i] - h[$site_j,$site_j]] = ", real(diagdiff))
    println("  Im[h[$site_i,$site_i] - h[$site_j,$site_j]] = ", imag(diagdiff))

    if nsites >= 2
        println()
        println("Bond asymmetries h[i,i+1] - h[i+1,i]:")
        for i in 1:(nsites - 1)
            println("  bond ", i, "-", i + 1, " : ", h[i, i + 1] - h[i + 1, i])
        end
    end
end

main(ARGS)
