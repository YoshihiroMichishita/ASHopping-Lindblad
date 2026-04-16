push!(LOAD_PATH, "/Users/michishita-su/Documents/Code/ASHopping-Lindblad/ASHoppingGKSL")
import ASHoppingGKSL as gksl
using LinearAlgebra

function one_particle_basis(p)
    _, cds = gksl.site_ops(p)
    d = size(cds[1], 1)
    vac = zeros(ComplexF64, d)
    vac[1] = 1.0
    return [cd * vac for cd in cds]
end

function one_particle_block(Op, p)
    basis = one_particle_basis(p)
    n = length(basis)
    h = zeros(ComplexF64, n, n)
    for i in 1:n, j in 1:n
        h[i, j] = basis[i]' * Op * basis[j]
    end
    return h
end

function heff_block(mu, omega, tauc; bathkind=:bath2, beta=8.0, Gamma=1.0)
    sys = gksl.SystemParams(2, 1.0, omega)
    if bathkind == :bath
        b1 = gksl.make_bath(Gamma=Gamma, mu=-mu, beta=beta, tau_c=tauc)
        b2 = gksl.make_bath(Gamma=Gamma, mu=mu, beta=beta, tau_c=tauc)
    elseif bathkind == :bath2
        b1 = gksl.make_bath2(Gamma=Gamma, mu=-mu, beta=beta, tau_c=tauc)
        b2 = gksl.make_bath2(Gamma=Gamma, mu=mu, beta=beta, tau_c=tauc)
    else
        error("bathkind must be :bath or :bath2")
    end
    m = gksl.TwoBathModel(sys, b1, b2)
    _, Hsys = gksl.system_hamiltonian(sys)
    Hls, Hg = gksl.redfield_Heff(m)
    hsys = one_particle_block(Hsys, sys)
    hls = one_particle_block(Hls, sys)
    hg = one_particle_block(Hg, sys)
    return hsys, hls, hg, hsys + hls + hg
end

function re_diagdiff(mu, omega, tauc; kwargs...)
    _, _, _, h = heff_block(mu, omega, tauc; kwargs...)
    return real(h[1, 1] - h[2, 2])
end

function bisect_root(f, a, b; tol=1e-12, maxiter=200)
    fa = f(a)
    fb = f(b)
    fa == 0 && return a
    fb == 0 && return b
    signbit(fa) == signbit(fb) && error("root is not bracketed: f(a)=$fa, f(b)=$fb")

    left, right = a, b
    fleft, fright = fa, fb
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
            fright = fmid
        end
    end
    return (left + right) / 2
end

function bracket_and_solve(mu, tauc; bathkind=:bath2, beta=8.0, Gamma=1.0, wmin=-2.0, wmax=2.0, ngrid=4001)
    f(w) = re_diagdiff(mu, w, tauc; bathkind=bathkind, beta=beta, Gamma=Gamma)
    xs = collect(range(wmin, wmax; length=ngrid))
    vals = f.(xs)
    for i in 1:(length(xs) - 1)
        if vals[i] == 0
            return xs[i]
        elseif signbit(vals[i]) != signbit(vals[i + 1])
            return bisect_root(f, xs[i], xs[i + 1])
        end
    end
    idx = argmin(abs.(vals))
    error("no sign change found; nearest point is w=$(xs[idx]) with residual=$(vals[idx])")
end

function show_case(mu, tauc, omega; bathkind=:bath2, beta=8.0, Gamma=1.0)
    hsys, hls, hg, h = heff_block(mu, omega, tauc; bathkind=bathkind, beta=beta, Gamma=Gamma)
    println("bathkind=", bathkind, ", mu=", mu, ", tau_c=", tauc, ", beta=", beta, ", Gamma=", Gamma, ", omega=", omega)
    println("hsys =")
    display(hsys)
    println("hls =")
    display(hls)
    println("hg =")
    display(hg)
    println("htotal =")
    display(h)
    println("h11-h22 = ", h[1, 1] - h[2, 2])
    println("Re[h11-h22] = ", real(h[1, 1] - h[2, 2]))
    println("Im[h11-h22] = ", imag(h[1, 1] - h[2, 2]))
    println("h12-h21 = ", h[1, 2] - h[2, 1])
    println()
end

mu = 0.8
tauc = 0.2
beta = 8.0
Gamma = 1.0

println("=== Fixed omega = -0.34 ===")
show_case(mu, tauc, -0.34; bathkind=:bath, beta=beta, Gamma=Gamma)
show_case(mu, tauc, -0.34; bathkind=:bath2, beta=beta, Gamma=Gamma)

println("=== Root-solved omega for Re[h11-h22] = 0 ===")
for bathkind in (:bath, :bath2)
    omega_star = bracket_and_solve(mu, tauc; bathkind=bathkind, beta=beta, Gamma=Gamma)
    println("omega_star(", bathkind, ") = ", omega_star)
    show_case(mu, tauc, omega_star; bathkind=bathkind, beta=beta, Gamma=Gamma)
end
