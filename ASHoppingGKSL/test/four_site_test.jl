using Test
using LinearAlgebra
import ASHoppingGKSL as gksl

function expected_H0_4site(t::Float64, omega::Float64)
    return -t * [
        0 1 0 0
        1 0 1 0
        0 1 0 1
        0 0 1 0
    ] - omega * Diagonal([1.5, 0.5, -0.5, -1.5])
end

function build_model_4site(omega::Float64; t::Float64 = 1.0, mu::Float64 = 0.8,
                           beta::Float64 = 8.0, Gamma::Float64 = 1.0, tau_c::Float64 = 0.2,
                           bathkind::Symbol = :bath2)
    sys = gksl.SystemParams(4, t, omega)
    if bathkind == :bath
        bath1 = gksl.make_bath(Gamma=Gamma, mu=-mu, beta=beta, tau_c=tau_c)
        bath2 = gksl.make_bath(Gamma=Gamma, mu=mu, beta=beta, tau_c=tau_c)
    elseif bathkind == :bath2
        bath1 = gksl.make_bath2(Gamma=Gamma, mu=-mu, beta=beta, tau_c=tau_c)
        bath2 = gksl.make_bath2(Gamma=Gamma, mu=mu, beta=beta, tau_c=tau_c)
    else
        error("bathkind must be :bath or :bath2")
    end
    return sys, gksl.TwoBathModel(sys, bath1, bath2)
end

function heff_block_4site(omega::Float64; kwargs...)
    sys, model = build_model_4site(omega; kwargs...)
    _, Hsys = gksl.system_hamiltonian(sys)
    Hls, Hg = gksl.redfield_Heff(model)
    return sys, Hls, Hg, gksl.one_particle_block(Hsys + Hls + Hg, sys)
end

function re_edge_diagdiff_4site(omega::Float64; kwargs...)
    _, _, _, h = heff_block_4site(omega; kwargs...)
    return real(h[1, 1] - h[4, 4])
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

function find_root_4site(; omega_min::Float64 = -2.0, omega_max::Float64 = 2.0,
                         ngrid::Int = 4001, kwargs...)
    f(omega) = re_edge_diagdiff_4site(omega; kwargs...)
    xs = collect(range(omega_min, omega_max; length=ngrid))
    vals = f.(xs)

    for i in 1:(length(xs) - 1)
        if vals[i] == 0
            return xs[i]
        elseif signbit(vals[i]) != signbit(vals[i + 1])
            return bisect_root(f, xs[i], xs[i + 1])
        end
    end

    idx = argmin(abs.(vals))
    error("No sign change found. Nearest point: omega=$(xs[idx]), residual=$(vals[idx])")
end

@testset "4-site smoke test" begin
    t = 1.0
    omega = 0.3
    sys = gksl.SystemParams(4, t, omega)

    @testset "Operator construction" begin
        c_ops, cd_ops = gksl.site_ops(sys)
        @test length(c_ops) == 4
        @test length(cd_ops) == 4

        d = 2^4
        for i in 1:4
            @test size(c_ops[i]) == (d, d)
            @test size(cd_ops[i]) == (d, d)
            @test norm(cd_ops[i] - c_ops[i]') < 1e-12
        end
    end

    @testset "Single-particle consistency" begin
        H0, Hsys = gksl.system_hamiltonian(sys)
        hsys = gksl.one_particle_block(Hsys, sys)

        @test size(H0) == (4, 4)
        @test size(hsys) == (4, 4)
        @test norm(H0 - expected_H0_4site(t, omega)) < 1e-12
        @test norm(hsys - ComplexF64.(H0)) < 1e-12
    end

    @testset "Normal modes and couplings" begin
        eps, U = gksl.single_particle_eigensystem(sys)
        eps2, U2, d_modes, dd_modes = gksl.normal_modes(sys)

        @test length(eps) == 4
        @test size(U) == (4, 4)
        @test eps == eps2
        @test norm(abs.(U) - abs.(U2)) < 1e-12
        @test length(d_modes) == 4
        @test length(dd_modes) == 4

        for a in 1:4
            @test norm(dd_modes[a] - d_modes[a]') < 1e-12
        end

        for site in (1, 4)
            ops = gksl.coupling_ops_by_frequency(sys, site)
            freqs = sort(unique(w for ((_, w), _) in ops))
            @test !isempty(freqs)
            for ch in (:alpha, :beta), w in freqs
                @test haskey(ops, (ch, w))
                @test size(ops[(ch, w)]) == (16, 16)
            end
        end
    end

    @testset "Redfield Heff and root solve" begin
        sys4, model = build_model_4site(0.3)
        Hls, Hg = gksl.redfield_Heff(model)
        _, Hsys = gksl.system_hamiltonian(sys4)
        heff = gksl.one_particle_block(Hsys + Hls + Hg, sys4)

        @test size(Hls) == (16, 16)
        @test size(Hg) == (16, 16)
        @test size(heff) == (4, 4)
        @test norm(Hls - Hls') < 1e-10
        @test isfinite(norm(heff))

        omega_star = find_root_4site()
        residual = re_edge_diagdiff_4site(omega_star)
        _, _, _, h_star = heff_block_4site(omega_star)

        @test abs(residual) < 1e-10
        @test size(h_star) == (4, 4)
        @test isfinite(norm(h_star))
    end
end
