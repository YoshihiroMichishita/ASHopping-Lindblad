using Test
using LinearAlgebra
import ASHoppingGKSL as gksl

function build_gain_loss_model(; nsites::Int = 3, t::Float64 = 1.0, omega::Float64 = 0.0,
                               mu::Float64 = 0.8, beta::Float64 = 8.0,
                               Gamma::Float64 = 1.0, tau_c::Float64 = 0.2,
                               bathkind::Symbol = :bath2)
    sys = gksl.SystemParams(nsites, t, omega)
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

@testset "Gain-Loss Coefficient Check" begin
    sys, model = build_gain_loss_model()
    tol = 1e-10

    @testset "Redfield one-particle diagonal relation" begin
        _, Hg = gksl.redfield_Heff(model)
        h_g = gksl.one_particle_block(Hg, sys)

        gain, loss = gksl.redfield_local_gain_loss_coeffs(model)
        net = real.(gain - loss)
        total_gain = sum(real.(gain))
        lhs = -2 .* imag.(diag(h_g))
        rhs = total_gain .- net

        @test all(isapprox.(lhs, rhs; atol=tol, rtol=0.0))
    end

    @testset "NR one-particle diagonal relation" begin
        _, Hg = gksl.nr_Heff_from_redfield(model)
        h_g = gksl.one_particle_block(Hg, sys)

        gain, loss = gksl.nr_local_gain_loss_coeffs(model)
        net = gain - loss
        total_gain = sum(gain)
        lhs = -2 .* imag.(diag(h_g))
        rhs = total_gain .- net

        @test all(isapprox.(lhs, rhs; atol=tol, rtol=0.0))
    end
end
