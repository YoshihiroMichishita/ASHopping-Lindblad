using Test
using LinearAlgebra

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..")))
import ASHoppingGKSL as gksl

include(joinpath(@__DIR__, "..", "src2", "ASHoppingGKSL1P.jl"))
using .ASHoppingGKSL1P

function build_src2_model(; nsites::Int = 4, t::Float64 = 1.0, omega::Float64 = 0.0,
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

function correlator_from_rho(rho, sys)
    c_ops, cd_ops = gksl.site_ops(sys)
    n = sys.n_sites
    C = zeros(ComplexF64, n, n)
    for i in 1:n, j in 1:n
        C[i, j] = tr(cd_ops[i] * c_ops[j] * rho)
    end
    return C
end

@testset "src2 one-particle correlators" begin
    sys, model = build_src2_model()

    @testset "reduced-space helpers" begin
        c_red = ASHoppingGKSL1P.reduced_annihilation_ops(sys.n_sites)
        cd_red = ASHoppingGKSL1P.reduced_creation_ops(sys.n_sites)
        ids, ops = ASHoppingGKSL1P.reduced_bilinear_basis(sys.n_sites)
        reduced_dim = 1 + sys.n_sites + (sys.n_sites * (sys.n_sites - 1)) ÷ 2

        @test length(c_red) == sys.n_sites
        @test length(cd_red) == sys.n_sites
        @test size(c_red[1]) == (reduced_dim, reduced_dim)
        @test c_red === ASHoppingGKSL1P.reduced_annihilation_ops(sys.n_sites)
        @test cd_red === ASHoppingGKSL1P.reduced_creation_ops(sys.n_sites)
        @test c_red[1][1, 2] == 1.0
        @test cd_red[1][2, 1] == 1.0
        @test c_red[2][2, 1 + sys.n_sites + 1] == -1.0
        @test c_red[1][3, 1 + sys.n_sites + 1] == 1.0
        @test ids[1] == :I
        @test length(ops) == 1 + sys.n_sites^2

        Ored = zeros(ComplexF64, reduced_dim, reduced_dim)
        Ored[1, 1] = 0.25
        Ored[2:1 + sys.n_sites, 2:1 + sys.n_sites] .= 0.25 .* Matrix{ComplexF64}(I, sys.n_sites, sys.n_sites)
        Ored[2, 3] += 0.4
        s, X = ASHoppingGKSL1P.source_and_bilinear_from_reduced(Ored)
        @test s ≈ 0.25 atol = 1e-12
        @test X[1, 2] ≈ 0.4 atol = 1e-12
    end

    @testset "basis construction" begin
        basis = ASHoppingGKSL1P.bilinear_basis(sys)
        @test length(basis.ops) == 1 + sys.n_sites^2
        @test basis.ids[1] == :I
        @test size(basis.gram) == (1 + sys.n_sites^2, 1 + sys.n_sites^2)
    end

    @testset "NR correlator matches full-rho reference" begin
        _, rho_ref, _ = gksl.steady_states(model)
        C_ref = correlator_from_rho(rho_ref, sys)
        C_new = ASHoppingGKSL1P.steady_correlator_nr(model)
        @test size(C_new) == (sys.n_sites, sys.n_sites)
        @test C_new ≈ C_ref atol = 1e-8
    end

    @testset "Redfield correlator matches full-rho reference" begin
        rho_ref, _, _ = gksl.steady_states(model)
        C_ref = correlator_from_rho(rho_ref, sys)
        C_new = ASHoppingGKSL1P.steady_correlator_redfield(model)
        @test size(C_new) == (sys.n_sites, sys.n_sites)
        @test C_new ≈ C_ref atol = 1e-8
    end

    @testset "weighted observables are reconstructed from C" begin
        loss_coeff = ComplexF64[0.2 0.1; 0.3 0.4]
        gain_coeff = ComplexF64[0.7 0.2; 0.1 0.6]
        C = ComplexF64[0.4 0.05; 0.07 0.3]
        weighted = ASHoppingGKSL1P.loss_gain_weighted_from_C(loss_coeff, gain_coeff, C)

        @test weighted.loss_weighted ≈ loss_coeff .* transpose(C) atol = 1e-12
        @test weighted.gain_weighted ≈ gain_coeff .* (Matrix{ComplexF64}(I, 2, 2) - C) atol = 1e-12
    end
end
