using Test
using LinearAlgebra

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..")))
import ASHoppingGKSL as gksl
if !isdefined(@__MODULE__, :ASHoppingGKSL1P)
    include(joinpath(@__DIR__, "..", "src2", "ASHoppingGKSL1P.jl"))
end
using .ASHoppingGKSL1P

# Independent reference: construct the Gibbs state in the full Fock space.
function gibbs_correlator_fock(sys, beta, mu)
    _, H = gksl.system_hamiltonian(sys)
    cs, cds = gksl.site_ops(sys)
    N = sum(cds[i] * cs[i] for i in 1:sys.n_sites)
    rho = exp(-beta * (H - mu * N))
    rho ./= tr(rho)
    return ComplexF64[tr(rho * cds[i] * cs[j])
                      for i in 1:sys.n_sites, j in 1:sys.n_sites]
end

function two_site_gaussian_density(C)
    # The code uses C_ij = <c_i† c_j>, hence the transpose below.
    F = eigen(Hermitian(Matrix(transpose(C))))
    k = F.vectors * Diagonal(log.((1 .- F.values) ./ F.values)) * F.vectors'
    cs, cds = gksl.site_ops(gksl.SystemParams(2, 1.0, 0.0))
    K = sum(k[i, j] * cds[i] * cs[j] for i in 1:2, j in 1:2)
    rho = exp(-K)
    return rho / tr(rho)
end

@testset "reduced full-system Gibbs temperature" begin
    @testset "thermal reference includes the surrounding sites" begin
        sys = gksl.SystemParams(6, 1.0, 0.15)
        C = gibbs_correlator_fock(sys, 1.7, 0.3)
        Cmid = C[3:4, 3:4]
        @test reduced_gibbs_correlator(sys, 1.7; mu_ref=0.3) ≈ Cmid atol=1e-12
        @test reduced_gibbs_correlator(sys, 0.0; mu_ref=0.3) ≈ 0.5I atol=1e-12
        @test norm(Cmid - reduced_gibbs_correlator(
            gksl.SystemParams(2, 1.0, 0.15), 1.7; mu_ref=0.3,
        )) > 0.01
    end

    @testset "quantum relative entropy retains noncommuting coherences" begin
        C = ComplexF64[0.4 0.08+0.11im; 0.08-0.11im 0.65]
        Q = ComplexF64[0.55 0.17-0.03im; 0.17+0.03im 0.35]
        rho, sigma = two_site_gaussian_density(C), two_site_gaussian_density(Q)
        expected = real(tr(rho * (log(Hermitian(rho)) - log(Hermitian(sigma)))))
        @test norm(C * Q - Q * C) > 0.01
        @test gaussian_relative_entropy(C, Q) ≈ expected atol=1e-12
        @test gaussian_relative_entropy(C, C) ≈ 0.0 atol=1e-12
        @test gaussian_relative_entropy(C, Q) > 0
        @test abs(gaussian_relative_entropy(C, Q) - gaussian_relative_entropy(Q, C)) > 1e-4
        @test gaussian_relative_entropy(Diagonal([0.0, 1.0]), 0.5 .* Matrix(I, 2, 2)) ≈ 2log(2) atol=1e-12
    end

    @testset "fixed chemical potential and equilibrium temperature recovery" begin
        sys = gksl.SystemParams(6, 1.0, 0.15)
        for beta in (1.7, 10.0, -2.3)
            C = gibbs_correlator_fock(sys, beta, 0.3)
            fit = local_temperature_from_reduced_gibbs(
                C, sys; mu_ref=0.3, beta_bounds=(-20.0, 20.0),
            )
            @test fit.beta_eff ≈ beta atol=1e-6
            @test fit.temp_eff ≈ 1 / beta atol=1e-7
            @test fit.mu_ref == 0.3
            @test fit.central_indices == (3, 4)
            @test fit.Cmid ≈ C[3:4, 3:4] atol=1e-12
            @test fit.reference_Cmid ≈ C[3:4, 3:4] atol=1e-8
            @test fit.relative_entropy ≈ 0.0 atol=1e-12
            @test fit.fit_residual < 1e-8
            @test fit.converged
            @test !fit.at_boundary
        end
        C = gibbs_correlator_fock(sys, 1.7, 0.3)
        local_fit = local_temperature_from_reduced_gibbs(C[3:4, 3:4], sys; mu_ref=0.3)
        @test local_fit.beta_eff ≈ 1.7 atol=1e-6
    end

    @testset "current contributes a residual but does not bias the fit" begin
        sys = gksl.SystemParams(6, 1.0, 0.0)
        C = gibbs_correlator_fock(sys, 2.0, 0.0)
        C[3, 4] += 0.03im
        C[4, 3] -= 0.03im
        fit = local_temperature_from_reduced_gibbs(C, sys; beta_bounds=(0.0, 10.0))
        @test fit.beta_eff ≈ 2.0 atol=1e-6
        @test fit.relative_entropy > 0
        @test fit.fit_residual ≈ sqrt(2) * 0.03 atol=1e-8
    end

    @testset "infinite temperature, bounded search, and invalid inputs" begin
        sys = gksl.SystemParams(6, 1.0, 0.0)
        fit = local_temperature_from_reduced_gibbs(0.5 .* Matrix(I, 6, 6), sys)
        @test fit.beta_eff == 0.0
        @test fit.temp_eff == Inf
        @test fit.at_boundary
        zero_fit = local_temperature_from_reduced_gibbs(
            0.5 .* Matrix(I, 6, 6), sys; beta_bounds=(-10.0, 10.0),
        )
        @test zero_fit.beta_eff == 0.0
        @test zero_fit.temp_eff == Inf
        @test !zero_fit.at_boundary
        C = gibbs_correlator_fock(sys, 2.0, 0.0)
        bounded = local_temperature_from_reduced_gibbs(C, sys; beta_bounds=(0.0, 1.0))
        @test bounded.beta_eff == 1.0
        @test bounded.at_boundary
        @test bounded.relative_entropy > 0
        @test_throws ArgumentError local_temperature_from_reduced_gibbs(C, sys; beta_bounds=(2.0, 1.0))
        @test_throws ArgumentError local_temperature_from_reduced_gibbs(C, sys; mu_ref=NaN)
        @test_throws ArgumentError local_temperature_from_reduced_gibbs(C, sys; grid_points=2)
        @test_throws ArgumentError local_temperature_from_reduced_gibbs(C, sys; maxiter=0)
        @test_throws DimensionMismatch local_temperature_from_reduced_gibbs(zeros(3, 3), sys)
        @test_throws ArgumentError reduced_gibbs_correlator(gksl.SystemParams(5, 1.0, 0.0), 1.0)
        @test_throws ArgumentError gaussian_relative_entropy(ComplexF64[0.5 0.1; 0 0.5], 0.5 .* Matrix(I, 2, 2))
        @test_throws DomainError gaussian_relative_entropy(Diagonal([-0.1, 0.5]), 0.5 .* Matrix(I, 2, 2))
        @test_throws DomainError gaussian_relative_entropy(Diagonal([0.5, 1.1]), 0.5 .* Matrix(I, 2, 2))
        @test_throws ArgumentError local_temperature_from_reduced_gibbs(
            0.5 .* Matrix(I, 6, 6), gksl.SystemParams(6, 0.0, 0.0),
        )
    end
end
