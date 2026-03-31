using Test
using LinearAlgebra
using IsThereSkinEffect

@testset "Boundary comparison" begin
    params = ModelParams(12, 1.0, 0.4)
    comparison = compare_boundaries(params)

    @test length(comparison.pbc.density) == params.L
    @test length(comparison.obc.density) == params.L
    @test ishermitian(comparison.pbc.correlation)
    @test ishermitian(comparison.obc.correlation)
    @test steady_state_residual(comparison.pbc) < 1e-10
    @test steady_state_residual(comparison.obc) < 1e-10
    @test maximum(abs.(comparison.pbc.density .- comparison.pbc.density[1])) < 1e-10
    @test maximum(abs.(comparison.pbc.currents .- comparison.pbc.currents[1])) < 1e-10
    @test abs(current_expectation(comparison.pbc)) > 1e-6
    @test abs(current_expectation(comparison.obc)) > 1e-6

    eig_pbc = eigvals(Hermitian(comparison.pbc.correlation))
    eig_obc = eigvals(Hermitian(comparison.obc.correlation))
    @test minimum(real.(eig_pbc)) > -1e-10
    @test maximum(real.(eig_pbc)) < 1.0 + 1e-10
    @test minimum(real.(eig_obc)) > -1e-10
    @test maximum(real.(eig_obc)) < 1.0 + 1e-10

    cont_pbc = continuity_residuals(comparison.pbc)
    cont_obc = continuity_residuals(comparison.obc)
    @test maximum(abs.(cont_pbc)) < 1e-10
    @test maximum(abs.(cont_obc)) < 1e-10
end
