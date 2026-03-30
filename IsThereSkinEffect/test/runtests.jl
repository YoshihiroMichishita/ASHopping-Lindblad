using Test
using IsThereSkinEffect

@testset "Boundary comparison" begin
    params = ModelParams(10, 1.0, 0.3, 0.5, 3.0; mu = 0.0)
    comparison = compare_boundaries(params; tol = 1e-9, max_steps = 500_000)

    @test length(comparison.pbc.occupations) == params.L
    @test length(comparison.obc.occupations) == params.L
    @test length(comparison.pbc.density) == params.L
    @test length(comparison.obc.density) == params.L
    @test maximum(abs.(comparison.pbc.density .- comparison.pbc.density[1])) < 1e-10
    @test all(0.0 .<= comparison.pbc.occupations .<= 1.0)
    @test all(0.0 .<= comparison.obc.occupations .<= 1.0)
    @test steady_state_residual(comparison.pbc, params) < 1e-9
    @test steady_state_residual(comparison.obc, params) < 1e-9
    @test abs(current_expectation(comparison.pbc, params)) > 1e-6
    @test abs(current_expectation(comparison.obc, params)) > 1e-6
    @test maximum(abs.(coherent_bond_currents(comparison.obc, params))) < 1e-12
end
