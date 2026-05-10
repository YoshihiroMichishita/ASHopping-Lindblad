using Test
using LinearAlgebra
using IsThereSkinEffect

function fourier_matrix(L::Int)
    ks = 2π .* (0:(L - 1)) ./ L
    U = Matrix{ComplexF64}(undef, L, L)
    for n in 1:L
        for m in 1:L
            U[n, m] = exp(im * ks[m] * (n - 1)) / sqrt(L)
        end
    end
    return U, ks
end

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

@testset "Eq. (15) dissipation kernels" begin
    params = ModelParams(8, 1.0, 0.4)
    gamma_gain, gamma_loss = dissipation_kernels_new(params, PBC())

    @test ishermitian(gamma_gain)
    @test ishermitian(gamma_loss)

    U, ks = fourier_matrix(params.L)
    gain_k = U' * gamma_gain * U
    loss_k = U' * gamma_loss * U
    expected_gain = params.J .* (2 .+ 2 .* sin.(ks))
    expected_loss = params.J .* (2 .- 2 .* sin.(ks))

    @test maximum(abs.(diag(gain_k) .- expected_gain)) < 1e-10
    @test maximum(abs.(diag(loss_k) .- expected_loss)) < 1e-10
    @test maximum(abs.(gain_k .- Diagonal(diag(gain_k)))) < 1e-10
    @test maximum(abs.(loss_k .- Diagonal(diag(loss_k)))) < 1e-10
end

@testset "Eq. (15) steady state integration" begin
    params = ModelParams(12, 1.0, 0.4)
    comparison = compare_boundaries(params; dissipation = :eq15)

    @test steady_state_residual(comparison.pbc) < 1e-10
    @test steady_state_residual(comparison.obc) < 1e-10
    @test ishermitian(comparison.pbc.correlation)
    @test ishermitian(comparison.obc.correlation)
    @test maximum(abs.(comparison.pbc.density .- comparison.pbc.density[1])) < 1e-10
    @test maximum(abs.(comparison.pbc.currents .- comparison.pbc.currents[1])) < 1e-10
    @test abs(current_expectation(comparison.pbc)) > 1e-6
    @test abs(current_expectation(comparison.obc)) > 1e-6

    eig_gain_obc = eigvals(Hermitian(comparison.obc.gamma_gain))
    eig_loss_obc = eigvals(Hermitian(comparison.obc.gamma_loss))
    @test minimum(real.(eig_gain_obc)) > -1e-10
    @test minimum(real.(eig_loss_obc)) > -1e-10
end
