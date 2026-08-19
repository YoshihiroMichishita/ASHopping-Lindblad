using Test
using CSV
using DataFrames
using JLD2

project_dir = abspath(joinpath(@__DIR__, ".."))
script_path = joinpath(project_dir, "script", "compute_local_observables.jl")

@testset "local observables CLI" begin
    output = read(`julia --project=$project_dir $script_path --method nr --nsites 4 --t 1.0 --omega 0.0 --mu 0.8 --beta 8.0 --Gamma 1.0 --tau-c 0.2`, String)
    @test occursin("beta_eff", output)
    @test occursin("mu_eff", output)
    @test occursin("current", output)
    @test occursin("central_indices", output)
    @test occursin("beta_eff_energy", output)
    @test occursin("temp_eff_energy", output)
    @test occursin("mu_eff_energy", output)
    @test occursin("energy_loc", output)
    @test occursin("number_loc", output)
end

@testset "local temperature method comparison CLI" begin
    comparison_script = joinpath(project_dir, "script", "compare_local_temperature_methods.jl")

    mktempdir() do tmpdir
        input_dir = joinpath(tmpdir, "input")
        output_dir = joinpath(tmpdir, "output")

        for (gamma, beta, mu) in ((0.05, 0.8, 0.1), (0.10, 1.2, -0.2), (0.45, 1.6, 0.0))
            gamma_dir = joinpath(input_dir, "Gamma_$(gamma)")
            mkpath(gamma_dir)

            n_minus = 1 / (exp(beta * (-1.0 - mu)) + 1)
            n_plus = 1 / (exp(beta * (1.0 - mu)) + 1)
            d = 0.5 * (n_minus + n_plus)
            r = 0.5 * (n_minus - n_plus)
            C = ComplexF64[
                0.5 0 0 0
                0 d r 0
                0 r d 0
                0 0 0 0.5
            ]
            jldsave(joinpath(gamma_dir, "nr_correlator.jld2"); matrix=C)
        end

        output = read(`julia --project=$project_dir $comparison_script --indir $input_dir --outdir $output_dir --gamma-min 0.05 --gamma-max 0.10 --gamma-step 0.05`, String)

        @test occursin("maximum |delta beta_eff|", output)
        @test isfile(joinpath(output_dir, "local_temperature_methods_gamma.csv"))
        @test isfile(joinpath(output_dir, "local_temperature_methods_gamma.png"))
        @test isfile(joinpath(output_dir, "local_temperature_methods_gamma.pdf"))

        table = CSV.read(joinpath(output_dir, "local_temperature_methods_gamma.csv"), DataFrame)
        @test table.Gamma == [0.05, 0.10]
        @test table.beta_eff_existing ≈ [0.8, 1.2] atol=1e-10
        @test table.mu_eff_energy ≈ [0.1, -0.2] atol=1e-10
        @test maximum(abs.(table.delta_beta_eff)) < 1e-12
        @test maximum(abs.(table.delta_mu_eff)) < 1e-12
    end
end
