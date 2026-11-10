using Test
using LinearAlgebra
using CSV
using DataFrames
using JLD2

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..")))
import ASHoppingGKSL as gksl

gamma_plot_script_path = abspath(joinpath(
    @__DIR__, "..", "script", "plot_current_temperature_vs_gamma_reduced_gibbs.jl",
))

function write_gamma_gibbs_fixture(path, sys, beta, mu_ref, current_imag, gamma)
    _, H = gksl.system_hamiltonian(sys)
    cs, cds = gksl.site_ops(sys)
    N = sum(cds[i] * cs[i] for i in 1:sys.n_sites)
    rho = exp(-beta * (H - mu_ref * N))
    rho ./= tr(rho)
    C = ComplexF64[tr(rho * cds[i] * cs[j]) for i in 1:sys.n_sites, j in 1:sys.n_sites]
    m = sys.n_sites ÷ 2
    C[m, m + 1] += im * current_imag
    C[m + 1, m] -= im * current_imag
    jldopen(path, "w") do file
        file["matrix"] = C
        file["metadata"] = Dict("nsites" => sys.n_sites, "t" => sys.t, "omega" => sys.omega,
                                "Gamma" => gamma, "mu" => 2.4, "beta" => 10.0, "tau_c" => 0.002)
    end
end

@testset "reduced Gibbs gamma-sweep plotting script" begin
    @test isfile(gamma_plot_script_path)
    if isfile(gamma_plot_script_path)
        include(gamma_plot_script_path)
        script = ReducedGibbsGammaSweep
        indir = mktempdir()
        sys = gksl.SystemParams(4, 1.2, 0.2)
        for (gamma, beta, yim) in ((0.4, -2.0, 0.02), (0.05, 2.0, 0.0), (0.1, 1.5, -0.01))
            dir = joinpath(indir, "Gamma_$gamma")
            mkpath(dir)
            write_gamma_gibbs_fixture(joinpath(dir, "nr_correlator.jld2"), sys, beta, 0.3, yim, gamma)
        end
        mkpath(joinpath(indir, "mu_0.5")) # Not a gamma-sweep point.
        original_path = joinpath(indir, "Gamma_0.05", "nr_correlator.jld2")
        original_bytes = read(original_path)
        project_dir = abspath(joinpath(@__DIR__, ".."))
        command = `julia --project=$project_dir $gamma_plot_script_path --indir $indir --mu-ref 0.3 --t0 0.6`
        run(addenv(command, "GKSwstype" => "100"))
        outdir = joinpath(indir, "reduced_gibbs")
        for name in ("current_temperature_vs_gamma.csv", "JandTeff_vs_gamma.png",
                     "JandTeff_vs_gamma.pdf", "J_vs_Teff_semilog.png", "J_vs_Teff_semilog.pdf")
            @test isfile(joinpath(outdir, name))
            @test filesize(joinpath(outdir, name)) > 0
            @test !isfile(joinpath(indir, name))
        end
        @test read(original_path) == original_bytes
        table = CSV.read(joinpath(outdir, "current_temperature_vs_gamma.csv"), DataFrame)
        @test table.Gamma == [0.05, 0.1, 0.4]
        @test table.current ≈ [0.0, -0.024, 0.048] atol=1e-12
        @test table.current_abs ≈ [0.0, 0.024, 0.048] atol=1e-12
        @test table.beta_eff ≈ [2.0, 1.5, -2.0] atol=1e-6
        @test table.temp_eff ≈ [0.5, 2 / 3, -0.5] atol=1e-7
        @test table.delta_temp_eff ≈ [-0.1, 1 / 15, -1.1] atol=1e-7
        @test table.mu_ref == fill(0.3, 3)
        @test table.mu == fill(2.4, 3)
        @test table.beta_bath == fill(10.0, 3)
        @test table.bath_Gamma == [0.05, 0.1, 0.4]
        @test table.t0 == fill(0.6, 3)
        @test table.t == fill(1.2, 3)
        @test table.omega == fill(0.2, 3)
        @test all(table.converged)
        @test !any(table.at_boundary)
        @test table.semilog_eligible == [false, true, false]
        figure = script.build_current_temperature_gamma_plot(table)
        @test String(figure.subplots[1][:xaxis][:guide]) == raw"$\gamma$"
        @test String(figure.subplots[2][:yaxis][:guide]) == raw"$T_{\mathrm{eff}}$"
        @test figure.series_list[1][:x] == [0.05, 0.1, 0.4]
        @test figure.series_list[2][:x] == [0.05, 0.1, 0.4]
        @test figure.series_list[2][:y] ≈ [0.5, 2 / 3, -0.5] atol=1e-7
        @test figure.subplots[1][:legend_position] == :topleft
        @test figure.subplots[2][:legend_position] == :none
        @test table.mu == fill(2.4, 3) # Building the plot must not replace the physical mu.
        semilog = script.build_reduced_gibbs_semilog_plot(table)
        @test semilog.subplots[1][:yaxis][:scale] == :log10
        @test isnan(semilog.series_list[1][:y][1])
        @test semilog.series_list[1][:y][2] ≈ 1 / 15 atol=1e-7
        @test isnan(semilog.series_list[1][:y][3])
        for gamma in (0.01, 0.5)
            dir = joinpath(indir, "Gamma_$gamma")
            mkpath(dir)
            write_gamma_gibbs_fixture(joinpath(dir, "nr_correlator.jld2"), sys, 2.0, 0.3, 0.0, gamma)
        end
        restricted = script.collect_reduced_gibbs_observables_over_gamma(indir; mu_ref=0.3)
        @test restricted.Gamma == [0.05, 0.1, 0.4]
        @test_throws ArgumentError script.collect_reduced_gibbs_observables_over_gamma(indir; t0=NaN)
        @test_throws ErrorException script.collect_reduced_gibbs_observables_over_gamma(mktempdir())
    end
end
