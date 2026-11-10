using Test
using LinearAlgebra
using CSV
using DataFrames
using JLD2

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..")))
import ASHoppingGKSL as gksl

reduced_plot_script_path = abspath(joinpath(
    @__DIR__, "..", "script", "plot_current_temperature_vs_mu_reduced_gibbs.jl",
))

function write_reduced_gibbs_plot_fixture(path, sys, beta, mu_ref, current_imag)
    _, H = gksl.system_hamiltonian(sys)
    cs, cds = gksl.site_ops(sys)
    N = sum(cds[i] * cs[i] for i in 1:sys.n_sites)
    rho = exp(-beta * (H - mu_ref * N))
    rho ./= tr(rho)
    C = ComplexF64[tr(rho * cds[i] * cs[j]) for i in 1:sys.n_sites, j in 1:sys.n_sites]
    left = sys.n_sites ÷ 2
    C[left, left + 1] += im * current_imag
    C[left + 1, left] -= im * current_imag
    jldopen(path, "w") do file
        file["matrix"] = C
        file["metadata"] = Dict("nsites" => sys.n_sites, "t" => sys.t, "omega" => sys.omega)
    end
end

@testset "reduced Gibbs mu-sweep plotting script" begin
    @test isfile(reduced_plot_script_path)
    if isfile(reduced_plot_script_path)
        include(reduced_plot_script_path)
        script = ReducedGibbsMuSweep
        indir = mktempdir()
        sys = gksl.SystemParams(4, 1.2, 0.2)
        for (mu, beta, yim) in ((0.4, -2.0, 0.02), (0.0, 2.0, 0.0), (0.2, 1.5, -0.01))
            dir = joinpath(indir, "mu_$mu")
            mkpath(dir)
            write_reduced_gibbs_plot_fixture(joinpath(dir, "nr_correlator.jld2"), sys, beta, 0.3, yim)
        end

        # Exercise the actual command line, including the default output directory.
        project_dir = abspath(joinpath(@__DIR__, ".."))
        command = `julia --project=$project_dir $reduced_plot_script_path --indir $indir --mu-ref 0.3 --t0 0.6`
        run(addenv(command, "GKSwstype" => "100"))
        outdir = joinpath(indir, "reduced_gibbs")
        for name in ("current_temperature_vs_mu.csv", "current_temperature_vs_mu.png",
                     "current_temperature_vs_mu.pdf", "J_vs_Teff_semilog.png", "J_vs_Teff_semilog.pdf")
            @test isfile(joinpath(outdir, name))
            @test filesize(joinpath(outdir, name)) > 0
            @test !isfile(joinpath(indir, name))
        end
        table = CSV.read(joinpath(outdir, "current_temperature_vs_mu.csv"), DataFrame)
        @test table.mu == [0.0, 0.2, 0.4]
        @test table.current ≈ [0.0, -0.024, 0.048] atol=1e-12
        @test table.current_abs ≈ [0.0, 0.024, 0.048] atol=1e-12
        @test table.beta_eff ≈ [2.0, 1.5, -2.0] atol=1e-6
        @test table.temp_eff ≈ [0.5, 2 / 3, -0.5] atol=1e-7
        @test table.delta_temp_eff ≈ [-0.1, 1 / 15, -1.1] atol=1e-7
        @test table.mu_ref == fill(0.3, 3)
        @test table.t0 == fill(0.6, 3)
        @test table.nsites == fill(4, 3)
        @test table.t == fill(1.2, 3)
        @test table.omega == fill(0.2, 3)
        @test table.semilog_eligible == [false, true, false]
        @test all(table.converged)
        @test !any(table.at_boundary)
        @test table.relative_entropy[1] < 1e-12
        @test all(table.relative_entropy[2:3] .> 0)

        figure = script.build_current_temperature_plot(table)
        @test String(figure.subplots[1][:xaxis][:guide]) == raw"$\Delta\mu$"
        @test String(figure.subplots[2][:yaxis][:guide]) == raw"$T_{\mathrm{eff}}$"
        @test figure.series_list[2][:y] ≈ [0.5, 2 / 3, -0.5] atol=1e-7
        @test figure.subplots[1][:legend_position] == :topleft
        semilog = script.build_reduced_gibbs_semilog_plot(table)
        @test semilog.subplots[1][:yaxis][:scale] == :log10
        @test semilog.series_list[1][:x] ≈ [0.0, 0.024, 0.048] atol=1e-12
        @test isnan(semilog.series_list[1][:y][1])
        @test semilog.series_list[1][:y][2] ≈ 1 / 15 atol=1e-7
        @test isnan(semilog.series_list[1][:y][3])
        no_positive = DataFrame(current_abs=[0.0, 0.1, 0.2], delta_temp_eff=[-0.1, 0.0, Inf])
        empty_figure = script.build_reduced_gibbs_semilog_plot(no_positive)
        @test all(isnan, empty_figure.series_list[1][:y])
        savefig = script.savefig
        savefig(empty_figure, joinpath(outdir, "no_positive.png"))
        @test filesize(joinpath(outdir, "no_positive.png")) > 0

        # Older correlator files can supply Hamiltonian parameters explicitly.
        old_dir = mktempdir()
        mkpath(joinpath(old_dir, "mu_0.0"))
        C = Matrix{ComplexF64}(I, 4, 4) .* 0.5
        jldopen(joinpath(old_dir, "mu_0.0", "nr_correlator.jld2"), "w") do file
            file["C"] = C
        end
        @test_throws ArgumentError script.collect_reduced_gibbs_observables_over_mu(old_dir)
        old_table = script.collect_reduced_gibbs_observables_over_mu(old_dir; t=1.2, omega=0.2)
        @test old_table.temp_eff == [Inf]
        @test old_table.at_boundary == [false]
        @test old_table.semilog_eligible == [false]
        @test_throws ArgumentError script.collect_reduced_gibbs_observables_over_mu(indir; t0=NaN)
    end
end
