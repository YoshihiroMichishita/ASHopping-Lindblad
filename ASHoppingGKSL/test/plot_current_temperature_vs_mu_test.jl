using Test
using CSV
using DataFrames
using JLD2
using Plots.PlotMeasures: mm

project_dir = abspath(joinpath(@__DIR__, ".."))
script_path = joinpath(project_dir, "script", "plot_current_temperature_vs_mu.jl")

function write_correlator_fixture(path::AbstractString, beta::Float64, imag_offdiag::Float64)
    real_offdiag = 0.5 * tanh(beta / 2)
    correlator = ComplexF64[
        0.5 real_offdiag + im * imag_offdiag
        real_offdiag - im * imag_offdiag 0.5
    ]
    jldopen(path, "w") do file
        file["matrix"] = correlator
    end
end

function png_dimensions(path::AbstractString)
    bytes = read(path)
    bytes[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] ||
        error("Not a PNG file: $path")
    decode_uint32(offset) = foldl((value, byte) -> (value << 8) | Int(byte),
                                  bytes[offset:offset + 3]; init=0)
    return decode_uint32(17), decode_uint32(21)
end

@testset "current and effective temperature versus mu plot" begin
    indir = mktempdir()
    outdir = mktempdir()

    fixtures = [
        ("mu_0.4", 1.0, 0.02),
        ("mu_0.0", 2.0, 0.00),
        ("mu_0.2", 1.5, -0.01),
    ]
    for (dirname, beta, imag_offdiag) in fixtures
        dir = joinpath(indir, dirname)
        mkpath(dir)
        write_correlator_fixture(joinpath(dir, "nr_correlator.jld2"), beta, imag_offdiag)
    end

    command = `julia --project=$project_dir $script_path --indir $indir --outdir $outdir --filename nr_correlator.jld2 --t 1.0`
    run(addenv(command, "GKSwstype" => "100"))

    csv_path = joinpath(outdir, "current_temperature_vs_mu.csv")
    png_path = joinpath(outdir, "current_temperature_vs_mu.png")
    pdf_path = joinpath(outdir, "current_temperature_vs_mu.pdf")
    current_temperature_png_path = joinpath(outdir, "J_vs_Teff_semilog.png")
    current_temperature_pdf_path = joinpath(outdir, "J_vs_Teff_semilog.pdf")
    @test isfile(csv_path)
    @test isfile(png_path)
    @test isfile(pdf_path)
    @test isfile(current_temperature_png_path)
    @test isfile(current_temperature_pdf_path)
    @test filesize(png_path) > 0
    @test filesize(pdf_path) > 0
    @test filesize(current_temperature_png_path) > 0
    @test filesize(current_temperature_pdf_path) > 0
    @test png_dimensions(png_path) == (600, 400)
    @test png_dimensions(current_temperature_png_path) == (600, 400)
    @test !isfile(joinpath(outdir, "J_vs_Teff.png"))
    @test !isfile(joinpath(outdir, "J_vs_Teff.pdf"))

    table = CSV.read(csv_path, DataFrame)
    @test table.mu == [0.0, 0.2, 0.4]
    @test table.current ≈ [0.0, -0.02, 0.04] atol=1e-12
    @test table.current_abs ≈ [0.0, 0.02, 0.04] atol=1e-12
    @test table.beta_eff ≈ [2.0, 1.5, 1.0] atol=1e-10
    @test table.temp_eff ≈ [0.5, 2 / 3, 1.0] atol=1e-10
    @test table.delta_temp_eff ≈ [0.4, 17 / 30, 0.9] atol=1e-10
    @test table.mu_eff ≈ zeros(3) atol=1e-10
    @test table.beta_eff_energy ≈ table.beta_eff atol=1e-10
    @test table.temp_eff_energy ≈ table.temp_eff atol=1e-10
    @test table.mu_eff_energy ≈ table.mu_eff atol=1e-10

    saved_args = copy(ARGS)
    empty!(ARGS)
    append!(ARGS, ["--indir", indir, "--outdir", outdir,
                   "--filename", "nr_correlator.jld2", "--t", "1.0"])
    try
        include(script_path)
    finally
        empty!(ARGS)
        append!(ARGS, saved_args)
    end
    has_plot_builder = isdefined(@__MODULE__, :build_current_temperature_plot)
    @test has_plot_builder
    if has_plot_builder
        p = build_current_temperature_plot(table)
        @test p.attr[:size] == (600, 400)
        @test length(p.series_list) == 3
        @test p.series_list[1][:markershape] == :circle
        @test p.series_list[1][:linewidth] == 1
        @test p.series_list[2][:markershape] == :rect
        @test p.series_list[2][:linewidth] == 1
        @test p.series_list[2][:y] ≈ [0.5, 2 / 3, 1.0] atol=1e-10
        if length(p.series_list) >= 3
            @test p.series_list[3][:markershape] == :rect
            @test p.series_list[3][:linewidth] == 1
        end

        left_axis, right_axis = p.subplots
        @test left_axis[:legend_position] == :topleft
        @test right_axis[:legend_position] == :none
        left_legend_labels = [
            String(series[:label]) for series in p.series_list
            if series[:subplot][:subplot_index] == 1 && !isempty(String(series[:label]))
        ]
        @test left_legend_labels == [raw"$|J|$", raw"$T_{\mathrm{eff}}$"]
        @test isempty(String(p.series_list[2][:label]))
        @test String(left_axis[:xaxis][:guide]) == raw"$\Delta\mu$"
        @test String(right_axis[:yaxis][:guide]) == raw"$T_{\mathrm{eff}}$"
        @test left_axis[:xaxis][:tickfontsize] == 14
        @test left_axis[:xaxis][:guidefontsize] == 16
        @test left_axis[:xaxis][:grid]
        @test left_axis[:left_margin] == 4mm
        @test left_axis[:bottom_margin] == 5mm
        @test right_axis[:yaxis][:tickfontsize] == 14
        @test right_axis[:yaxis][:guidefontsize] == 18
        @test right_axis[:right_margin] == 8mm
    end

    has_current_temperature_builder = isdefined(
        @__MODULE__, :build_current_delta_temperature_plot,
    )
    @test has_current_temperature_builder
    if has_current_temperature_builder
        p = build_current_delta_temperature_plot(table)
        @test p.attr[:size] == (600, 400)
        @test length(p.series_list) == 1
        @test p.series_list[1][:x] ≈ [0.0, 0.02, 0.04] atol=1e-12
        @test p.series_list[1][:y] ≈ [0.4, 17 / 30, 0.9] atol=1e-10
        @test p.series_list[1][:markershape] == :circle
        @test p.series_list[1][:linewidth] == 1

        axis = only(p.subplots)
        @test axis[:legend_position] == :none
        @test axis[:yaxis][:scale] == :log10
        @test String(axis[:xaxis][:guide]) == raw"$|J|$"
        @test String(axis[:yaxis][:guide]) == raw"$\Delta T_{\mathrm{eff}}$"
        @test axis[:xaxis][:grid]

        table_with_nonpositive_temperature = DataFrame(
            mu=[0.0, 0.2, 0.4],
            current_abs=[0.0, 0.02, 0.04],
            delta_temp_eff=[0.4, 0.0, -0.2],
        )
        full_plot = build_current_delta_temperature_plot(
            table_with_nonpositive_temperature,
        )
        @test full_plot.series_list[1][:x] ≈ [0.0, 0.02, 0.04] atol=1e-12
        @test full_plot.series_list[1][:y] ≈ [0.4, 0.0, -0.2] atol=1e-12
    end
end
