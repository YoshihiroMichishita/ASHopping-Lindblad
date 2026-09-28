push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..")))

include(joinpath(@__DIR__, "..", "src2", "ASHoppingGKSL1P.jl"))
using .ASHoppingGKSL1P

using CSV
using DataFrames
using LaTeXStrings
using Plots
using Plots.PlotMeasures: mm

function parse_args(args)
    opts = Dict{String, String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        startswith(arg, "--") || error("Unexpected argument: $arg")
        i == length(args) && error("Missing value for $arg")
        opts[arg[3:end]] = args[i + 1]
        i += 2
    end
    return opts
end

get_opt(opts, key, default) = get(opts, key, string(default))

function mu_from_path(path::AbstractString)
    dirname = basename(path)
    startswith(dirname, "mu_") || error("Expected directory name starting with mu_: $dirname")
    return parse(Float64, dirname[length("mu_") + 1:end])
end

function collect_observables_over_mu(rootdir::AbstractString;
                                     filename::AbstractString,
                                     t::Real)
    mu_dirs = filter(path -> isdir(path) && startswith(basename(path), "mu_"),
                     readdir(rootdir; join=true))
    isempty(mu_dirs) && error("No mu_* directories found in $rootdir")

    records = NamedTuple[]
    for dir in mu_dirs
        mu = mu_from_path(dir)
        obs = ASHoppingGKSL1P.local_observables_from_file(
            joinpath(dir, filename), t; format=:jld2,
        )
        push!(records, (
            mu=mu,
            current=obs.current,
            current_abs=abs(obs.current),
            beta_eff=obs.beta_eff,
            temp_eff=obs.temp_eff,
            mu_eff=obs.mu_eff,
            beta_eff_energy=obs.beta_eff_energy,
            temp_eff_energy=obs.temp_eff_energy,
            mu_eff_energy=obs.mu_eff_energy,
            energy_loc=obs.energy_loc,
            number_loc=obs.number_loc,
        ))
    end

    sort!(records; by=record -> record.mu)
    return DataFrame(records)
end

function build_current_temperature_plot(table::DataFrame)
    figure = plot(
        table.mu,
        table.current_abs;
        size=(600, 400),
        color=1,
        linewidth=1,
        marker=:circle,
        xlabel=L"\mu",
        ylabel=L"|J|",
        label=L"|J|",
        legend=:topleft,
        grid=true,
        tickfontsize=14,
        guidefontsize=16,
        legendfontsize=14,
        titlefontsize=18,
        left_margin=4mm,
        right_margin=8mm,
        bottom_margin=5mm,
        top_margin=5mm,
    )
    plot!(
        twinx(),
        table.mu,
        table.temp_eff;
        color=2,
        markercolor=2,
        linewidth=1,
        marker=:square,
        ylabel=L"T_{\mathrm{eff}}",
        label="",
        legend=false,
        tickfontsize=14,
        guidefontsize=18,
        legendfontsize=18,
        titlefontsize=20,
        left_margin=4mm,
        right_margin=8mm,
        bottom_margin=5mm,
        top_margin=5mm,
    )
    plot!(
        figure,
        Float64[],
        Float64[];
        subplot=1,
        color=2,
        markercolor=2,
        linewidth=1,
        marker=:square,
        label=L"T_{\mathrm{eff}}",
    )
    return figure
end

function render_plot(table::DataFrame, output_path::AbstractString)
    savefig(build_current_temperature_plot(table), output_path)
    return output_path
end

function main(args)
    opts = parse_args(args)
    input_dir = abspath(get_opt(opts, "indir", joinpath("out", "nr_mu_sweep6")))
    output_dir = abspath(get_opt(opts, "outdir", input_dir))
    filename = get_opt(opts, "filename", "nr_correlator.jld2")
    t = parse(Float64, get_opt(opts, "t", 1.0))

    table = collect_observables_over_mu(input_dir; filename, t)
    mkpath(output_dir)

    csv_path = joinpath(output_dir, "current_temperature_vs_mu.csv")
    png_path = joinpath(output_dir, "current_temperature_vs_mu.png")
    pdf_path = joinpath(output_dir, "current_temperature_vs_mu.pdf")
    CSV.write(csv_path, table)
    render_plot(table, png_path)
    render_plot(table, pdf_path)

    println("μ range = ", first(table.mu), " to ", last(table.mu))
    println("Saved data to: ", csv_path)
    println("Saved plots to: ", png_path, " and ", pdf_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
