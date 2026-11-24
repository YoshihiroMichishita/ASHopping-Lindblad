"""
Plot |J| versus delta T_eff and its quadratic fit for 0 <= mu <= 0.6.

Run from the repository root:
    julia --project=ASHoppingGKSL ASHoppingGKSL/script/plot_current_delta_temperature_linear_small_mu.jl

Reads the precomputed reduced-full-chain-Gibbs fit results in
out/nr_mu_sweep6_beta10/reduced_gibbs/current_temperature_vs_mu.csv.
No steady-state calculation or temperature fit is repeated. The default
baseline is T0=0.1; negative or zero temperature increases are retained.
Fits delta T_eff = a*|J| + b*|J|^2 by unweighted least squares using all
selected points, without a constant term, and overlays a dashed curve.
The fitted coefficients and RMSE are printed to the terminal.
Outputs PDF, PNG, and selected CSV data to
out/nr_mu_sweep6_beta10/linear_small_mu_0_0p6_fit. Existing outputs are never
overwritten. Options: --csv PATH, --outdir DIR, --t0 VALUE, --help.
"""
module SmallMuLinearPlot

using CSV
using DataFrames
using LaTeXStrings
using LinearAlgebra
using Plots
using Plots.PlotMeasures: mm

function load_plot_data(csv_path::AbstractString; t0::Real=0.1)
    isfinite(t0) && t0 >= 0 || throw(ArgumentError("t0 must be finite and nonnegative"))
    source = CSV.read(csv_path, DataFrame)
    all(name -> name in propertynames(source), (:mu, :current, :temp_eff)) ||
        throw(ArgumentError("Input CSV must contain mu, current, and temp_eff columns"))
    all(value -> value isa Real && isfinite(value), source.mu) ||
        throw(ArgumentError("mu values must be finite numbers"))
    table = source[(0.0 .<= source.mu) .& (source.mu .<= 0.6), :]
    isempty(table.mu) && throw(ArgumentError("No data in the inclusive mu range [0.0, 0.6]"))
    for column in (:current, :temp_eff)
        all(value -> value isa Real && isfinite(value), table[!, column]) ||
            throw(ArgumentError("Selected $column values must be finite numbers"))
    end
    sort!(table, :mu)
    table.current_abs = abs.(table.current)
    # Recompute the increase, rather than trusting a possibly different
    # baseline stored in an existing delta_temp_eff column.
    table.delta_temp_eff = table.temp_eff .- t0
    table.t0 = fill(Float64(t0), nrow(table))
    return table
end

function fit_quadratic(table::DataFrame)
    x = table.current_abs
    y = table.delta_temp_eff
    design = hcat(x, x.^2)
    rank(design) == 2 || throw(ArgumentError(
        "Quadratic fit needs at least two distinct nonzero current magnitudes",
    ))
    coefficients = design \ y
    residuals = y - design * coefficients
    return (a=coefficients[1], b=coefficients[2],
            rmse=sqrt(sum(abs2, residuals) / length(y)))
end

function build_linear_plot(table::DataFrame; fit=fit_quadratic(table))
    figure = plot(
        table.current_abs, table.delta_temp_eff;
        size=(600, 400), color=1, linewidth=1, marker=:circle,
        xscale=:identity, yscale=:identity,
        xlabel=L"|J|", ylabel=L"\Delta T_{\mathrm{eff}}",
        title="(c)",
        label="Data", legend=:topleft, legendfontsize=12, grid=true,
        tickfontsize=14, guidefontsize=18,
        left_margin=4mm, right_margin=8mm,
        bottom_margin=5mm, top_margin=5mm,
    )
    currents = range(0.0, maximum(table.current_abs); length=300)
    plot!(figure, currents, fit.a .* currents .+ fit.b .* currents.^2;
          color=2, linewidth=1.5, linestyle=:dash,
          label=L"a|J|+b|J|^2")
    return figure
end

function refuse_existing(path::AbstractString)
    (ispath(path) || islink(path)) && throw(ArgumentError(
        "Refusing to overwrite existing output: $path. Choose another --outdir.",
    ))
    return nothing
end

function main(args)
    if args == ["--help"]
        println("Usage: julia --project=ASHoppingGKSL $(@__FILE__) [--csv PATH] [--outdir DIR] [--t0 VALUE]")
        println("Default input: out/nr_mu_sweep6_beta10/reduced_gibbs/current_temperature_vs_mu.csv")
        println("Default output: out/nr_mu_sweep6_beta10/linear_small_mu_0_0p6_fit (PDF, PNG, CSV)")
        println("Inclusive mu range: [0.0, 0.6]; both axes linear; delta T_eff = T_eff - T0 (default 0.1).")
        println("Overlays an unweighted fit delta T_eff = a*|J| + b*|J|^2, without a constant term.")
        println("Includes nonpositive temperature increases; refuses to overwrite existing output files.")
        return nothing
    end
    iseven(length(args)) || throw(ArgumentError("Every option needs a value"))
    opts = Dict{String, String}()
    for i in 1:2:length(args)
        args[i] in ("--csv", "--outdir", "--t0") ||
            throw(ArgumentError("Unknown option: $(args[i])"))
        startswith(args[i + 1], "--") && throw(ArgumentError("Missing value for $(args[i])"))
        opts[args[i]] = args[i + 1]
    end
    data_root = abspath(joinpath(@__DIR__, "..", "..", "out", "nr_mu_sweep6_beta10"))
    csv_path = abspath(get(opts, "--csv", joinpath(data_root, "reduced_gibbs", "current_temperature_vs_mu.csv")))
    outdir = abspath(get(opts, "--outdir", joinpath(data_root, "linear_small_mu_0_0p6_fit")))
    stem = "J_vs_delta_Teff_linear_mu_0_0p6"
    outputs = [joinpath(outdir, "$stem.$extension") for extension in ("csv", "pdf", "png")]
    # Preflight all three paths before creating or saving anything.
    foreach(refuse_existing, outputs)
    table = load_plot_data(csv_path; t0=parse(Float64, get(opts, "--t0", "0.1")))
    fit = fit_quadratic(table)
    figure = build_linear_plot(table; fit)
    mkpath(outdir)
    refuse_existing(outputs[1])
    CSV.write(outputs[1], table)
    for path in outputs[2:3]
        refuse_existing(path)
        savefig(figure, path)
    end
    println("Selected mu values: ", join(table.mu, ", "))
    println("Temperature definition: precomputed reduced-full-chain-Gibbs fit; T0 = ", first(table.t0))
    println("Quadratic fit (no constant term): a = ", fit.a, ", b = ", fit.b,
            ", RMSE = ", fit.rmse)
    foreach(path -> println("Saved: ", path), outputs)
    return table
end

end # module SmallMuLinearPlot

if abspath(PROGRAM_FILE) == @__FILE__
    SmallMuLinearPlot.main(ARGS)
end
