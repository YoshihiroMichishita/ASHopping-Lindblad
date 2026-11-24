"""
Plot current and reduced-full-system-Gibbs effective temperature over mu_* data.

Example (run from the repository root):
    julia --project=ASHoppingGKSL ASHoppingGKSL/script/plot_current_temperature_vs_mu_reduced_gibbs.jl \
        --indir out/nr_mu_sweep6_beta10 --mu-ref 0.0

Outputs a CSV and the same two figures as plot_current_temperature_vs_mu.jl,
using KL-fit temperatures. Defaults: output directory INDIR/reduced_gibbs,
T0=0.1, fixed reference chemical potential 0, beta range [-100, 100].
Hamiltonian parameters t and omega are read from JLD2 metadata; --t and
--omega override them (or supply them when metadata is absent). The sweep
parameter mu is kept separate from the fixed Gibbs-reference mu_ref.
"""
module ReducedGibbsMuSweep

# Reuse the original plotting style in its own namespace. Its entry point
# is guarded, so including it does not run the original calculation.
module OriginalMuPlot
include(joinpath(@__DIR__, "plot_current_temperature_vs_mu.jl"))
end
using .OriginalMuPlot: ASHoppingGKSL1P, parse_args, get_opt, mu_from_path,
                      build_current_temperature_plot, build_current_delta_temperature_plot
using CSV
using DataFrames
using JLD2
using Plots
import ASHoppingGKSL as gksl

function load_reduced_gibbs_input(path::AbstractString)
    return jldopen(path, "r") do file
        key = haskey(file, "matrix") ? "matrix" : haskey(file, "C") ? "C" :
              error("JLD2 file must contain matrix or C: $path")
        C = Matrix{ComplexF64}(file[key])
        metadata = haskey(file, "metadata") ? file["metadata"] : Dict{String, Any}()
        return C, metadata
    end
end

function reference_system_parameter(metadata, name, override, path)
    override !== nothing && return Float64(override)
    haskey(metadata, name) || throw(ArgumentError(
        "Missing Hamiltonian parameter $name in $path; specify --$name explicitly",
    ))
    return Float64(metadata[name])
end

function collect_reduced_gibbs_observables_over_mu(rootdir::AbstractString;
                                                 filename::AbstractString="nr_correlator.jld2",
                                                 t::Union{Nothing,Real}=nothing,
                                                 omega::Union{Nothing,Real}=nothing,
                                                 mu_ref::Real=0.0, t0::Real=0.1,
                                                 beta_bounds::Tuple{<:Real,<:Real}=(-100.0, 100.0),
                                                 grid_points::Integer=201,
                                                 beta_tol::Real=1e-9, maxiter::Integer=200,
                                                 eps::Real=1e-12)
    isfinite(t0) && t0 >= 0 || throw(ArgumentError("t0 must be finite and nonnegative"))
    mu_dirs = filter(path -> isdir(path) && startswith(basename(path), "mu_"),
                     readdir(rootdir; join=true))
    isempty(mu_dirs) && error("No mu_* directories found in $rootdir")
    sort!(mu_dirs; by=mu_from_path)
    records = NamedTuple[]
    for dir in mu_dirs
        path = joinpath(dir, filename)
        C, metadata = load_reduced_gibbs_input(path)
        n = size(C, 1)
        if haskey(metadata, "nsites") && metadata["nsites"] != n
            throw(DimensionMismatch("nsites metadata disagrees with correlator size in $path"))
        end
        hopping = reference_system_parameter(metadata, "t", t, path)
        potential = reference_system_parameter(metadata, "omega", omega, path)
        sys = gksl.SystemParams(n, hopping, potential)
        fit = ASHoppingGKSL1P.local_temperature_from_reduced_gibbs(
            C, sys; mu_ref, beta_bounds, grid_points, beta_tol, maxiter, eps,
        )
        fit.converged || error("Temperature fit did not converge for $path; increase --grid-points or --maxiter")
        current = ASHoppingGKSL1P.local_current_from_Cmid(fit.Cmid, hopping)
        delta_temperature = fit.temp_eff - t0
        push!(records, (
            mu=mu_from_path(dir),
            current=current,
            current_abs=abs(current),
            beta_eff=fit.beta_eff,
            temp_eff=fit.temp_eff,
            delta_temp_eff=delta_temperature,
            mu_ref=fit.mu_ref,
            t0=Float64(t0),
            relative_entropy=fit.relative_entropy,
            fit_residual=fit.fit_residual,
            at_boundary=fit.at_boundary,
            converged=fit.converged,
            iterations=fit.iterations,
            reference_regularization=fit.reference_regularization,
            semilog_eligible=isfinite(delta_temperature) && delta_temperature > 0,
            energy_loc=-2 * hopping * real(fit.Cmid[1, 2]),
            number_loc=real(fit.Cmid[1, 1] + fit.Cmid[2, 2]),
            nsites=n,
            t=hopping,
            omega=potential,
            beta_min=fit.beta_bounds[1],
            beta_max=fit.beta_bounds[2],
        ))
    end
    return DataFrame(records)
end

function build_reduced_gibbs_semilog_plot(table::DataFrame)
    plot_table = copy(table)
    eligible = isfinite.(table.delta_temp_eff) .& (table.delta_temp_eff .> 0)
    # NaN gaps avoid joining positive values across omitted nonpositive points.
    plot_table.delta_temp_eff = [eligible[i] ? table.delta_temp_eff[i] : NaN
                                for i in 1:nrow(table)]
    figure = build_current_delta_temperature_plot(plot_table)
    if !any(eligible)
        plot!(figure; ylims=(1e-4, 1.0), title="(d)")
        annotate!(figure, sum(extrema(table.current_abs)) / 2, 0.01,
                  text("No positive finite temperature increase", 10))
    end
    return figure
end

function main(args)
    if args == ["--help"]
        println("Usage: julia --project=ASHoppingGKSL $(@__FILE__) --indir DIR [options]")
        println("--outdir DIR       default: INDIR/reduced_gibbs")
        println("--filename NAME    default: nr_correlator.jld2")
        println("--mu-ref VALUE     fixed reference chemical potential (default 0.0)")
        println("--t0 VALUE         baseline for delta T_eff (default 0.1)")
        println("--t VALUE --omega VALUE   optional Hamiltonian metadata overrides")
        println("--beta-min VALUE --beta-max VALUE   inverse-temperature bounds (default -100, 100)")
        println("--grid-points N --beta-tol VALUE --maxiter N --eps VALUE   optional fit controls")
        return nothing
    end
    opts = parse_args(args)
    allowed = Set(["indir", "outdir", "filename", "t", "omega", "mu-ref", "t0",
                   "beta-min", "beta-max", "grid-points", "beta-tol", "maxiter", "eps"])
    for key in keys(opts)
        key in allowed || throw(ArgumentError("Unknown option --$key"))
    end
    input_dir = abspath(get_opt(opts, "indir", joinpath("out", "nr_mu_sweep6")))
    output_dir = abspath(get_opt(opts, "outdir", joinpath(input_dir, "reduced_gibbs")))
    table = collect_reduced_gibbs_observables_over_mu(
        input_dir;
        filename=get_opt(opts, "filename", "nr_correlator.jld2"),
        t=haskey(opts, "t") ? parse(Float64, opts["t"]) : nothing,
        omega=haskey(opts, "omega") ? parse(Float64, opts["omega"]) : nothing,
        mu_ref=parse(Float64, get_opt(opts, "mu-ref", 0.0)),
        t0=parse(Float64, get_opt(opts, "t0", 0.1)),
        beta_bounds=(parse(Float64, get_opt(opts, "beta-min", -100.0)),
                     parse(Float64, get_opt(opts, "beta-max", 100.0))),
        grid_points=parse(Int, get_opt(opts, "grid-points", 201)),
        beta_tol=parse(Float64, get_opt(opts, "beta-tol", 1e-9)),
        maxiter=parse(Int, get_opt(opts, "maxiter", 200)),
        eps=parse(Float64, get_opt(opts, "eps", 1e-12)),
    )
    mkpath(output_dir)
    csv_path = joinpath(output_dir, "current_temperature_vs_mu.csv")
    CSV.write(csv_path, table)
    figure = build_current_temperature_plot(table)
    semilog = build_reduced_gibbs_semilog_plot(table)
    for extension in ("png", "pdf")
        savefig(figure, joinpath(output_dir, "current_temperature_vs_mu.$extension"))
        savefig(semilog, joinpath(output_dir, "J_vs_Teff_semilog.$extension"))
    end
    println("mu range = ", first(table.mu), " to ", last(table.mu))
    println("Fixed Gibbs reference chemical potential = ", first(table.mu_ref))
    println("Saved data to: ", csv_path)
    println("Saved current-temperature and semilog plots to: ", output_dir)
    if any(table.at_boundary)
        println("Fit reached beta bounds at mu = ", table.mu[table.at_boundary], "; inspect beta_min/beta_max in the CSV.")
    end
    if !all(table.semilog_eligible)
        println("Semilog omits nonpositive/nonfinite delta T_eff at mu = ",
                table.mu[.!table.semilog_eligible], "; all values are retained in the CSV.")
    end
    return table
end

end # module ReducedGibbsMuSweep

if abspath(PROGRAM_FILE) == @__FILE__
    ReducedGibbsMuSweep.main(ARGS)
end
