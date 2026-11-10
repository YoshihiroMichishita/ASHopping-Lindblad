"""
Plot current and reduced-full-system-Gibbs KL-fit temperature over Gamma_* data.

Run from the repository root:
    julia --project=ASHoppingGKSL ASHoppingGKSL/script/plot_current_temperature_vs_gamma_reduced_gibbs.jl \
        --indir out/nr_sweep6 --mu-ref 0.0

Outputs a CSV, JandTeff_vs_gamma.{png,pdf}, and J_vs_Teff_semilog.{png,pdf}
in INDIR/reduced_gibbs by default. The gamma figure uses T_eff, whereas the
semilog figure uses delta T_eff = T_eff - T0 with default T0=0.1.
Both figures and the CSV include only 0.05 <= gamma <= 0.4.
The sweep coordinate comes from Gamma_* directory names; metadata mu is
the driving parameter, separate from the fixed Gibbs reference mu_ref.
Hamiltonian parameters come from metadata unless --t/--omega are given.
"""
module ReducedGibbsGammaSweep

# Including the mu-sweep module reuses the fit helpers and plotting style
# without running its guarded entry point or changing the existing script.
include(joinpath(@__DIR__, "plot_current_temperature_vs_mu_reduced_gibbs.jl"))
using .ReducedGibbsMuSweep: ASHoppingGKSL1P, parse_args, get_opt,
                           load_reduced_gibbs_input, reference_system_parameter,
                           build_reduced_gibbs_semilog_plot
using CSV
using DataFrames
using Plots
using LaTeXStrings
import ASHoppingGKSL as gksl

function collect_reduced_gibbs_observables_over_gamma(rootdir::AbstractString;
                                                    filename::AbstractString="nr_correlator.jld2",
                                                    t::Union{Nothing,Real}=nothing,
                                                    omega::Union{Nothing,Real}=nothing,
                                                    mu_ref::Real=0.0, t0::Real=0.1,
                                                    beta_bounds::Tuple{<:Real,<:Real}=(-100.0, 100.0),
                                                    grid_points::Integer=201,
                                                    beta_tol::Real=1e-9, maxiter::Integer=200,
                                                    eps::Real=1e-12)
    isfinite(t0) && t0 >= 0 || throw(ArgumentError("t0 must be finite and nonnegative"))
    gamma_dirs = filter(path -> isdir(path) && startswith(basename(path), "Gamma_"),
                        readdir(rootdir; join=true))
    filter!(path -> 0.05 <= ASHoppingGKSL1P.gamma_from_path(path) <= 0.4, gamma_dirs)
    isempty(gamma_dirs) && error("No Gamma_* directories with 0.05 <= gamma <= 0.4 found in $rootdir")
    sort!(gamma_dirs; by=ASHoppingGKSL1P.gamma_from_path)
    records = NamedTuple[]
    for dir in gamma_dirs
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
            Gamma=ASHoppingGKSL1P.gamma_from_path(dir),
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
            mu=get(metadata, "mu", missing),
            beta_bath=get(metadata, "beta", missing),
            bath_Gamma=get(metadata, "Gamma", missing),
            tau_c=get(metadata, "tau_c", missing),
        ))
    end
    return DataFrame(records)
end

function build_current_temperature_gamma_plot(table::DataFrame)
    plot_table = copy(table)
    # The shared plot helper expects a mu column as its x coordinate.
    # Substitute gamma only in a copy, preserving the physical metadata mu.
    plot_table.mu = table.Gamma
    figure = ReducedGibbsMuSweep.build_current_temperature_plot(plot_table)
    plot!(figure; subplot=1, xlabel=L"\gamma", title="(c)")
    return figure
end

function main(args)
    if args == ["--help"]
        println("Usage: julia --project=ASHoppingGKSL $(@__FILE__) [options]")
        println("--indir DIR        default: out/nr_sweep6 (Gamma_* directories, 0.05 <= gamma <= 0.4)")
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
    input_dir = abspath(get_opt(opts, "indir", joinpath("out", "nr_sweep6")))
    output_dir = abspath(get_opt(opts, "outdir", joinpath(input_dir, "reduced_gibbs")))
    table = collect_reduced_gibbs_observables_over_gamma(
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
    csv_path = joinpath(output_dir, "current_temperature_vs_gamma.csv")
    CSV.write(csv_path, table)
    figure = build_current_temperature_gamma_plot(table)
    semilog = build_reduced_gibbs_semilog_plot(table)
    for extension in ("png", "pdf")
        savefig(figure, joinpath(output_dir, "JandTeff_vs_gamma.$extension"))
        savefig(semilog, joinpath(output_dir, "J_vs_Teff_semilog.$extension"))
    end
    println("gamma range = ", first(table.Gamma), " to ", last(table.Gamma))
    println("Fixed Gibbs reference chemical potential = ", first(table.mu_ref))
    println("Saved data to: ", csv_path)
    println("Saved current-temperature and semilog plots to: ", output_dir)
    if any(table.at_boundary)
        println("Fit reached beta bounds at gamma = ", table.Gamma[table.at_boundary],
                "; inspect beta_min/beta_max in the CSV.")
    end
    if !all(table.semilog_eligible)
        println("Semilog omits nonpositive/nonfinite delta T_eff at gamma = ",
                table.Gamma[.!table.semilog_eligible], "; all values are retained in the CSV.")
    end
    return table
end

end # module ReducedGibbsGammaSweep

if abspath(PROGRAM_FILE) == @__FILE__
    ReducedGibbsGammaSweep.main(ARGS)
end
