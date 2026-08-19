push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..")))

include(joinpath(@__DIR__, "..", "src2", "ASHoppingGKSL1P.jl"))
using .ASHoppingGKSL1P

using CairoMakie
using CSV
using DataFrames

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

function gamma_grid(gamma_min::Real, gamma_max::Real, gamma_step::Real)
    gamma_step > 0 || error("gamma-step must be positive")
    gamma_max >= gamma_min || error("gamma-max must be at least gamma-min")

    nsteps_float = (gamma_max - gamma_min) / gamma_step
    nsteps = round(Int, nsteps_float)
    isapprox(nsteps_float, nsteps; atol=1e-10, rtol=1e-10) ||
        error("gamma range must contain an integer number of gamma-step intervals")
    return gamma_min .+ gamma_step .* collect(0:nsteps)
end

function select_gamma_indices(available::AbstractVector{<:Real}, requested::AbstractVector{<:Real})
    tolerance = 100 * eps(Float64)
    return [begin
        index = findfirst(gamma -> isapprox(gamma, target; atol=tolerance, rtol=1e-10), available)
        isnothing(index) && error("No correlator found for Gamma=$target")
        index
    end for target in requested]
end

function comparison_table(sweep, indices)
    beta_existing = sweep.beta_eff[indices]
    beta_energy = sweep.beta_eff_energy[indices]
    temp_existing = sweep.temp_eff[indices]
    temp_energy = sweep.temp_eff_energy[indices]
    mu_existing = sweep.mu_eff[indices]
    mu_energy = sweep.mu_eff_energy[indices]

    return DataFrame(
        Gamma=sweep.Gamma[indices],
        beta_eff_existing=beta_existing,
        beta_eff_energy=beta_energy,
        temp_eff_existing=temp_existing,
        temp_eff_energy=temp_energy,
        mu_eff_existing=mu_existing,
        mu_eff_energy=mu_energy,
        delta_beta_eff=beta_energy .- beta_existing,
        delta_temp_eff=temp_energy .- temp_existing,
        delta_mu_eff=mu_energy .- mu_existing,
        energy_loc=sweep.energy_loc[indices],
        number_loc=sweep.number_loc[indices],
    )
end

function add_method_comparison!(ax, gamma, existing, energy)
    lines!(ax, gamma, existing; color=:dodgerblue, linewidth=2.5,
           label="existing correlator method")
    scatter!(ax, gamma, existing; color=:dodgerblue, marker=:circle, markersize=10)
    lines!(ax, gamma, energy; color=:darkorange, linewidth=2.5, linestyle=:dash,
           label="energy/number method")
    scatter!(ax, gamma, energy; color=:darkorange, marker=:rect, markersize=10)
    return ax
end

function render_comparison(table::DataFrame, output_path::AbstractString)
    figure = Figure(size=(1200, 900), fontsize=18)
    gamma = table.Gamma

    beta_axis = Axis(figure[1, 1], xlabel="Gamma", ylabel="beta_eff", title="Inverse temperature")
    temp_axis = Axis(figure[1, 2], xlabel="Gamma", ylabel="T_eff", title="Effective temperature")
    mu_axis = Axis(figure[2, 1], xlabel="Gamma", ylabel="mu_eff", title="Effective chemical potential")
    diff_axis = Axis(
        figure[2, 2],
        xlabel="Gamma",
        ylabel="absolute difference (zero shown at eps)",
        title="Difference between methods",
        yscale=log10,
    )

    add_method_comparison!(beta_axis, gamma, table.beta_eff_existing, table.beta_eff_energy)
    add_method_comparison!(temp_axis, gamma, table.temp_eff_existing, table.temp_eff_energy)
    add_method_comparison!(mu_axis, gamma, table.mu_eff_existing, table.mu_eff_energy)

    plot_floor = eps(Float64)
    lines!(diff_axis, gamma, max.(abs.(table.delta_beta_eff), plot_floor);
           color=:dodgerblue, linewidth=2.5, label="|delta beta_eff|")
    scatter!(diff_axis, gamma, max.(abs.(table.delta_beta_eff), plot_floor);
             color=:dodgerblue, marker=:circle, markersize=9)
    lines!(diff_axis, gamma, max.(abs.(table.delta_temp_eff), plot_floor);
           color=:seagreen, linewidth=2.5, label="|delta T_eff|")
    scatter!(diff_axis, gamma, max.(abs.(table.delta_temp_eff), plot_floor);
             color=:seagreen, marker=:diamond, markersize=10)
    lines!(diff_axis, gamma, max.(abs.(table.delta_mu_eff), plot_floor);
           color=:darkorange, linewidth=2.5, label="|delta mu_eff|")
    scatter!(diff_axis, gamma, max.(abs.(table.delta_mu_eff), plot_floor);
             color=:darkorange, marker=:rect, markersize=9)

    axislegend(beta_axis; position=:rt, labelsize=14)
    axislegend(diff_axis; position=:rt, labelsize=14)
    save(output_path, figure)
    return output_path
end

function main(args)
    opts = parse_args(args)
    input_dir = abspath(get_opt(opts, "indir", joinpath("out", "nr_sweep6")))
    output_dir = abspath(get_opt(opts, "outdir", joinpath(input_dir, "temperature_comparison")))
    filename = get_opt(opts, "filename", "nr_correlator.jld2")
    t = parse(Float64, get_opt(opts, "t", 1.0))
    gamma_min = parse(Float64, get_opt(opts, "gamma-min", 0.05))
    gamma_max = parse(Float64, get_opt(opts, "gamma-max", 0.40))
    gamma_step = parse(Float64, get_opt(opts, "gamma-step", 0.05))

    requested_gamma = gamma_grid(gamma_min, gamma_max, gamma_step)
    sweep = ASHoppingGKSL1P.collect_local_observables_over_gamma(
        input_dir; filename, t, format=:jld2,
    )
    indices = select_gamma_indices(sweep.Gamma, requested_gamma)
    table = comparison_table(sweep, indices)

    mkpath(output_dir)
    csv_path = joinpath(output_dir, "local_temperature_methods_gamma.csv")
    png_path = joinpath(output_dir, "local_temperature_methods_gamma.png")
    pdf_path = joinpath(output_dir, "local_temperature_methods_gamma.pdf")
    CSV.write(csv_path, table)
    render_comparison(table, png_path)
    render_comparison(table, pdf_path)

    println("Gamma range = ", first(table.Gamma), ":", gamma_step, ":", last(table.Gamma))
    println("maximum |delta beta_eff| = ", maximum(abs.(table.delta_beta_eff)))
    println("maximum |delta T_eff| = ", maximum(abs.(table.delta_temp_eff)))
    println("maximum |delta mu_eff| = ", maximum(abs.(table.delta_mu_eff)))
    println("Saved comparison data to: ", csv_path)
    println("Saved comparison plots to: ", png_path, " and ", pdf_path)
end

main(ARGS)
