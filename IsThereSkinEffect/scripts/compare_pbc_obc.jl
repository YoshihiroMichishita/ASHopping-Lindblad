import Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

using IsThereSkinEffect
using Plots

params = ModelParams(24, 1.0, 0.4, 0.5, 4.0; mu = 0.0)
comparison = compare_boundaries_site_drive(params)

println("PBC energies = ", comparison.pbc.energies)
println("PBC occupations = ", comparison.pbc.occupations)
println("PBC density = ", comparison.pbc.density)
println("PBC density contrast = ", maximum(comparison.pbc.density) - minimum(comparison.pbc.density))
println("PBC coherent current = ", sum(coherent_bond_currents(comparison.pbc, params)) / length(coherent_bond_currents(comparison.pbc, params)))
println("PBC dissipative flux = ", dissipative_flux_expectation(comparison.pbc, params))
println("PBC bond fluxes = ", dissipative_fluxes(comparison.pbc, params))
println("PBC residual = ", steady_state_residual(comparison.pbc, params))
println()
println("OBC energies = ", comparison.obc.energies)
println("OBC occupations = ", comparison.obc.occupations)
println("OBC density = ", comparison.obc.density)
println("OBC density contrast = ", maximum(comparison.obc.density) - minimum(comparison.obc.density))
println("OBC coherent current = ", sum(coherent_bond_currents(comparison.obc, params)) / length(coherent_bond_currents(comparison.obc, params)))
println("OBC dissipative flux = ", dissipative_flux_expectation(comparison.obc, params))
println("OBC bond fluxes = ", dissipative_fluxes(comparison.obc, params))
println("OBC residual = ", steady_state_residual(comparison.obc, params))

sites = collect(1:params.L)
fig = plot(
    sites,
    comparison.pbc.density;
    label = "PBC",
    xlabel = "site",
    ylabel = "particle density",
    marker = :circle,
    lw = 2,
    ylims = (0.0, 1.0),
)
plot!(
    fig,
    sites,
    comparison.obc.density;
    label = "OBC",
    marker = :square,
    lw = 2,
)

png_path = joinpath(@__DIR__, "pbc_obc_density.png")
savefig(fig, png_path)
println("saved density plot to ", png_path)
