import Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

using IsThereSkinEffect
using Plots

params = ModelParams(32, 1.0, 0.4)
comparison = compare_boundaries(params)

println("PBC density = ", comparison.pbc.density)
println("PBC density contrast = ", maximum(comparison.pbc.density) - minimum(comparison.pbc.density))
println("PBC bond currents = ", comparison.pbc.currents)
println("PBC current average = ", current_expectation(comparison.pbc))
println("PBC dissipative site net = ", dissipative_site_net(comparison.pbc))
println("PBC continuity residual = ", continuity_residuals(comparison.pbc))
println("PBC residual = ", steady_state_residual(comparison.pbc))
println()
println("OBC density = ", comparison.obc.density)
println("OBC density contrast = ", maximum(comparison.obc.density) - minimum(comparison.obc.density))
println("OBC bond currents = ", comparison.obc.currents)
println("OBC current average = ", current_expectation(comparison.obc))
println("OBC dissipative site net = ", dissipative_site_net(comparison.obc))
println("OBC continuity residual = ", continuity_residuals(comparison.obc))
println("OBC residual = ", steady_state_residual(comparison.obc))

sites = collect(1:params.L)
density_fig = plot(
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
    density_fig,
    sites,
    comparison.obc.density;
    label = "OBC",
    marker = :square,
    lw = 2,
    ylims = (0.0, 1.0),
)

density_path = joinpath(@__DIR__, "pbc_obc_density.png")
savefig(density_fig, density_path)
println("saved density plot to ", density_path)

pbc_bonds = collect(1:length(comparison.pbc.currents))
obc_bonds = collect(1:length(comparison.obc.currents))
current_fig = plot(
    pbc_bonds,
    comparison.pbc.currents;
    label = "PBC",
    xlabel = "bond",
    ylabel = "coherent current",
    marker = :circle,
    lw = 2,
    ylims = (0.0, 1.0),
)
plot!(
    current_fig,
    obc_bonds,
    comparison.obc.currents;
    label = "OBC",
    marker = :square,
    lw = 2,
    ylims = (0.0, 1.0),
)


current_path = joinpath(@__DIR__, "pbc_obc_current.png")
savefig(current_fig, current_path)
println("saved current plot to ", current_path)
