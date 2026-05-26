push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..")))
import ASHoppingGKSL as gksl
include(joinpath(@__DIR__, "..", "src2", "ASHoppingGKSL1P.jl"))
using .ASHoppingGKSL1P

using LaTeXStrings
using Plots

function main(ARGS)
    dir = ARGS[1]
    Gamma, beta, temp, mu, current = ASHoppingGKSL1P.collect_local_observables_over_gamma(dir; filename="nr_correlator.jld2", t=1.0, format=:jld2)
    c_max = maximum(abs.(current))
    temp_max = maximum(temp)
    g = Gamma
    p1 = plot(g, abs.(current), marker=:circle, lc=1, xlabel=L"\gamma", ylabel=L"J", label=L"J", tickfontsize = 14,guidefontsize = 16, legendfontsize = 14, titlefontsize = 18)     # タイトル)
    p1 = plot!(twinx(),g, temp, marker=:square, lc=2, markercolor=2, legend=:bottomright, ylabel=L"T_\mathrm{eff}", label=L"T_\mathrm{eff}", title="(c)", tickfontsize = 14,guidefontsize = 18, legendfontsize = 18, titlefontsize = 20)
    savefig(p1, dir*"/JandTeff_vs_gamma.pdf")

    sys = gksl.SystemParams(6, 1.0, 0.0)
    obs = ASHoppingGKSL1P.isolated_local_observables(sys, 10.0)
    T0 = 1/obs.beta_eff
    c_ratio = abs.(current)
    temp_ratio = temp
    p2 = plot(c_ratio, temp_ratio.-T0, marker=:circle, yscale=:log10, xlabel=L"J", ylabel=L"\Delta T_\mathrm{eff}", title="(d)", tickfontsize = 14,guidefontsize = 18, legendfontsize = 18, titlefontsize = 20, legend=false)
    savefig(p2, dir*"J_vs_Teff_semilog.pdf")

    C = ASHoppingGKSL1P.load_correlator(dir*"/Gamma_0.2/nr_correlator.jld2"; format=:jld2)
    js = [imag(C[i, i+1]-C[i+1, i]) for i in 1:5]
    p1 = plot(js, marker=:circle, xlabel=L"i", ylabel=L"<J_i>", title="(b)", lw=2.0, ylim=(-1.0, 0.0), tickfontsize = 14,guidefontsize = 18, legendfontsize = 18, titlefontsize = 20, legend=false)
    savefig(p1, dir*"/local_current.pdf")

    ns = [real(C[i, i]) for i in 1:6]
    p = plot(ns, marker=:circle, xlabel=L"i", ylabel=L"<n_i>", title="(a)", tickfontsize = 14,guidefontsize = 18, legendfontsize = 18, titlefontsize = 20, legend=false)
    savefig(p, dir*"/occupation_profile.pdf")
end

main(ARGS)