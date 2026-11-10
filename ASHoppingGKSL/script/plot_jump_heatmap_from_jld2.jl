"""
Replot saved NR jump coefficients with a fixed colorbar range (0, 0.3).

Run from the repository root:
    julia --project=ASHoppingGKSL ASHoppingGKSL/script/plot_jump_heatmap_from_jld2.jl

Reads nr_bath{1,2}_{gain,loss}_coeff.jld2 without recalculating the model.
Plots the absolute values without normalization, in the existing heatmap
style. Default input: out/nr_mu_sweep6_beta10/mu_0.0. Default output:
INDIR/heatmaps_clim_0_0p3, with four PDFs and four PNGs. Existing output
files are never overwritten: a collision aborts before any plot is saved.
Use --indir DIR and/or --outdir DIR to select other directories.
"""
module JumpHeatmapFromJLD2

using JLD2
using CairoMakie
using LaTeXStrings

function load_coefficient(path::AbstractString)
    isfile(path) || throw(ArgumentError("Missing coefficient file: $path"))
    return jldopen(path, "r") do file
        haskey(file, "matrix") || throw(ArgumentError("Missing matrix key in $path"))
        Matrix{ComplexF64}(file["matrix"])
    end
end

function build_coefficient_heatmap(mat::AbstractMatrix; title::AbstractString="")
    size(mat, 1) == size(mat, 2) && size(mat, 1) > 0 ||
        throw(ArgumentError("Coefficient matrix must be nonempty and square"))
    all(isfinite, mat) || throw(ArgumentError("Coefficient matrix must contain finite values"))
    values = abs.(mat)
    fig = Figure(size=(1020, 760))
    ax = Axis(
        fig[1, 1];
        title, xlabel=L"j", ylabel=L"i",
        xticks=collect(1:size(values, 2)), yticks=collect(1:size(values, 1)),
        titlesize=60, xlabelsize=36, ylabelsize=36,
        xticklabelsize=32, yticklabelsize=32,
    )
    # Makie indexes heatmaps as (x, y), whereas matrix indices are (i, j).
    hm = heatmap!(ax, 1:size(values, 2), 1:size(values, 1), permutedims(values);
                  colormap=:viridis, colorrange=(0.0, 0.3))
    cb = Colorbar(fig[1, 2], hm; width=28,
                  ticks=([0.0, 0.15, 0.3], ["0.0", "0.15", "0.3"]))
    colsize!(fig.layout, 2, 90)
    cb.labelsize = 36
    cb.ticklabelsize = 32
    return (figure=fig, axis=ax, heatmap=hm, colorbar=cb)
end

function main(args)
    if args == ["--help"]
        println("Usage: julia --project=ASHoppingGKSL $(@__FILE__) [--indir DIR] [--outdir DIR]")
        println("Default input: out/nr_mu_sweep6_beta10/mu_0.0 (relative to the repository root)")
        println("Default output: INDIR/heatmaps_clim_0_0p3 (PDF and PNG)")
        println("Plots absolute coefficients with colorbar range (0, 0.3); refuses to overwrite existing outputs.")
        return String[]
    end
    opts = Dict{String, String}()
    iseven(length(args)) || throw(ArgumentError("Every option needs a value"))
    for i in 1:2:length(args)
        args[i] in ("--indir", "--outdir") || throw(ArgumentError("Unknown option: $(args[i])"))
        startswith(args[i + 1], "--") && throw(ArgumentError("Missing value for $(args[i])"))
        opts[args[i]] = args[i + 1]
    end
    default_input = joinpath(@__DIR__, "..", "..", "out", "nr_mu_sweep6_beta10", "mu_0.0")
    input_dir = abspath(get(opts, "--indir", default_input))
    output_dir = abspath(get(opts, "--outdir", joinpath(input_dir, "heatmaps_clim_0_0p3")))
    specs = [
        (stem="nr_bath1_gain_coeff", title=L"\gamma^{(l)}_{g;ij}"),
        (stem="nr_bath1_loss_coeff", title=L"\gamma^{(l)}_{l;ij}"),
        (stem="nr_bath2_gain_coeff", title=L"\gamma^{(r)}_{g;ij}"),
        (stem="nr_bath2_loss_coeff", title=L"\gamma^{(r)}_{l;ij}"),
    ]
    outputs = [joinpath(output_dir, "$(spec.stem)_abs.$extension")
               for spec in specs for extension in ("pdf", "png")]
    # Check every destination before writing anything, including partial
    # collisions. Source files and previously generated figures stay intact.
    for path in outputs
        (ispath(path) || islink(path)) && throw(ArgumentError(
            "Refusing to overwrite existing output: $path. Choose a different --outdir.",
        ))
    end
    matrices = [load_coefficient(joinpath(input_dir, "$(spec.stem).jld2")) for spec in specs]
    figures = [build_coefficient_heatmap(mat; title=spec.title).figure
               for (spec, mat) in zip(specs, matrices)]
    mkpath(output_dir)
    for (spec, figure) in zip(specs, figures), extension in ("pdf", "png")
        path = joinpath(output_dir, "$(spec.stem)_abs.$extension")
        (ispath(path) || islink(path)) && throw(ArgumentError("Refusing to overwrite existing output: $path"))
        save(path, figure)
        println("Saved: ", path)
    end
    return outputs
end

end # module JumpHeatmapFromJLD2

if abspath(PROGRAM_FILE) == @__FILE__
    JumpHeatmapFromJLD2.main(ARGS)
end
