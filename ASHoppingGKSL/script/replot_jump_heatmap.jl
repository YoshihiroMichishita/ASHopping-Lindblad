using JLD2
using CairoMakie
using LaTeXStrings

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

function parse_symbol_list(value::AbstractString)
    isempty(value) && return Symbol[]
    return Symbol.(split(value, ","))
end

function parse_bool(value::AbstractString)
    return lowercase(value) in ("true", "1", "yes", "y")
end

function matrix_component(mat::AbstractMatrix{<:Complex}, component::Symbol)
    if component == :real
        return real.(mat)
    elseif component == :imag
        return imag.(mat)
    elseif component == :abs
        return abs.(mat)
    end
    error("component must be :real, :imag, or :abs")
end

function component_plot_settings(values::AbstractMatrix{<:Real}, component::Symbol; unit_range::Bool = true)
    if component == :abs
        if unit_range
            return :viridis, (0.0, 1.0)
        end
        vmax = maximum(values)
        vmax = vmax == 0 ? 1.0 : vmax
        return :viridis, (0.0, vmax)
    end

    vmax = maximum(abs.(values))
    vmax = vmax == 0 ? 1.0 : vmax
    return :balance, (-vmax, vmax)
end

function title_from_basename(name::AbstractString)
    stem = replace(name, ".jld2" => "")
    parts = split(stem, "_")
    length(parts) >= 4 || return replace(stem, "_" => " ")

    bath_part = parts[2]
    body = join(parts[3:end], " ")
    bath_label = startswith(bath_part, "bath") ? "Bath " * bath_part[5:end] : bath_part
    return string(bath_label, " ", titlecase(body))
end

function render_heatmap(path::AbstractString, mat::AbstractMatrix{<:Complex};
                        component::Symbol = :abs,
                        title::AbstractString = "",
                        unit_range::Bool = true)
    values = matrix_component(mat, component)
    cmap, crange = component_plot_settings(values, component; unit_range)

    fig = Figure(size=(1020, 760))
    ax = Axis(
        fig[1, 1],
        title=title,
        xlabel=L"j",
        ylabel=L"i",
        xticks=collect(1:size(values, 2)),
        yticks=collect(1:size(values, 1)),
        titlesize=60,
        xlabelsize=36,
        ylabelsize=36,
        xticklabelsize=32,
        yticklabelsize=32,
    )
    hm = heatmap!(ax, 1:size(values, 2), 1:size(values, 1), values;
                  colormap=cmap, colorrange=crange)
    cb = Colorbar(fig[1, 2], hm, width=28)
    colsize!(fig.layout, 2, 90)
    cb.labelsize = 36
    cb.ticklabelsize = 32
    save(path, fig)
    return path
end

function load_matrix(path::AbstractString)
    data = load(path)
    haskey(data, "matrix") || error("missing `matrix` dataset in $path")
    metadata = get(data, "metadata", Dict{String, Any}())
    return Matrix{ComplexF64}(data["matrix"]), metadata
end

function selected_jld2_files(indir::AbstractString, suffix::AbstractString)
    files = sort(filter(name -> endswith(name, "_$(suffix).jld2"), readdir(indir)))
    isempty(files) && error("No files matching *_$(suffix).jld2 found in $indir")
    return [joinpath(indir, name) for name in files]
end

function main(args)
    opts = parse_args(args)

    indir = abspath(get_opt(opts, "indir", joinpath("out", "nr_sweep6", "Gamma_0.2")))
    outdir = abspath(get_opt(opts, "outdir", joinpath(indir, "replot")))
    suffix = get_opt(opts, "suffix", "weighted")
    component = Symbol(get_opt(opts, "component", "abs"))
    formats = parse_symbol_list(get_opt(opts, "formats", "png,pdf"))
    unit_range = parse_bool(get_opt(opts, "unit-range", "true"))

    mkpath(outdir)

    written = String[]
    counts = 1
    titles = [L"\gamma^{(l)}_{g;ij}", L"\gamma^{(l)}_{l;ij}", L"\gamma^{(r)}_{g;ij}", L"\gamma^{(r)}_{l;ij}"]
    for path in selected_jld2_files(indir, suffix)
        mat, _ = load_matrix(path)
        stem = replace(basename(path), ".jld2" => "")
        title = titles[counts]
        #title = title_from_basename(basename(path))
        for fmt in formats
            outpath = joinpath(outdir, "$(stem)_$(String(component)).$(String(fmt))")
            render_heatmap(outpath, mat; component, title, unit_range)
            push!(written, outpath)
        end
        counts += 1
    end

    println("Saved replotted heatmaps to: ", outdir)
    for path in written
        println("  ", path)
    end
end

main(ARGS)
