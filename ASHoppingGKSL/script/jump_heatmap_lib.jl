module JumpHeatmapLib

using LinearAlgebra
using CSV
using DataFrames
using JLD2
using CairoMakie
using LaTeXStrings

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..")))
import ASHoppingGKSL as gksl

include(joinpath(@__DIR__, "..", "src2", "ASHoppingGKSL1P.jl"))
using .ASHoppingGKSL1P

export build_model,
       nr_jump_blocks_by_bath,
       redfield_jump_blocks_by_bath,
       attach_weighted_from_C,
       jump_blocks_for_method,
       correlator_for_method,
       save_correlator_matrix,
       save_bath_blocks,
       save_heatmaps,
       render_heatmap,
       matrix_component

function build_model(nsites::Int, t::Float64, omega::Float64, mu::Float64,
                     beta::Float64, Gamma::Float64, tau_c::Float64,
                     bathkind::Symbol)
    sys = gksl.SystemParams(nsites, t, omega)
    if bathkind == :bath
        b1 = gksl.make_bath(Gamma=Gamma, mu=-mu, beta=beta, tau_c=tau_c)
        b2 = gksl.make_bath(Gamma=Gamma, mu=mu, beta=beta, tau_c=tau_c)
    elseif bathkind == :bath2
        b1 = gksl.make_bath2(Gamma=Gamma, mu=-mu, beta=beta, tau_c=tau_c)
        b2 = gksl.make_bath2(Gamma=Gamma, mu=mu, beta=beta, tau_c=tau_c)
    else
        error("bathkind must be :bath or :bath2")
    end
    return sys, gksl.TwoBathModel(sys, b1, b2)
end

function nr_jump_blocks_by_bath(m::gksl.TwoBathModel)
    return ASHoppingGKSL1P.nr_jump_blocks_by_bath_1p(m)
end

function redfield_jump_blocks_by_bath(m::gksl.TwoBathModel)
    return ASHoppingGKSL1P.redfield_jump_blocks_by_bath_1p(m)
end

function jump_blocks_for_method(m::gksl.TwoBathModel, method::Symbol; with_weighted::Bool = true)
    if with_weighted
        return ASHoppingGKSL1P.jump_blocks_with_weighted_1p(m, method)
    end
    if method == :nr
        return nr_jump_blocks_by_bath(m)
    elseif method == :redfield
        return redfield_jump_blocks_by_bath(m)
    else
        error("method must be :nr or :redfield")
    end
end

function attach_weighted_from_C(blocks::Dict{Symbol, Matrix{ComplexF64}},
                                C::AbstractMatrix{<:Complex})
    return ASHoppingGKSL1P.attach_weighted_from_C(blocks, C)
end

function correlator_for_method(m::gksl.TwoBathModel, method::Symbol)
    return ASHoppingGKSL1P.one_particle_correlator_for_method(m, method)
end

function matrix_component(mat::AbstractMatrix{<:Complex}, component::Symbol)
    if component == :real
        return real.(mat)
    elseif component == :imag
        return imag.(mat)
    elseif component == :abs
        return abs.(mat)
    else
        error("component must be :real, :imag, or :abs")
    end
end

function matrix_to_dataframe(mat::AbstractMatrix, row_prefix::String)
    nrows, ncols = size(mat)
    df = DataFrame(row = ["$(row_prefix)_$(i)" for i in 1:nrows])
    for j in 1:ncols
        df[!, "site_$(j)"] = mat[:, j]
    end
    return df
end

function save_matrix_csvs(basepath::AbstractString, mat::AbstractMatrix{<:Complex})
    CSV.write(basepath * "_real.csv", matrix_to_dataframe(real.(mat), "site"))
    CSV.write(basepath * "_imag.csv", matrix_to_dataframe(imag.(mat), "site"))
    CSV.write(basepath * "_abs.csv", matrix_to_dataframe(abs.(mat), "site"))
end

function save_matrix_jld2(path::AbstractString, mat::AbstractMatrix{<:Complex};
                          metadata::Dict{String, Any} = Dict{String, Any}())
    jldopen(path, "w") do file
        file["matrix"] = Matrix{ComplexF64}(mat)
        file["metadata"] = metadata
    end
end

function save_bath_blocks(outdir::AbstractString, prefix::AbstractString,
                          blocks_by_bath::Dict{Int, Dict{Symbol, Matrix{ComplexF64}}};
                          matrix_formats::Vector{Symbol} = [:csv, :jld2],
                          metadata::Dict{String, Any} = Dict{String, Any}())
    mkpath(outdir)
    for bath in sort(collect(keys(blocks_by_bath)))
        for (name, mat) in sort(collect(blocks_by_bath[bath]); by=first)
            base = joinpath(outdir, "$(prefix)_bath$(bath)_$(String(name))")
            if :csv in matrix_formats
                save_matrix_csvs(base, mat)
            end
            if :jld2 in matrix_formats
                save_matrix_jld2(base * ".jld2", mat; metadata=merge(metadata, Dict(
                    "bath" => bath,
                    "matrix_name" => String(name),
                )))
            end
        end
    end
end

function save_correlator_matrix(outdir::AbstractString, prefix::AbstractString,
                                C::AbstractMatrix{<:Complex};
                                matrix_formats::Vector{Symbol} = [:csv, :jld2],
                                metadata::Dict{String, Any} = Dict{String, Any}())
    mkpath(outdir)
    base = joinpath(outdir, "$(prefix)_correlator")
    if :csv in matrix_formats
        save_matrix_csvs(base, C)
    end
    if :jld2 in matrix_formats
        save_matrix_jld2(base * ".jld2", C; metadata=merge(metadata, Dict(
            "matrix_name" => "correlator",
        )))
    end
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
#=
function component_plot_settings(values::AbstractMatrix{<:Real}, component::Symbol)
    if component == :abs
        vmax = maximum(values)
        vmax = vmax == 0 ? 1.0 : vmax
        return :viridis, (0.0, vmax)
    end

    vmax = maximum(abs.(values))
    vmax = vmax == 0 ? 1.0 : vmax
    return :balance, (-vmax, vmax)
end=#

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
#=
function render_heatmap(path::AbstractString, mat::AbstractMatrix{<:Complex};
                        component::Symbol = :abs, title::AbstractString = "")
    values = matrix_component(mat, component)
    cmap, crange = component_plot_settings(values, component)

    fig = Figure(size=(700, 550))
    ax = Axis(
        fig[1, 1],
        title=title,
        xlabel="j",
        ylabel="i",
        xticks=collect(1:size(values, 2)),
        yticks=collect(1:size(values, 1)),
    )
    hm = heatmap!(ax, 1:size(values, 2), 1:size(values, 1), values;
                    colormap=cmap, colorrange=(0.0,1.0))              
    #colormap=cmap, colorrange=crange)
    Colorbar(fig[1, 2], hm, label=String(component))
    save(path, fig)
    return path
end=#

function save_heatmaps(outdir::AbstractString, prefix::AbstractString,
                       blocks_by_bath::Dict{Int, Dict{Symbol, Matrix{ComplexF64}}};
                       components::Vector{Symbol} = [:abs],
                       image_formats::Vector{Symbol} = [:png])
    mkpath(outdir)
    counts = 1
    titles = [L"\gamma^{(l)}_{g;ij}", L"\gamma^{(l)}_{l;ij}", L"\gamma^{(r)}_{g;ij}", L"\gamma^{(r)}_{l;ij}"]
    for bath in sort(collect(keys(blocks_by_bath)))
        for (name, mat) in sort(collect(blocks_by_bath[bath]); by=first)
            title = titles[counts]
            for component in components
                for fmt in image_formats
                    path = joinpath(
                        outdir,
                        "$(prefix)_bath$(bath)_$(String(name))_$(String(component)).$(String(fmt))",
                    )
                    render_heatmap(path, mat;
                                   component=component,
                                   title=title)
                end
            end
            counts = (counts % 4) + 1
        end
    end
end

end # module JumpHeatmapLib
