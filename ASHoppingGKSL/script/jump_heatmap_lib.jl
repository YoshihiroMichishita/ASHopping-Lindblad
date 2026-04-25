module JumpHeatmapLib

using LinearAlgebra
using CSV
using DataFrames
using JLD2
using CairoMakie

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..")))
import ASHoppingGKSL as gksl

export build_model,
       nr_jump_blocks_by_bath,
       redfield_jump_blocks_by_bath,
       correlator_blocks,
       attach_correlators,
       save_bath_blocks,
       save_heatmaps,
       render_heatmap,
       matrix_component,
       default_rho_for_method

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

function empty_block_dict(n::Int)
    return Dict(
        :loss_coeff => zeros(ComplexF64, n, n),
        :gain_coeff => zeros(ComplexF64, n, n),
    )
end

function accumulate_coefficients!(blocks::Dict, coeff::ComplexF64,
                                  u_left::AbstractVector{<:Complex},
                                  v_left::AbstractVector{<:Complex},
                                  u_right::AbstractVector{<:Complex},
                                  v_right::AbstractVector{<:Complex})
    blocks[:loss_coeff] .+= 2 * coeff .* (u_left * u_right')
    blocks[:gain_coeff] .+= 2 * coeff .* (v_left * v_right')
    return blocks
end

function nr_jump_blocks_by_bath(m::gksl.TwoBathModel)
    n = m.sys.n_sites
    jumps_by_bath = gksl.nr_jumps_by_bath(m)
    results = Dict{Int, Dict{Symbol, Matrix{ComplexF64}}}()

    for bath in 1:2
        blocks = empty_block_dict(n)
        for J in jumps_by_bath[bath]
            u, v = gksl.operator_linear_coeffs_in_site_basis(J, m.sys)
            accumulate_coefficients!(blocks, 1.0 + 0.0im, u, v, u, v)
        end
        results[bath] = blocks
    end

    return results
end

function redfield_jump_blocks_by_bath(m::gksl.TwoBathModel)
    n = m.sys.n_sites
    results = Dict{Int, Dict{Symbol, Matrix{ComplexF64}}}()

    for bath in 1:2
        blocks = empty_block_dict(n)
        site = gksl.bath_site(m, bath)
        spec = gksl.bath_by_index(m, bath)
        ops = gksl.coupling_ops_by_frequency(m.sys, site)
        freqs = gksl.frequency_list(ops)

        for p in gksl.majorana_channels(), q in gksl.majorana_channels(), w in freqs, wp in freqs
            Ap = ops[(p, w)]
            Aq = ops[(q, wp)]

            pref = (p == q) ? 1.0 : -1.0
            gamma_w, eta_w = gksl.gamma_eta_from_phi(spec, p, q, w)
            gamma_wp, eta_wp = gksl.gamma_eta_from_phi(spec, p, q, wp)
            Gamma = 0.5 * (gamma_w + gamma_wp) - 1im * (eta_w - eta_wp)

            u_p, v_p = gksl.operator_linear_coeffs_in_site_basis(Ap, m.sys)
            u_q, v_q = gksl.operator_linear_coeffs_in_site_basis(Aq, m.sys)
            accumulate_coefficients!(blocks, ComplexF64(pref * Gamma), u_q, v_q, u_p, v_p)
        end

        results[bath] = blocks
    end

    return results
end

function correlator_blocks(rho::AbstractMatrix{<:Complex}, sys::gksl.SystemParams)
    c_ops, cd_ops = gksl.site_ops(sys)
    n = sys.n_sites

    loss_corr = zeros(ComplexF64, n, n)
    gain_corr = zeros(ComplexF64, n, n)
    pair_minus_corr = zeros(ComplexF64, n, n)
    pair_plus_corr = zeros(ComplexF64, n, n)

    for i in 1:n, j in 1:n
        loss_corr[i, j] = tr(c_ops[i] * rho * cd_ops[j])
        gain_corr[i, j] = tr(cd_ops[i] * rho * c_ops[j])
        pair_minus_corr[i, j] = tr(c_ops[i] * rho * c_ops[j])
        pair_plus_corr[i, j] = tr(cd_ops[i] * rho * cd_ops[j])
    end

    return Dict(
        :loss_corr => loss_corr,
        :gain_corr => gain_corr,
        :pair_minus_corr => pair_minus_corr,
        :pair_plus_corr => pair_plus_corr,
    )
end

function attach_correlators(blocks::Dict{Symbol, Matrix{ComplexF64}},
                            corr::Dict{Symbol, Matrix{ComplexF64}})
    out = copy(blocks)
    out[:loss_weighted] = out[:loss_coeff] .* corr[:loss_corr]
    out[:gain_weighted] = out[:gain_coeff] .* corr[:gain_corr]
    return out
end

function default_rho_for_method(m::gksl.TwoBathModel, method::Symbol)
    if method == :nr
        return gksl.steady_state_tb_nr(m)
    elseif method == :redfield
        return gksl.steady_state_tb(m)
    else
        error("method must be :nr or :redfield")
    end
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

function component_plot_settings(values::AbstractMatrix{<:Real}, component::Symbol)
    if component == :abs
        vmax = maximum(values)
        vmax = vmax == 0 ? 1.0 : vmax
        return :viridis, (0.0, vmax)
    end

    vmax = maximum(abs.(values))
    vmax = vmax == 0 ? 1.0 : vmax
    return :balance, (-vmax, vmax)
end

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
end

function save_heatmaps(outdir::AbstractString, prefix::AbstractString,
                       blocks_by_bath::Dict{Int, Dict{Symbol, Matrix{ComplexF64}}};
                       components::Vector{Symbol} = [:abs],
                       image_formats::Vector{Symbol} = [:png])
    mkpath(outdir)
    for bath in sort(collect(keys(blocks_by_bath)))
        for (name, mat) in sort(collect(blocks_by_bath[bath]); by=first)
            for component in components
                for fmt in image_formats
                    path = joinpath(
                        outdir,
                        "$(prefix)_bath$(bath)_$(String(name))_$(String(component)).$(String(fmt))",
                    )
                    render_heatmap(path, mat;
                                   component=component,
                                   title="$(prefix) bath $(bath) $(String(name)) $(String(component))")
                end
            end
        end
    end
end

end # module JumpHeatmapLib
