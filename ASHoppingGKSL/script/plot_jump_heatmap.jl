push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..")))
import ASHoppingGKSL as gksl

include(joinpath(@__DIR__, "jump_heatmap_lib.jl"))
using .JumpHeatmapLib

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

function blocks_for_method(m::gksl.TwoBathModel, method::Symbol)
    if method == :nr
        return nr_jump_blocks_by_bath(m)
    elseif method == :redfield
        return redfield_jump_blocks_by_bath(m)
    else
        error("method must be :nr or :redfield")
    end
end

function add_correlator_content(blocks_by_bath, rho, sys)
    corr = correlator_blocks(rho, sys)
    return Dict(bath => attach_correlators(blocks_by_bath[bath], corr) for bath in keys(blocks_by_bath))
end

function main(args)
    opts = parse_args(args)

    nsites = parse(Int, get_opt(opts, "nsites", 3))
    t = parse(Float64, get_opt(opts, "t", 1.0))
    omega = parse(Float64, get_opt(opts, "omega", 0.0))
    mu = parse(Float64, get_opt(opts, "mu", 0.8))
    beta = parse(Float64, get_opt(opts, "beta", 8.0))
    Gamma = parse(Float64, get_opt(opts, "Gamma", 1.0))
    tau_c = parse(Float64, get_opt(opts, "tau-c", 0.2))
    bathkind = Symbol(get_opt(opts, "bathkind", "bath2"))
    method = Symbol(get_opt(opts, "method", "nr"))
    outdir = abspath(get_opt(opts, "outdir", joinpath(@__DIR__, "..", "out", "jump_heatmaps")))
    image_formats = parse_symbol_list(get_opt(opts, "formats", "png"))
    matrix_formats = parse_symbol_list(get_opt(opts, "matrix-formats", "csv,jld2"))
    components = parse_symbol_list(get_opt(opts, "components", "abs"))
    with_corr = lowercase(get_opt(opts, "with-corr", "true")) in ("true", "1", "yes")

    sys, model = build_model(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind)
    blocks = blocks_for_method(model, method)

    if with_corr
        rho = default_rho_for_method(model, method)
        blocks = add_correlator_content(blocks, rho, sys)
    end

    metadata = Dict(
        "method" => String(method),
        "nsites" => nsites,
        "t" => t,
        "omega" => omega,
        "mu" => mu,
        "beta" => beta,
        "Gamma" => Gamma,
        "tau_c" => tau_c,
        "bathkind" => String(bathkind),
    )

    prefix = String(method)
    save_bath_blocks(outdir, prefix, blocks; matrix_formats=matrix_formats, metadata=metadata)
    save_heatmaps(outdir, prefix, blocks; components=components, image_formats=image_formats)

    println("Saved jump heatmap outputs to: ", outdir)
    for bath in sort(collect(keys(blocks)))
        println("  bath ", bath, ": ", join(sort(String.(keys(blocks[bath]))), ", "))
    end
end

main(ARGS)
