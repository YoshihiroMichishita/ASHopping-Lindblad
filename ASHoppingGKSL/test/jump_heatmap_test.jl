using Test
using LinearAlgebra

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..")))
import ASHoppingGKSL as gksl

include(joinpath(@__DIR__, "..", "script", "jump_heatmap_lib.jl"))

function build_model_for_jump_heatmap(; nsites::Int = 3, t::Float64 = 1.0, omega::Float64 = 0.0,
                                      mu::Float64 = 0.8, beta::Float64 = 8.0,
                                      Gamma::Float64 = 1.0, tau_c::Float64 = 0.2,
                                      bathkind::Symbol = :bath2)
    return JumpHeatmapLib.build_model(nsites, t, omega, mu, beta, Gamma, tau_c, bathkind)
end

@testset "Jump heatmap matrices" begin
    sys, model = build_model_for_jump_heatmap()
    coeff_keys = Set([:loss_coeff, :gain_coeff])
    weighted_keys = Set([:loss_coeff, :gain_coeff, :loss_weighted, :gain_weighted])

    @testset "NR coefficient diagonals match existing local coefficients" begin
        blocks = JumpHeatmapLib.nr_jump_blocks_by_bath(model)
        gains, losses = gksl.nr_local_gain_loss_coeffs_bath(model)

        for bath in 1:2
            @test Set(keys(blocks[bath])) == coeff_keys
            @test diag(blocks[bath][:gain_coeff]) ≈ ComplexF64.(gains[bath]) atol = 1e-10
            @test diag(blocks[bath][:loss_coeff]) ≈ ComplexF64.(losses[bath]) atol = 1e-10
        end
    end

    @testset "Redfield coefficient diagonals match existing local coefficients" begin
        blocks = JumpHeatmapLib.redfield_jump_blocks_by_bath(model)
        gain_diag = zeros(ComplexF64, sys.n_sites)
        loss_diag = zeros(ComplexF64, sys.n_sites)
        for bath in 1:2
            @test Set(keys(blocks[bath])) == coeff_keys
            gain_diag .+= diag(blocks[bath][:gain_coeff])
            loss_diag .+= diag(blocks[bath][:loss_coeff])
        end

        gain_ref, loss_ref = gksl.redfield_local_gain_loss_coeffs(model)
        @test gain_diag ≈ gain_ref atol = 1e-10
        @test loss_diag ≈ loss_ref atol = 1e-10
    end

    @testset "Correlator and weighted matrices have consistent shapes" begin
        rho = gksl.steady_state_tb_nr(model)
        corr = JumpHeatmapLib.correlator_blocks(rho, sys)
        blocks = JumpHeatmapLib.nr_jump_blocks_by_bath(model)
        weighted = JumpHeatmapLib.attach_correlators(blocks[1], corr)

        @test Set(keys(weighted)) == weighted_keys
        @test size(weighted[:loss_weighted]) == (sys.n_sites, sys.n_sites)
        @test size(weighted[:gain_weighted]) == (sys.n_sites, sys.n_sites)
        @test weighted[:loss_weighted] ≈ blocks[1][:loss_coeff] .* corr[:loss_corr] atol = 1e-10
        @test weighted[:gain_weighted] ≈ blocks[1][:gain_coeff] .* corr[:gain_corr] atol = 1e-10
    end
end

@testset "Jump heatmap CLI" begin
    outdir = mktempdir()
    project_dir = abspath(joinpath(@__DIR__, ".."))
    script_path = joinpath(project_dir, "script", "plot_jump_heatmap.jl")

    cmd = `julia --project=$project_dir $script_path \
        --nsites 3 --mu 0.8 --beta 8.0 --Gamma 1.0 --tau-c 0.2 \
        --method nr --formats png --matrix-formats csv,jld2 \
        --components abs --outdir $outdir`

    run(cmd)

    @test isfile(joinpath(outdir, "nr_bath1_loss_coeff_abs.png"))
    @test isfile(joinpath(outdir, "nr_bath1_loss_coeff_real.csv"))
    @test isfile(joinpath(outdir, "nr_bath1_loss_coeff.jld2"))
    @test !isfile(joinpath(outdir, "nr_bath1_loss_corr_abs.png"))
    @test !isfile(joinpath(outdir, "nr_bath1_pair_minus_coeff.jld2"))
    files = Set(readdir(outdir))
    @test all(occursin.(Ref(r"_(loss|gain)_(coeff|weighted)"), collect(files)))
end
