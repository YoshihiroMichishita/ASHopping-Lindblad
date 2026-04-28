using Test

project_dir = abspath(joinpath(@__DIR__, ".."))
script_path = joinpath(project_dir, "script", "plot_jump_heatmap_gamma_sweep.sh")

@testset "plot jump heatmap Gamma sweep shell script" begin
    outroot = mktempdir()
    cmd = `bash $script_path --outroot $outroot 0.4 0.9 -- --nsites 3 --mu 0.8 --beta 8.0 --tau-c 0.2 --method nr --formats png --matrix-formats csv,jld2 --components abs`
    run(cmd)

    dir1 = joinpath(outroot, "Gamma_0.4")
    dir2 = joinpath(outroot, "Gamma_0.9")

    @test isdir(dir1)
    @test isdir(dir2)
    @test isfile(joinpath(dir1, "nr_bath1_loss_coeff_abs.png"))
    @test isfile(joinpath(dir1, "nr_correlator.jld2"))
    @test isfile(joinpath(dir2, "nr_bath2_gain_weighted_abs.png"))
    @test isfile(joinpath(dir2, "nr_correlator_real.csv"))
end
