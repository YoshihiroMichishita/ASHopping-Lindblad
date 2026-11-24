using Test
using JLD2

project_dir = abspath(joinpath(@__DIR__, ".."))
script_path = joinpath(project_dir, "script", "plot_jump_heatmap_mu_sweep.sh")

@testset "plot jump heatmap mu sweep shell script" begin
    outroot = mktempdir()
    cmd = Cmd([
        "bash", script_path,
        "--outroot", outroot,
        "--",
        "--nsites", "2",
        "--beta", "8.0",
        "--tau-c", "0.2",
        "--method", "nr",
        "--formats", "",
        "--matrix-formats", "jld2",
        "--components", "",
    ])
    run(cmd)

    expected_dirs = [
        "mu_0.0", "mu_0.2", "mu_0.4", "mu_0.6", "mu_0.8", "mu_1.0", "mu_1.2",
        "mu_1.4", "mu_1.6", "mu_1.8", "mu_2.0", "mu_2.2", "mu_2.4",
    ]
    actual_dirs = sort(filter(name -> isdir(joinpath(outroot, name)), readdir(outroot)))
    @test actual_dirs == expected_dirs

    @test isfile(joinpath(outroot, "mu_0.0", "nr_correlator.jld2"))
    @test isfile(joinpath(outroot, "mu_1.2", "nr_bath1_loss_coeff.jld2"))
    @test isfile(joinpath(outroot, "mu_2.4", "nr_bath2_gain_weighted.jld2"))

    for (dirname, expected_mu) in (("mu_0.0", 0.0), ("mu_1.2", 1.2), ("mu_2.4", 2.4))
        metadata = JLD2.load(joinpath(outroot, dirname, "nr_correlator.jld2"), "metadata")
        @test metadata["Gamma"] == 0.1
        @test metadata["mu"] == expected_mu
    end
end
