using Test

project_dir = abspath(joinpath(@__DIR__, ".."))
script_path = joinpath(project_dir, "script", "compute_local_observables.jl")

@testset "local observables CLI" begin
    output = read(`julia --project=$project_dir $script_path --method nr --nsites 4 --t 1.0 --omega 0.0 --mu 0.8 --beta 8.0 --Gamma 1.0 --tau-c 0.2`, String)
    @test occursin("beta_eff", output)
    @test occursin("mu_eff", output)
    @test occursin("current", output)
    @test occursin("central_indices", output)
end
