using Test
using CSV
using DataFrames

script_path = abspath(joinpath(@__DIR__, "..", "script", "plot_current_delta_temperature_linear_small_mu.jl"))

@testset "small-mu linear current-temperature plot" begin
    @test isfile(script_path)
    if isfile(script_path)
        include(script_path)
        script = SmallMuLinearPlot
        input_dir = mktempdir()
        csv_path = joinpath(input_dir, "input.csv")
        # Unsorted data exercise both inclusive boundaries and exclusion.
        CSV.write(csv_path, DataFrame(
            mu=[0.9, 0.6, 0.0, 0.3, -0.1],
            current=[-0.1, -0.075, 0.0, -0.027, 0.01],
            temp_eff=[0.9, 0.688, 0.0994, 0.197, 0.1],
            delta_temp_eff=fill(999.0, 5),
        ))
        original = read(csv_path)
        table = script.load_plot_data(csv_path)
        @test table.mu == [0.0, 0.3, 0.6]
        @test table.current_abs ≈ [0.0, 0.027, 0.075]
        @test table.delta_temp_eff ≈ [-0.0006, 0.097, 0.588]
        @test table.t0 == fill(0.1, 3)
        @test script.load_plot_data(csv_path; t0=0.2).delta_temp_eff ≈ [-0.1006, -0.003, 0.488]
        figure = script.build_linear_plot(table)
        @test figure.subplots[1][:xaxis][:scale] == :identity
        @test figure.subplots[1][:yaxis][:scale] == :identity
        @test figure.series_list[1][:x] ≈ [0.0, 0.027, 0.075]
        @test figure.series_list[1][:y] ≈ [-0.0006, 0.097, 0.588]

        outdir = joinpath(input_dir, "new-output")
        project_dir = abspath(joinpath(@__DIR__, ".."))
        command = `julia --project=$project_dir $script_path --csv $csv_path --outdir $outdir`
        run(addenv(command, "GKSwstype" => "100"))
        stem = "J_vs_delta_Teff_linear_mu_0_0p6"
        @test sort(readdir(outdir)) == sort(["$stem.csv", "$stem.pdf", "$stem.png"])
        @test all(filesize(joinpath(outdir, name)) > 0 for name in readdir(outdir))
        saved = CSV.read(joinpath(outdir, "$stem.csv"), DataFrame)
        @test saved.mu == [0.0, 0.3, 0.6]
        @test saved.delta_temp_eff ≈ [-0.0006, 0.097, 0.588]
        @test read(csv_path) == original
        outputs = Dict(name => read(joinpath(outdir, name)) for name in readdir(outdir))
        @test_throws ArgumentError script.main(["--csv", csv_path, "--outdir", outdir])
        @test all(read(joinpath(outdir, name)) == bytes for (name, bytes) in outputs)

        collision_dir = mktempdir()
        sentinel = joinpath(collision_dir, "$stem.png")
        write(sentinel, "existing output")
        @test_throws ArgumentError script.main(["--csv", csv_path, "--outdir", collision_dir])
        @test readdir(collision_dir) == ["$stem.png"]
        @test read(sentinel, String) == "existing output"
        @test_throws ArgumentError script.main(["--unknown", "value"])
        @test_throws ArgumentError script.main(["--csv"])
        @test_throws ArgumentError script.load_plot_data(csv_path; t0=NaN)
        empty_path = joinpath(input_dir, "empty-range.csv")
        CSV.write(empty_path, DataFrame(mu=[1.0], current=[0.1], temp_eff=[1.0]))
        @test_throws ArgumentError script.load_plot_data(empty_path)
        bad_path = joinpath(input_dir, "invalid.csv")
        CSV.write(bad_path, DataFrame(mu=[0.0], current=[0.0], temp_eff=[Inf]))
        @test_throws ArgumentError script.load_plot_data(bad_path)
    end
end
