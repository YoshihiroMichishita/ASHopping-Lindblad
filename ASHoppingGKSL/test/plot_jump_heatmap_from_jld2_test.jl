using Test
using JLD2

heatmap_script = abspath(joinpath(@__DIR__, "..", "script", "plot_jump_heatmap_from_jld2.jl"))

@testset "saved jump coefficients: fixed color range and no overwrite" begin
    @test isfile(heatmap_script)
    if isfile(heatmap_script)
        include(heatmap_script)
        script = JumpHeatmapFromJLD2
        indir = mktempdir()
        mat = ComplexF64[0.1 0.3 + 0.4im; 0.2im 0.05]
        inputs = String[]
        for bath in (1, 2), kind in ("gain", "loss")
            path = joinpath(indir, "nr_bath$(bath)_$(kind)_coeff.jld2")
            jldopen(path, "w") do file
                file["matrix"] = mat
                file["metadata"] = Dict("bath" => bath, "matrix_name" => "$(kind)_coeff")
            end
            push!(inputs, path)
        end
        originals = Dict(path => read(path) for path in inputs)
        built = script.build_coefficient_heatmap(mat; title="test")
        @test collect(built.heatmap.colorrange[]) ≈ [0.0, 0.3]
        @test built.colorbar.ticks[] == ([0.0, 0.15, 0.3], ["0.0", "0.15", "0.3"])
        @test built.heatmap[3][] ≈ [0.1 0.2; 0.5 0.05]

        project_dir = abspath(joinpath(@__DIR__, ".."))
        run(`julia --project=$project_dir $heatmap_script --indir $indir`)
        outdir = joinpath(indir, "heatmaps_clim_0_0p3")
        @test length(readdir(outdir)) == 8
        for bath in (1, 2), kind in ("gain", "loss"), extension in ("png", "pdf")
            path = joinpath(outdir, "nr_bath$(bath)_$(kind)_coeff_abs.$extension")
            @test isfile(path)
            @test filesize(path) > 0
            @test !isfile(joinpath(indir, basename(path)))
        end
        @test all(read(path) == bytes for (path, bytes) in originals)
        generated = Dict(joinpath(outdir, name) => read(joinpath(outdir, name)) for name in readdir(outdir))
        @test_throws ArgumentError script.main(["--indir", indir])
        @test all(read(path) == bytes for (path, bytes) in generated)

        # A collision with the last output must prevent every output write.
        collision_dir = mktempdir()
        existing = joinpath(collision_dir, "nr_bath2_loss_coeff_abs.pdf")
        write(existing, "existing file must survive")
        @test_throws ArgumentError script.main(["--indir", indir, "--outdir", collision_dir])
        @test readdir(collision_dir) == ["nr_bath2_loss_coeff_abs.pdf"]
        @test read(existing, String) == "existing file must survive"
        @test_throws ArgumentError script.main(["--unknown", "value"])
        @test_throws ArgumentError script.main(["--indir"])
        @test_throws ArgumentError script.build_coefficient_heatmap(zeros(ComplexF64, 2, 3))
        @test_throws ArgumentError script.build_coefficient_heatmap(fill(ComplexF64(NaN), 2, 2))
        @test_throws ArgumentError script.load_coefficient(joinpath(indir, "missing.jld2"))
    end
end
