using Test
using DataFrames

include(joinpath(@__DIR__, "..", "script", "plot_current_delta_temperature_linear_small_mu.jl"))

@testset "quadratic fit overlay without an intercept" begin
    script = SmallMuLinearPlot
    @test isdefined(script, :fit_quadratic)
    if isdefined(script, :fit_quadratic)
        # The negative zero-current point must remain in the data, but it
        # cannot shift a fit with no constant term.
        table = DataFrame(current_abs=[0.0, 0.01, 0.04, 0.08],
                          delta_temp_eff=[-0.0006, 0.0203, 0.0848, 0.1792])
        fit = script.fit_quadratic(table)
        @test fit.a ≈ 2.0
        @test fit.b ≈ 3.0
        @test fit.rmse ≈ 0.0003
        figure = script.build_linear_plot(table)
        @test length(figure.series_list) == 2
        @test figure.series_list[1][:y] == [-0.0006, 0.0203, 0.0848, 0.1792]
        curve = figure.series_list[2]
        @test first(curve[:x]) == 0.0
        @test last(curve[:x]) ≈ 0.08
        @test first(curve[:y]) == 0.0
        @test last(curve[:y]) ≈ 0.1792
        @test curve[:y] ≈ 2 .* curve[:x] .+ 3 .* curve[:x].^2
        @test curve[:linestyle] == :dash
        @test figure.subplots[1][:legend_position] == :topleft
        @test figure.subplots[1][:xaxis][:scale] == :identity
        @test figure.subplots[1][:yaxis][:scale] == :identity
        @test_throws ArgumentError script.fit_quadratic(DataFrame(
            current_abs=[0.0, 0.01], delta_temp_eff=[0.0, 0.1],
        ))

        actual = DataFrame(
            current_abs=[2.5442572714968083e-16, 0.007016584024841256,
                         0.011020018176640656, 0.027347940218751265,
                         0.05169493230470472, 0.07531948392088139],
            delta_temp_eff=[-0.0005903872930938664, 0.00994720592226142,
                            0.022466044419896997, 0.09706099306776853,
                            0.3033631594802664, 0.5884018883320129],
        )
        actual_fit = script.fit_quadratic(actual)
        @test actual_fit.a ≈ 1.24790470391635
        @test actual_fit.b ≈ 87.49816464734333
        @test actual_fit.rmse ≈ 0.0028617386186160347
    end
end
