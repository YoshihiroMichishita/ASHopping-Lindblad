using Test
using LinearAlgebra
using CSV
using DataFrames
using JLD2

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "..")))
import ASHoppingGKSL as gksl

include(joinpath(@__DIR__, "..", "src2", "ASHoppingGKSL1P.jl"))
using .ASHoppingGKSL1P

function build_src2_model(; nsites::Int = 4, t::Float64 = 1.0, omega::Float64 = 0.0,
                          mu::Float64 = 0.8, beta::Float64 = 8.0,
                          Gamma::Float64 = 1.0, tau_c::Float64 = 0.2,
                          bathkind::Symbol = :bath2)
    sys = gksl.SystemParams(nsites, t, omega)
    if bathkind == :bath
        bath1 = gksl.make_bath(Gamma=Gamma, mu=-mu, beta=beta, tau_c=tau_c)
        bath2 = gksl.make_bath(Gamma=Gamma, mu=mu, beta=beta, tau_c=tau_c)
    elseif bathkind == :bath2
        bath1 = gksl.make_bath2(Gamma=Gamma, mu=-mu, beta=beta, tau_c=tau_c)
        bath2 = gksl.make_bath2(Gamma=Gamma, mu=mu, beta=beta, tau_c=tau_c)
    else
        error("bathkind must be :bath or :bath2")
    end
    return sys, gksl.TwoBathModel(sys, bath1, bath2)
end

function correlator_from_rho(rho, sys)
    c_ops, cd_ops = gksl.site_ops(sys)
    n = sys.n_sites
    C = zeros(ComplexF64, n, n)
    for i in 1:n, j in 1:n
        C[i, j] = tr(cd_ops[i] * c_ops[j] * rho)
    end
    return C
end

@testset "src2 one-particle correlators" begin
    sys, model = build_src2_model()

    @testset "reduced-space helpers" begin
        c_red = ASHoppingGKSL1P.reduced_annihilation_ops(sys.n_sites)
        cd_red = ASHoppingGKSL1P.reduced_creation_ops(sys.n_sites)
        ids, ops = ASHoppingGKSL1P.reduced_bilinear_basis(sys.n_sites)
        reduced_dim = 1 + sys.n_sites + (sys.n_sites * (sys.n_sites - 1)) ÷ 2

        @test length(c_red) == sys.n_sites
        @test length(cd_red) == sys.n_sites
        @test size(c_red[1]) == (reduced_dim, reduced_dim)
        @test c_red === ASHoppingGKSL1P.reduced_annihilation_ops(sys.n_sites)
        @test cd_red === ASHoppingGKSL1P.reduced_creation_ops(sys.n_sites)
        @test c_red[1][1, 2] == 1.0
        @test cd_red[1][2, 1] == 1.0
        @test c_red[2][2, 1 + sys.n_sites + 1] == -1.0
        @test c_red[1][3, 1 + sys.n_sites + 1] == 1.0
        @test ids[1] == :I
        @test length(ops) == 1 + sys.n_sites^2

        Ored = zeros(ComplexF64, reduced_dim, reduced_dim)
        Ored[1, 1] = 0.25
        Ored[2:1 + sys.n_sites, 2:1 + sys.n_sites] .= 0.25 .* Matrix{ComplexF64}(I, sys.n_sites, sys.n_sites)
        Ored[2, 3] += 0.4
        s, X = ASHoppingGKSL1P.source_and_bilinear_from_reduced(Ored)
        @test s ≈ 0.25 atol = 1e-12
        @test X[1, 2] ≈ 0.4 atol = 1e-12
    end

    @testset "basis construction" begin
        basis = ASHoppingGKSL1P.bilinear_basis(sys)
        @test length(basis.ops) == 1 + sys.n_sites^2
        @test basis.ids[1] == :I
        @test size(basis.gram) == (1 + sys.n_sites^2, 1 + sys.n_sites^2)
    end

    @testset "NR correlator matches full-rho reference" begin
        _, rho_ref, _ = gksl.steady_states(model)
        @show C_ref = correlator_from_rho(rho_ref, sys)
        @show  C_new = ASHoppingGKSL1P.steady_correlator_nr(model)
        @test size(C_new) == (sys.n_sites, sys.n_sites)
        @test C_new ≈ C_ref atol = 1e-8
    end

    @testset "Redfield correlator matches full-rho reference" begin
        rho_ref, _, _ = gksl.steady_states(model)
        @show C_ref = correlator_from_rho(rho_ref, sys)
        @show C_new = ASHoppingGKSL1P.steady_correlator_redfield(model)
        @test size(C_new) == (sys.n_sites, sys.n_sites)
        @test C_new ≈ C_ref atol = 1e-8
    end

    @testset "weighted observables are reconstructed from C" begin
        loss_coeff = ComplexF64[0.2 0.1; 0.3 0.4]
        gain_coeff = ComplexF64[0.7 0.2; 0.1 0.6]
        C = ComplexF64[0.4 0.05; 0.07 0.3]
        weighted = ASHoppingGKSL1P.loss_gain_weighted_from_C(loss_coeff, gain_coeff, C)

        @test weighted.loss_weighted ≈ loss_coeff .* transpose(C) atol = 1e-12
        @test weighted.gain_weighted ≈ gain_coeff .* (Matrix{ComplexF64}(I, 2, 2) - C) atol = 1e-12
    end

    @testset "local two-site observables" begin
        C4 = ComplexF64[
            0.1 0.2 0.3 0.4;
            0.5 0.6 0.7 0.8;
            0.9 1.0 1.1 1.2;
            1.3 1.4 1.5 1.6;
        ]
        Cmid, inds = ASHoppingGKSL1P.central_block(C4)
        @test inds == (2, 3)
        @test Cmid == ComplexF64[0.6 0.7; 1.0 1.1]

        beta_ref = 1.7
        mu_ref = 0.3
        tmid = 1.2
        nminus = 1 / (exp(beta_ref * (-tmid - mu_ref)) + 1)
        nplus = 1 / (exp(beta_ref * (tmid - mu_ref)) + 1)
        d = 0.5 * (nminus + nplus)
        r = 0.5 * (nminus - nplus)
        yim = 0.08
        Cthermal = ComplexF64[
            d r + 1im * yim;
            r - 1im * yim d;
        ]

        obs = ASHoppingGKSL1P.local_beta_mu_from_Cmid(Cthermal, tmid)
        @test obs.beta_eff ≈ beta_ref atol = 1e-10
        @test obs.mu_eff ≈ mu_ref atol = 1e-10
        @test obs.n_minus ≈ nminus atol = 1e-10
        @test obs.n_plus ≈ nplus atol = 1e-10
        @test obs.residual_diag ≈ 0.0 atol = 1e-12
        @test obs.residual_imag ≈ abs(yim) atol = 1e-12

        nminus_half = 1 / (exp(beta_ref * (-tmid)) + 1)
        nplus_half = 1 / (exp(beta_ref * tmid) + 1)
        r_half = 0.5 * (nminus_half - nplus_half)
        beta_half = ASHoppingGKSL1P.local_beta_from_real_offdiag(ComplexF64[0.5 r_half; r_half 0.5], tmid)
        @test beta_half.beta_eff ≈ beta_ref atol = 1e-10

        @test ASHoppingGKSL1P.local_current_from_Cmid(Cthermal, tmid) ≈ 2 * tmid * yim atol = 1e-12
    end

    @testset "local observable IO helpers" begin
        Cio = ComplexF64[
            0.4 + 0.0im 0.1 + 0.2im;
            0.1 - 0.2im 0.6 + 0.0im;
        ]
        tmpdir = mktempdir()

        jld2_path = joinpath(tmpdir, "corr.jld2")
        jldopen(jld2_path, "w") do file
            file["matrix"] = Cio
        end
        @test ASHoppingGKSL1P.load_correlator_jld2(jld2_path) == Cio
        @test ASHoppingGKSL1P.load_correlator(jld2_path) == Cio

        real_df = DataFrame(row=["site_1", "site_2"], site_1=real.(Cio[:, 1]), site_2=real.(Cio[:, 2]))
        imag_df = DataFrame(row=["site_1", "site_2"], site_1=imag.(Cio[:, 1]), site_2=imag.(Cio[:, 2]))
        real_path = joinpath(tmpdir, "corr_real.csv")
        imag_path = joinpath(tmpdir, "corr_imag.csv")
        CSV.write(real_path, real_df)
        CSV.write(imag_path, imag_df)

        @test ASHoppingGKSL1P.load_correlator_csv_pair(real_path, imag_path) == Cio
        @test ASHoppingGKSL1P.load_correlator(joinpath(tmpdir, "corr"); format=:csv_pair) == Cio

        obs_from_file = ASHoppingGKSL1P.local_observables_from_file(jld2_path, 1.0)
        @test obs_from_file.Cmid == Cio
        @test obs_from_file.central_indices == (1, 2)
    end

    @testset "Gamma sweep local observables" begin
        tmpdir = mktempdir()
        tmid = 1.0
        beta_vals = Dict(0.6 => 1.1, 0.2 => 0.7, 0.4 => 0.9)

        for (gamma, beta_ref) in beta_vals
            dir = joinpath(tmpdir, "Gamma_$(gamma)")
            mkpath(dir)
            nminus = 1 / (exp(beta_ref * (-tmid)) + 1)
            nplus = 1 / (exp(beta_ref * tmid) + 1)
            r = 0.5 * (nminus - nplus)
            yim = 0.01 * gamma
            Cmid = ComplexF64[0.5 r + 1im * yim; r - 1im * yim 0.5]
            jldopen(joinpath(dir, "nr_correlator.jld2"), "w") do file
                file["matrix"] = Cmid
            end
        end

        sweep = ASHoppingGKSL1P.collect_local_observables_over_gamma(tmpdir; filename="nr_correlator.jld2", t=tmid)
        @test sweep.Gamma == [0.2, 0.4, 0.6]
        @test sweep.beta_eff ≈ [0.7, 0.9, 1.1] atol = 1e-10
        @test sweep.temp_eff ≈ 1.0 ./ sweep.beta_eff atol = 1e-10
        @test sweep.mu_eff ≈ zeros(3) atol = 1e-10
        @test sweep.current ≈ [2 * tmid * 0.01 * g for g in sweep.Gamma] atol = 1e-12
    end

    @testset "isolated canonical correlator" begin
        sys2 = gksl.SystemParams(2, 1.0, 0.0)
        beta2 = 1.4
        C2 = ASHoppingGKSL1P.isolated_canonical_correlator(sys2, beta2)
        r2 = 0.5 * tanh(beta2 * sys2.t)
        C2_expected = ComplexF64[0.5 r2; r2 0.5]

        @test C2 ≈ C2_expected atol = 1e-12

        iso2 = ASHoppingGKSL1P.isolated_local_observables(sys2, beta2)
        @test iso2.C ≈ C2_expected atol = 1e-12
        @test iso2.beta_eff ≈ 2 * beta2 atol = 1e-10
        @test iso2.mu_eff ≈ 0.0 atol = 1e-10
        @test iso2.current ≈ 0.0 atol = 1e-12

        sys4 = gksl.SystemParams(4, 1.0, 0.0)
        C4 = ASHoppingGKSL1P.isolated_canonical_correlator(sys4, 0.8)
        @test C4 ≈ C4' atol = 1e-12
        @test real(tr(C4)) ≈ 2.0 atol = 1e-12
    end

    @testset "global effective temperature" begin
        sysg = gksl.SystemParams(4, 1.0, 0.0)
        beta_ref = 1.3
        eps_modes, U = gksl.single_particle_eigensystem(sysg)
        occ = 1.0 ./ (exp.(beta_ref .* eps_modes) .+ 1.0)
        Cth = ComplexF64.(U * Diagonal(occ) * U')

        obs = ASHoppingGKSL1P.global_effective_temperature(Cth, sysg)
        @test obs.beta_eff ≈ beta_ref atol = 1e-10
        @test obs.mu_eff ≈ 0.0 atol = 1e-12
        @test obs.mode_energies ≈ eps_modes atol = 1e-12
        @test obs.mode_occupations ≈ occ atol = 1e-12
        @test obs.offdiag_norm ≈ 0.0 atol = 1e-12
        @test obs.fit_residual ≈ 0.0 atol = 1e-10
    end
end
