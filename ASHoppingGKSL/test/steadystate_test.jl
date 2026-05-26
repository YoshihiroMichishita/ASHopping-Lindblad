using Test
using LinearAlgebra
import ASHoppingGKSL as gksl

n_sites = 4
scale = 1.0

function continuity_residuals(net, j)
    n = length(net)
    res = zeros(Float64, n)

    for i in 1:n
        if i == 1
            #res[i] = net[i] - j[i]
            res[i] = net[i] + j[i]
        elseif i == n
            #res[i] = net[i] + j[i-1]
            res[i] = net[i] - j[i-1]
        else
            #res[i] = net[i] - j[i] + j[i-1]
            res[i] = net[i] + j[i] - j[i-1]
        end
    end

    return res
end

@testset "Equilibrium" begin
    μ = 0.0
    t = 1.0
    ω = 0.0
    β = 8.0
    #Γ = 1.0
    Γ = 0.02
    τc = 0.2
    tol = 1e-8

    sys = gksl.SystemParams(n_sites, t, ω)
    bath1 = gksl.make_bath(Gamma = Γ, tau_c = τc, beta = β, mu = -μ)
    bath2 = gksl.make_bath(Gamma = Γ, tau_c = τc, beta = β, mu = μ)
    m = gksl.TwoBathModel(sys, bath1, bath2)
    rho_red, rho_nr, _ = gksl.steady_states(m)
    @testset "Is steady state equilibrium" begin
        tol2 = 1e-3
        println("Steady state consistency test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        thermal_eq = gksl.thermal_grand_canonical_state(gksl.system_hamiltonian(sys)[2], β)
        println("Thermal state: $([thermal_eq[i,i] for i in 1:size(thermal_eq, 1)])")
        println("Redfield steady state: $([rho_red[i,i] for i in 1:size(rho_red, 1)])")
        println("GKSL steady state:    $([rho_nr[i,i] for i in 1:size(rho_nr, 1)])")
        println(real.(rho_red - thermal_eq))
        @test norm(rho_red - thermal_eq) < tol2
        @test norm(rho_nr - thermal_eq) < tol2
    end

    @testset "Gain-Loss balance" begin
        gl_red = gksl.site_gain_loss_bath(m, rho_red; method=:redfield)
        gl_nr  = gksl.site_gain_loss_bath(m, rho_nr;  method=:nr)


        _, _, net_red = gksl.total_site_gain_loss(gl_red, sys.n_sites)
        _, _, net_nr  = gksl.total_site_gain_loss(gl_nr,  sys.n_sites)

        println("Redfield net by site: $(net_red)")
        println("NR net by site: $(net_nr)")

        @test all(abs(net_red[i]) < tol for i in [1,sys.n_sites])
        @test all(abs(net_nr[i])  < tol for i in [1,sys.n_sites])
    end

    @testset "Current expectation" begin
        L_diss_eq, H_LS_eq = gksl.redfield_parts(m)
        _, H_LS_gksl_eq = gksl.normalgksl_parts(m)
        j_eq = gksl.current_expectation(rho_red, sys, H_LS_eq)
        j_eq_ngksl = gksl.current_expectation(rho_nr, sys, H_LS_gksl_eq)
        j_eq_gksl = gksl.current_expectation(rho_nr, sys, H_LS_eq)

        J_red = gksl.current_expectation(rho_red, sys)
        J_nr = gksl.current_expectation(rho_nr, sys)
        println("Current expectation test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        println("tolerance = $(tol)")
        println("j_eq")
        println("Redfield current: $(J_red)")
        println("GKSL current:    $(J_nr)")
        println("Redfield current with LS: $(j_eq)")
        println("GKSL current with LS:    $(j_eq_gksl)")
        println("GKSL current with normalgksl LS:    $(j_eq_ngksl)")

        @test abs(J_red) < tol
        @test abs(J_nr) < tol
        @test abs(j_eq) < tol
        @test abs(j_eq_gksl) < tol
        @test abs(j_eq_ngksl) < tol
    end

    @testset "gain-loss-current consistency" begin
        _, H_LS_red = gksl.redfield_parts(m)
        _, H_LS_nr  = gksl.nr_parts_from_redfield(m)

        gl_red = gksl.site_gain_loss_bath(m, rho_red; method=:redfield)
        gl_nr  = gksl.site_gain_loss_bath(m, rho_nr;  method=:nr)

        gain_red, loss_red, net_red = gksl.total_site_gain_loss(gl_red, sys.n_sites)
        gain_nr,  loss_nr,  net_nr  = gksl.total_site_gain_loss(gl_nr,  sys.n_sites)

        j_red = gksl.current_expectation_sites(rho_red, sys, H_LS_red)
        j_nr  = gksl.current_expectation_sites(rho_nr,  sys, H_LS_nr)

        res_red = continuity_residuals(net_red, j_red)
        res_nr  = continuity_residuals(net_nr,  j_nr)

        println("Gain-loss-current consistency test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        println("tolerance = $(tol)")

        println("Redfield gain by site: $(gain_red)")
        println("Redfield loss by site: $(loss_red)")
        println("Redfield net by site:  $(net_red)")
        println("Redfield currents (right-to-left positive): $(j_red)")
        println("Redfield continuity residuals: $(res_red)")

        println("NR gain by site: $(gain_nr)")
        println("NR loss by site: $(loss_nr)")
        println("NR net by site:  $(net_nr)")
        println("NR currents (right-to-left positive): $(j_nr)")
        println("NR continuity residuals: $(res_nr)")

        @test abs(sum(net_red)) < tol
        @test abs(sum(net_nr))  < tol

        @test all(abs.(res_red) .< tol)
        @test all(abs.(res_nr)  .< tol)

        for b in 1:(length(j_red)-1)
            @test abs(j_red[b+1] - j_red[b]) < tol
            @test abs(j_nr[b+1]  - j_nr[b])  < tol
        end
    end
end

@testset "Non-equilibrium" begin
    μ = 0.8
    t = 1.0
    ω = 0.0
    β = 8.0
    Γ = 1.0
    #Γ = 0.02
    τc = 0.2
    tol = 1e-8

    sys = gksl.SystemParams(n_sites, t, ω)
    bath1 = gksl.make_bath(Gamma = Γ, tau_c = τc, beta = β, mu = -μ)
    bath2 = gksl.make_bath(Gamma = Γ, tau_c = τc, beta = β, mu = μ)
    m = gksl.TwoBathModel(sys, bath1, bath2)
    rho_red, rho_gksl, rho_nr, _ = gksl.steady_states3(m)

    @testset "Current expectation" begin
        _, H_LS_red  = gksl.redfield_parts(m)
        _, H_LS_gksl = gksl.normalgksl_parts(m)
        _, H_LS_nr   = gksl.nr_parts_from_redfield(m)

        j_red  = gksl.current_expectation_sites(rho_red,  sys, H_LS_red)
        j_gksl = gksl.current_expectation_sites(rho_gksl, sys, H_LS_gksl)
        j_nr   = gksl.current_expectation_sites(rho_nr,   sys, H_LS_nr)

        println("Current expectation test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        println("tolerance = $(tol)")
        println("Redfield currents (right-to-left positive): $(j_red)")
        println("GKSL currents (right-to-left positive): $(j_gksl)")
        println("NR currents (right-to-left positive): $(j_nr)")

        # non-equilibriumなので少なくともどこかの bond で流れがある
        @test maximum(abs.(j_red)) > tol
        @test maximum(abs.(j_nr))  > tol
    end

    @testset "gain-loss-current consistency" begin
        _, H_LS_red = gksl.redfield_parts(m)
        _, H_LS_nr  = gksl.nr_parts_from_redfield(m)
        
        net_red_by_bath = gksl.site_net_bath(m, rho_red; method=:redfield)
        net_nr_by_bath  = gksl.site_net_bath(m, rho_nr;  method=:nr)

        net_red = gksl.total_site_net(net_red_by_bath, sys.n_sites)
        net_nr  = gksl.total_site_net(net_nr_by_bath,  sys.n_sites)

        j_red = gksl.current_expectation_sites(rho_red, sys, H_LS_red)
        j_nr  = gksl.current_expectation_sites(rho_nr,  sys, H_LS_nr)

        res_red = continuity_residuals(net_red, j_red)
        res_nr  = continuity_residuals(net_nr,  j_nr)

        println("Gain-loss-current consistency test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        println("tolerance = $(tol)")

        println("Redfield net by site:  $(net_red)")
        println("Redfield currents:     $(j_red)")
        println("Redfield continuity residuals: $(res_red)")

        println("NR net by site:  $(net_nr)")
        println("NR currents:     $(j_nr)")
        println("NR continuity residuals: $(res_nr)")

        # 全粒子数保存
        @test abs(sum(net_red)) < tol
        @test abs(sum(net_nr))  < tol

        # 各 site の連続の式
        @test all(abs.(res_red) .< tol)
        @test all(abs.(res_nr)  .< tol)
    end
end