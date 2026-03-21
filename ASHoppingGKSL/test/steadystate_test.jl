using Test
using LinearAlgebra
import ASHoppingGKSL as gksl

@testset "Equilibrium" begin
    μ = 0.0
    t = 1.0
    ω = 0.0
    β = 8.0
    Γ = 1.0
    #Γ = 0.02
    τc = 0.2
    tol = 1e-8

    sys = gksl.SystemParams(t, ω)
    bath1 = gksl.make_bath(Gamma = Γ, tau_c = τc, beta = β, mu = -μ)
    bath2 = gksl.make_bath(Gamma = Γ, tau_c = τc, beta = β, mu = μ)
    m = gksl.TwoBathModel(sys, bath1, bath2)
    rho_red, rho_nr, _ = gksl.steady_states(m)
    rho_td = gksl.steady_state_tb_ncondition(m, 1.0)
    @testset "Is steady state equilibrium" begin
        tol2 = 1e-4
        println("Steady state consistency test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        #_, H_LS_eq = gksl.redfield_parts(m)
        #_, H_LS_gksl_eq = gksl.normalgksl_parts(m)
        thermal_eq = gksl.thermal_grand_canonical_state(gksl.system_hamiltonian(sys), β)
        #thermal_eq = gksl.thermal_grand_canonical_state(H_LS_eq, β)
        println("Thermal state: $([thermal_eq[i,i] for i in 1:size(thermal_eq, 1)])")
        println("Redfield steady state: $([rho_red[i,i] for i in 1:size(rho_red, 1)])")
        println("GKSL steady state:    $([rho_nr[i,i] for i in 1:size(rho_nr, 1)])")
        println(real.(rho_red - thermal_eq))
        @test norm(rho_red - thermal_eq) < tol2
        @test norm(rho_nr - thermal_eq) < tol2
    end

    @testset "Twisted boundary condition steady state" begin
        
        tol2 = 1e-4
        println("Twisted boundary condition steady state test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        thermal_eq = gksl.thermal_grand_canonical_state(gksl.system_hamiltonian(sys), β)
        println("Thermal state: $([thermal_eq[i,i] for i in 1:size(thermal_eq, 1)])")
        println("Redfield TD steady state: $([rho_td[i,i] for i in 1:size(rho_td, 1)])")
        @test norm(rho_td - thermal_eq) < tol2
    end

    @testset "Gain-Loss balance" begin
        gl_gksl = gksl.site_gain_loss_bath(m, rho_nr; method=:gksl)
        gl_red = gksl.site_gain_loss_bath(m, rho_red; method=:redfield)

        println("Gain-Loss net flow test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        println("tolerance = $(tol)")
        println("Redfield gain: (bath, site); (1,1)=$(gl_red[(1,1)][1]), (1,2)=$(gl_red[(1,2)][1]), (2,1)=$(gl_red[(2,1)][1]), (2,2)=$(gl_red[(2,2)][1])")
        println("GKSL gain:    (bath, site); (1,1)=$(gl_gksl[(1,1)][1]), (1,2)=$(gl_gksl[(1,2)][1]), (2,1)=$(gl_gksl[(2,1)][1]), (2,2)=$(gl_gksl[(2,2)][1])")
        println("Redfield loss: (bath, site); (1,1)=$(gl_red[(1,1)][2]), (1,2)=$(gl_red[(1,2)][2]), (2,1)=$(gl_red[(2,1)][2]), (2,2)=$(gl_red[(2,2)][2])")
        println("GKSL loss:    (bath, site); (1,1)=$(gl_gksl[(1,1)][2]), (1,2)=$(gl_gksl[(1,2)][2]), (2,1)=$(gl_gksl[(2,1)][2]), (2,2)=$(gl_gksl[(2,2)][2])")
        println("Redfield net: (bath, site); (1,1)=$(gl_red[(1,1)][3]), (1,2)=$(gl_red[(1,2)][3]), (2,1)=$(gl_red[(2,1)][3]), (2,2)=$(gl_red[(2,2)][3])")
        println("GKSL net:    (bath, site); (1,1)=$(gl_gksl[(1,1)][3]), (1,2)=$(gl_gksl[(1,2)][3]), (2,1)=$(gl_gksl[(2,1)][3]), (2,2)=$(gl_gksl[(2,2)][3])")

        @test all(abs(gl_red[(bath,site)][3]) < tol for site in (1, 2), bath in (1,2))
        @test all(abs(gl_gksl[(bath,site)][3]) < tol for site in (1, 2), bath in (1,2))
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
        @test abs(J_red) < tol
        @test abs(J_nr) < tol
    end

    @testset "Current expectation under TBC" begin
        rho_td = gksl.steady_state_tb_ncondition(m, 1.0)
        L_diss_eq, H_LS_eq = gksl.redfield_parts_TwistedBoundary(m)
        j_eq = gksl.current_expectation(rho_td, sys, H_LS_eq)
        tol2 = 1e-4
        println("Twisted boundary condition steady state test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        thermal_eq = gksl.thermal_grand_canonical_state(gksl.system_hamiltonian(sys), β)
        println("Thermal state: $([thermal_eq[i,i] for i in 1:size(thermal_eq, 1)])")
        println("Redfield TD steady state: $([rho_td[i,i] for i in 1:size(rho_td, 1)])")
        @test norm(rho_td - thermal_eq) < tol2
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

    sys = gksl.SystemParams(t, ω)
    bath1 = gksl.make_bath(Gamma = Γ, tau_c = τc, beta = β, mu = -μ)
    bath2 = gksl.make_bath(Gamma = Γ, tau_c = τc, beta = β, mu = μ)
    m = gksl.TwoBathModel(sys, bath1, bath2)
    rho_red, rho_gksl, rho_nr, _ = gksl.steady_states3(m)
    rho_td = gksl.steady_state_tb_ncondition(m, 1.0)

    @testset "Current expectation" begin
        _, H_LS = gksl.redfield_parts(m)
        _, H_LS_gksl = gksl.normalgksl_parts(m)
        _, H_LS_TB = gksl.redfield_parts_TwistedBoundary(m)
        j_red = gksl.current_expectation(rho_red, sys, H_LS)
        j_gksl = gksl.current_expectation(rho_gksl, sys, H_LS_gksl)
        j_nr = gksl.current_expectation(rho_nr, sys, H_LS)
        j_td = gksl.current_expectation(rho_td, sys, H_LS_TB)
        #J_red = gksl.current_expectation(rho_red, sys)
        #J_nr = gksl.current_expectation(rho_nr, sys)
        println("Current expectation test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        println("tolerance = $(tol)")
        println("j_eq")
        println("Redfield current: $(j_red)")
        println("NR current:    $(j_nr)")
        println("GKSL current:    $(j_gksl)")
        println("Twisted Boundary current: $(j_td)")
        @test abs(j_red) > tol
        @test abs(j_nr) > tol
        @test abs(j_td) > tol
        @test abs(j_gksl) < tol
    end


end