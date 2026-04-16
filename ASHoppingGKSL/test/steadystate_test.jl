using Test
using LinearAlgebra
import ASHoppingGKSL as gksl

n_sites = 4
scale = 1.0
tb_test = false

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
    Γ = 1.0
    #Γ = 0.02
    τc = 0.2
    tol = 1e-8

    sys = gksl.SystemParams(n_sites, t, ω)
    bath1 = gksl.make_bath(Gamma = Γ, tau_c = τc, beta = β, mu = -μ)
    bath2 = gksl.make_bath(Gamma = Γ, tau_c = τc, beta = β, mu = μ)
    m = gksl.TwoBathModel(sys, bath1, bath2)
    rho_red, rho_nr, _ = gksl.steady_states(m)
    if tb_test
        rho_td = gksl.steady_state_tb(m; scale=scale)
        rho_td_nr = gksl.steady_state_tb_nr(m; scale=scale)
    end
    #rho_td = gksl.steady_state_tb_ncondition(m, 1.0)
    @testset "Is steady state equilibrium" begin
        tol2 = 1e-4
        println("Steady state consistency test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        #_, H_LS_eq = gksl.redfield_parts(m)
        #_, H_LS_gksl_eq = gksl.normalgksl_parts(m)
        thermal_eq = gksl.thermal_grand_canonical_state(gksl.system_hamiltonian(sys)[2], β)
        #thermal_eq = gksl.thermal_grand_canonical_state(H_LS_eq, β)
        println("Thermal state: $([thermal_eq[i,i] for i in 1:size(thermal_eq, 1)])")
        println("Redfield steady state: $([rho_red[i,i] for i in 1:size(rho_red, 1)])")
        println("GKSL steady state:    $([rho_nr[i,i] for i in 1:size(rho_nr, 1)])")
        if tb_test
            println("TwistedBoundary steady state: $([rho_td[i,i] for i in 1:size(rho_td, 1)])")
            println("TwistedBoundary NR steady state: $([rho_td_nr[i,i] for i in 1:size(rho_td_nr, 1)])")
        end
        println(real.(rho_red - thermal_eq))
        @test norm(rho_red - thermal_eq) < tol2
        @test norm(rho_nr - thermal_eq) < tol2
        if tb_test
            @test norm(rho_td - thermal_eq) < tol2
            @test norm(rho_td_nr - thermal_eq) < tol2
        end
    end
    if tb_test
        @testset "Twisted boundary condition steady state" begin
            
            tol2 = 1e-6
            println("Twisted boundary condition steady state test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
            thermal_eq = gksl.thermal_grand_canonical_state(gksl.system_hamiltonian(sys)[2], β)
            println("Thermal state: $([thermal_eq[i,i] for i in 1:size(thermal_eq, 1)])")
            println("Redfield TD steady state: $([rho_td[i,i] for i in 1:size(rho_td, 1)])")
            println("NR TD steady state: $([rho_td_nr[i,i] for i in 1:size(rho_td_nr, 1)])")
            @test norm(rho_td - thermal_eq) < tol2
            @test norm(rho_td_nr - thermal_eq) < tol2
        end
    end

    @testset "Gain-Loss balance" begin
        gl_red = gksl.site_gain_loss_bath(m, rho_red; method=:redfield)
        gl_nr  = gksl.site_gain_loss_bath(m, rho_nr;  method=:nr)


        _, _, net_red = gksl.total_site_gain_loss(gl_red, sys.n_sites)
        _, _, net_nr  = gksl.total_site_gain_loss(gl_nr,  sys.n_sites)

        #net_red_tb = gksl.redfield_TB_net(m, rho_red; scale=scale)
        #net_nr_tb  = gksl.redfield_TB_net(m, rho_nr; scale=scale)

        println("Redfield net by site: $(net_red)")
        println("NR net by site: $(net_nr)")
        #println("Redfield TB net by site: $(net_red_tb)")
        #println("NR TB net by site: $(net_nr_tb)")

        @test all(abs(net_red[i]) < tol for i in [1,sys.n_sites])
        @test all(abs(net_nr[i])  < tol for i in [1,sys.n_sites])
        #@test all(abs(net_red_tb[i]) < tol for i in [1,sys.n_sites])
        #@test all(abs(net_nr_tb[i])  < tol for i in [1,sys.n_sites])
    end

    @testset "Current expectation" begin
        L_diss_eq, H_LS_eq = gksl.redfield_parts(m)
        _, H_LS_gksl_eq = gksl.normalgksl_parts(m)
        _, H_LS_tb = gksl.redfield_parts_TwistedBoundary(m; scale=scale)
        j_eq = gksl.current_expectation(rho_red, sys, H_LS_eq)
        j_eq_ngksl = gksl.current_expectation(rho_nr, sys, H_LS_gksl_eq)
        j_eq_gksl = gksl.current_expectation(rho_nr, sys, H_LS_eq)
        if tb_test
            j_eq_tb = gksl.current_expectation(rho_td, sys, H_LS_eq+H_LS_tb)
            j_eq_nr_tb = gksl.current_expectation(rho_td_nr, sys, H_LS_eq+H_LS_tb)
        end
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
        if tb_test
            println("TwistedBoundary Redfield current with LS: $(j_eq_tb)")
            println("TwistedBoundary GKSL current with normalgksl LS: $(j_eq_nr_tb)")
        end
        @test abs(J_red) < tol
        @test abs(J_nr) < tol
        @test abs(j_eq) < tol
        @test abs(j_eq_gksl) < tol
        @test abs(j_eq_ngksl) < tol
        if tb_test
            @test abs(j_eq_tb) < tol
            @test abs(j_eq_nr_tb) < tol
        end
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
    #rho_td = gksl.steady_state_tb_ncondition(m, 1.0)
    #rho_td = gksl.steady_state_tb(m; scale=scale)
    #rho_td_nr = gksl.steady_state_tb_nr(m; scale=scale)

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

        # steady state では全 bond current が等しい
        for b in 1:(length(j_red)-1)
            @test abs(j_red[b+1] - j_red[b]) < tol
            @test abs(j_nr[b+1]  - j_nr[b])  < tol
        end
    end

    #=
    @testset "Current expectation" begin
        _, H_LS = gksl.redfield_parts(m)
        _, H_LS_gksl = gksl.normalgksl_parts(m)
        #_, H_LS_TB = gksl.redfield_parts_TwistedBoundary(m; scale=scale)
        j_red = gksl.current_expectation(rho_red, sys, H_LS)
        j_gksl = gksl.current_expectation(rho_gksl, sys, H_LS_gksl)
        j_nr = gksl.current_expectation(rho_nr, sys, H_LS)
        #j_td = gksl.current_expectation(rho_td, sys, H_LS + H_LS_TB)
        #j_td_nr = gksl.current_expectation(rho_td_nr, sys, H_LS + H_LS_TB)
        #J_red = gksl.current_expectation(rho_red, sys)
        #J_nr = gksl.current_expectation(rho_nr, sys)
        println("Current expectation test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        println("tolerance = $(tol)")
        println("j_eq")
        println("Redfield current: $(j_red)")
        println("NR current:    $(j_nr)")
        println("GKSL current:    $(j_gksl)")
        #println("Twisted Boundary current: $(j_td)")
        #println("Twisted Boundary NR current: $(j_td_nr)")

        @test abs(j_red) > tol
        @test abs(j_nr) > tol
        #@test abs(j_td) > tol
        @test abs(j_gksl) < tol
        #@test abs(j_td_nr) > tol
    end=#

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
        
        #=
        _, H_LS_red = gksl.redfield_parts(m)
        _, H_LS_nr  = gksl.nr_parts_from_redfield(m)

        gl_red = gksl.site_gain_loss_bath(m, rho_red; method=:redfield)
        gl_nr  = gksl.site_gain_loss_bath(m, rho_nr;  method=:nr)

        gain_red, loss_red, net_red = gksl.total_site_gain_loss(gl_red, sys.n_sites)
        gain_nr,  loss_nr,  net_nr  = gksl.total_site_gain_loss(gl_nr,  sys.n_sites)

        j_red = gksl.current_expectation_sites(rho_red, sys, H_LS_red)
        j_nr  = gksl.current_expectation_sites(rho_nr,  sys, H_LS_nr)

        res_red = continuity_residuals(net_red, j_red)
        res_nr  = continuity_residuals(net_nr,  j_nr)=#

        println("Gain-loss-current consistency test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        println("tolerance = $(tol)")

        #println("Redfield gain by site: $(gain_red)")
        #println("Redfield loss by site: $(loss_red)")
        println("Redfield net by site:  $(net_red)")
        println("Redfield currents:     $(j_red)")
        println("Redfield continuity residuals: $(res_red)")

        #println("NR gain by site: $(gain_nr)")
        #println("NR loss by site: $(loss_nr)")
        println("NR net by site:  $(net_nr)")
        println("NR currents:     $(j_nr)")
        println("NR continuity residuals: $(res_nr)")

        # 全粒子数保存
        @test abs(sum(net_red)) < tol
        @test abs(sum(net_nr))  < tol

        # 各 site の連続の式
        @test all(abs.(res_red) .< tol)
        @test all(abs.(res_nr)  .< tol)

        # steady state では bond current は一様
        for b in 1:(length(j_red)-1)
            @test abs(j_red[b+1] - j_red[b]) < tol
            @test abs(j_nr[b+1]  - j_nr[b])  < tol
        end
    end

    #=
    @testset "gain-loss-current consistency" begin
        _, H_LS = gksl.redfield_parts(m)
        _, H_LS_gksl = gksl.normalgksl_parts(m)
        _, H_LS_nr = gksl.nr_parts_from_redfield(m)
        #_, H_LS_TB = gksl.redfield_parts_TwistedBoundary(m; scale=scale)

        gl_red = gksl.site_gain_loss_bath(m, rho_red; method=:redfield)
        #gl_red_TB = gksl.site_gain_loss_bath(m, rho_td; method=:redfield_tbc, scale=scale)
        gl_nr = gksl.site_gain_loss_bath(m, rho_nr; method=:nr)

        j_red = gksl.current_expectation(rho_red, sys, H_LS)
        #j_red_TB = gksl.current_expectation(rho_td, sys, H_LS + H_LS_TB)
        j_nr = gksl.current_expectation(rho_nr, sys, H_LS_nr)

        println("Gain-loss-current consistency test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        println("tolerance = $(tol)")
        println("Redfield gain: (site); (1)=$(gl_red[(1,1)][1] + gl_red[(2,1)][1]), (2)=$(gl_red[(1,2)][1]+gl_red[(2,2)][1])")
        #println("Redfield gain (TB): (site); (1)=$(gl_red_TB[(1,1)][1] + gl_red_TB[(2,1)][1]), (2)=$(gl_red_TB[(1,2)][1]+gl_red_TB[(2,2)][1])")
        println("NR gain:    (site); (1)=$(gl_nr[(1,1)][1] + gl_nr[(2,1)][1]), (2)=$(gl_nr[(1,2)][1]+gl_nr[(2,2)][1])")
        println("Redfield loss: (site); (1)=$(gl_red[(1,1)][2]+gl_red[(2,1)][2]), (2)=$(gl_red[(1,2)][2]+gl_red[(2,2)][2])")
        #println("Redfield loss (TB): (site); (1)=$(gl_red_TB[(1,1)][2]+gl_red_TB[(2,1)][2]), (2)=$(gl_red_TB[(1,2)][2]+gl_red_TB[(2,2)][2])")
        println("NR loss:    (site); (1)=$(gl_nr[(1,1)][2]+gl_nr[(2,1)][2]), (2)=$(gl_nr[(1,2)][2]+gl_nr[(2,2)][2])")
        println("Redfield net: (site); (1)=$(gl_red[(1,1)][3]+gl_red[(2,1)][3]), (2)=$(gl_red[(1,2)][3]+gl_red[(2,2)][3])")
        #println("Redfield net (TB): (site); (1)=$(gl_red_TB[(1,1)][3]+gl_red_TB[(2,1)][3]), (2)=$(gl_red_TB[(1,2)][3]+gl_red_TB[(2,2)][3])")
        println("NR net:    (site); (1)=$(gl_nr[(1,1)][3]+gl_nr[(2,1)][3]), (2)=$(gl_nr[(1,2)][3]+gl_nr[(2,2)][3])")
        println("Redfield current: $(j_red)")
        #println("Redfield current (TB): $(j_red_TB)")
        println("NR current:    $(j_nr)")
        @test abs((gl_red[(1,1)][3]+gl_red[(2,1)][3]) + (gl_red[(1,2)][3]+gl_red[(2,2)][3])) < tol
        #@test abs((gl_red_TB[(1,1)][3]+gl_red_TB[(2,1)][3]) + (gl_red_TB[(1,2)][3]+gl_red_TB[(2,2)][3])) < tol
        @test abs((gl_nr[(1,1)][3]+gl_nr[(2,1)][3]) + (gl_nr[(1,2)][3]+gl_nr[(2,2)][3])) < tol
        @test abs(j_red + (gl_red[(1,1)][3]+gl_red[(2,1)][3])) < tol
        @test abs(j_nr + (gl_nr[(1,1)][3]+gl_nr[(2,1)][3])) < tol
        #@test abs(j_red_TB + (gl_red_TB[(1,1)][3]+gl_red_TB[(2,1)][3])) < tol
    end=#

    #=
    @testset "TwistedBoundary time-dependent periodic steady state" begin
        tol_period = 1e-6

        Δμ = bath2.μ - bath1.μ
        T = 2π / abs(Δμ)
        Nt = 1000
        ts = collect(range(0.0, T; length = Nt + 1))
        dt = ts[2] - ts[1]

        rho0, ts_used, Hls_list, Ldiss_list = gksl.periodic_steady_state_td(m; Nt = Nt, scale=scale)

        staticL = gksl.redfield_liouvillian(m)
        rhoT = gksl.propagate_one_period(rho0, staticL, Hls_list, Ldiss_list, dt)
        println("rho=F: $([real(rhoT[i,i]) for i in 1:size(rhoT, 1)])")

        @test norm(rhoT - rho0) < tol_period
        @test abs(real(tr(rho0)) - 1.0) < tol
        @test minimum(real(eigvals(Hermitian(rho0)))) > -1e-8
    end

    @testset "TwistedBoundary time-averaged continuity" begin
        rho, ts, Hls_list, Ldiss_list = gksl.periodic_steady_state_td(m; Nt = 1000)
        staticL = gksl.redfield_liouvillian(m)
        dt = ts[2] - ts[1]
        tol_period = dt

        c1, c2, cd1, cd2 = gksl.fermion_ops_2site()
        n1 = cd1 * c1
        n2 = cd2 * c2

        L_static, H_LS_static = gksl.redfield_parts(m)

        dn1_avg = 0.0
        dn2_avg = 0.0
        j_avg = 0.0

        for k in 1:length(Hls_list)
            Lt = staticL + gksl.commutator_super(Hls_list[k]) + Ldiss_list[k]
            L_diss = L_static + Ldiss_list[k]
            drho = reshape(L_diss * vec(rho), size(rho)...)

            #TODO: Dissipationのみ抽出して粒子数変化を計算するようにする。
            dn1_avg += real(tr(n1 * drho)) * dt
            dn2_avg += real(tr(n2 * drho)) * dt
            j_avg += gksl.current_expectation(rho, sys, H_LS_static + Hls_list[k]) * dt

            rho = reshape(exp(dt * Lt) * vec(rho), size(rho)...)
            rho = 0.5 * (rho + rho')
            rho ./= real(tr(rho))
        end

        T = ts[end] - ts[1]
        dn1_avg /= T
        dn2_avg /= T
        j_avg /= T
        println("Twisted boundary condition time-averaged continuity test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        println("tolerance = $(tol_period)")
        println("Average dn1/dt: $(dn1_avg)")
        println("Average dn2/dt: $(dn2_avg)")
        println("Average current: $(j_avg)")

        @test abs(dn1_avg + dn2_avg) < tol_period
        @test abs(j_avg + dn1_avg) < tol_period
    end=#
    #=
    @testset "TwistedBoundary time-dependent periodic steady state" begin
        #tol_period = 1e-6

        Δμ = bath2.μ - bath1.μ
        T = 2π / abs(Δμ)
        Nt = 1000

        rho0, ts, Hls_list, Ldiss_list = gksl.periodic_steady_state_td(m; Nt=Nt, scale=scale)

        staticL = gksl.redfield_liouvillian(m)
        dt = ts[2] - ts[1]
        tol_period = 2dt

        rhoT = gksl.propagate_one_period(rho0, staticL, Hls_list, Ldiss_list, dt)

        println("TwistedBoundary periodic steady state test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        println("period T = $(T)")
        println("Nt = $(Nt), dt = $(dt)")
        println("rho0 diag = $([real(rho0[i,i]) for i in 1:size(rho0,1)])")
        println("rhoT diag = $([real(rhoT[i,i]) for i in 1:size(rhoT,1)])")
        println("||rho(T)-rho(0)|| = $(norm(rhoT - rho0))")

        @test norm(rhoT - rho0) < tol_period
        @test abs(real(tr(rho0)) - 1.0) < tol_period
        @test minimum(real(eigvals(Hermitian(rho0)))) > -tol_period
    end
    @testset "TwistedBoundary time-averaged continuity" begin
        Nt = 1000
        rho, ts, Hls_list, Ldiss_list = gksl.periodic_steady_state_td(m; Nt=Nt, scale=scale)

        staticL = gksl.redfield_liouvillian(m)
        L_diss_static, H_LS_static = gksl.redfield_parts(m)

        dt = ts[2] - ts[1]
        T = ts[end] - ts[1]
        tol_period = 5dt

        net_avg = zeros(Float64, sys.n_sites)
        j_avg = zeros(Float64, sys.n_sites - 1)

        rho_t = copy(rho)

        for k in 1:length(Hls_list)
            # total dissipative contribution at this time
            L_diss_t = L_diss_static + Ldiss_list[k]
            net_t = gksl.dissipative_site_net_from_liouvillian(L_diss_t, rho_t, sys)

            # coherent current at this time
            H_LS_t = H_LS_static + Hls_list[k]
            j_t = gksl.current_expectation_sites(rho_t, sys, H_LS_t)

            net_avg .+= net_t .* dt
            j_avg .+= j_t .* dt

            Lt = staticL + gksl.commutator_super(Hls_list[k]) + Ldiss_list[k]
            rho_t = reshape(exp(dt * Lt) * vec(rho_t), size(rho_t)...)
            rho_t = 0.5 * (rho_t + rho_t')
            rho_t ./= real(tr(rho_t))
        end

        net_avg ./= T
        j_avg ./= T

        res_avg = continuity_residuals(net_avg, j_avg)

        println("TwistedBoundary time-averaged continuity test (muL=$( -μ ), muR=$(μ), omega=$(ω))")
        println("period T = $(T)")
        println("time-averaged net by site = $(net_avg)")
        println("time-averaged currents = $(j_avg)")
        println("time-averaged continuity residuals = $(res_avg)")
        println("tolerance = $(tol_period)")

        @test abs(sum(net_avg)) < tol_period
        @test all(abs.(res_avg) .< tol_period)

        for b in 1:(length(j_avg)-1)
            @test abs(j_avg[b+1] - j_avg[b]) < tol_period
        end
    end=#


end