using Test
import ASHoppingGKSL as gksl

@testset "Setting" begin
    omega::Float64 = 0.8
    mu::Float64 = 0.0
    beta::Float64 = 8.0
    Gamma::Float64 = 1.0
    tau_c::Float64 = 1.0

    sys = gksl.SystemParams(2, 1.0, omega)
    bathL = gksl.make_bath(Gamma = Gamma, mu = mu, beta = beta, tau_c = tau_c)
    bathR = gksl.make_bath(Gamma = Gamma, mu = -mu, beta = beta, tau_c = tau_c)
    model = gksl.TwoBathModel(sys, bathL, bathR)

    c1, c2, c1_dag, c2_dag = gksl.fermion_ops_2site()
    #rho0 = |00><00|
    rho0 = zeros(ComplexF64, 4, 4)
    rho0[1, 1] = 1.0
    #rho1 = |10><10| 
    rho1 = zeros(ComplexF64, 4, 4)
    rho1[2, 2] = 1.0
    #rho2 = |01><01|
    rho2 = zeros(ComplexF64, 4, 4)
    rho2[3, 3] = 1.0
    #rho3 = |11><11|
    rho3 = zeros(ComplexF64, 4, 4)
    rho3[4, 4] = 1.0

    @testset "Operator" begin
        @testset "loss" begin
            println("loss site1")
            @test c1 * rho0 * c1_dag == zeros(ComplexF64, 4, 4)
            @test c1 * rho1 *c1_dag == rho0
            @test c1 * rho2 *c1_dag == zeros(ComplexF64, 4, 4)
            @test c1 * rho3 *c1_dag == rho2

            println("loss site2")
            @test c2 * rho0 * c2_dag == zeros(ComplexF64, 4, 4)
            @test c2 * rho1 *c2_dag == zeros(ComplexF64, 4, 4)
            @test c2 * rho2 *c2_dag == rho0
            @test c2 * rho3 *c2_dag == rho1
        end

        @testset "gain" begin
            println("gain site1")
            @test c1_dag * rho0 * c1 == rho1
            @test c1_dag * rho1 * c1 == zeros(ComplexF64, 4, 4)
            @test c1_dag * rho2 * c1 == rho3
            @test c1_dag * rho3 * c1 == zeros(ComplexF64, 4, 4)

            println("gain site2")
            @test c2_dag * rho0 * c2 == rho2
            @test c2_dag * rho1 * c2 == rho3
            @test c2_dag * rho2 * c2 == zeros(ComplexF64, 4, 4)
            @test c2_dag * rho3 * c2 == zeros(ComplexF64, 4, 4)
        end

        @testset "number" begin
            println("number site1")
            @test c1_dag * c1 * rho0 == zeros(ComplexF64, 4, 4)
            @test c1_dag * c1 * rho1 == rho1
            @test c1_dag * c1 * rho2 == zeros(ComplexF64, 4, 4)
            @test c1_dag * c1 * rho3 == rho3

            println("number site2")
            @test c2_dag * c2 * rho0 == zeros(ComplexF64, 4, 4)
            @test c2_dag * c2 * rho1 == zeros(ComplexF64, 4, 4)
            @test c2_dag * c2 * rho2 == rho2
            @test c2_dag * c2 * rho3 == rho3

            println("1-number site1")
            @test (c1 * c1_dag) * rho0 == rho0
            @test (c1 * c1_dag) * rho1 == zeros(ComplexF64, 4, 4)
            @test (c1 * c1_dag) * rho2 == rho2
            @test (c1 * c1_dag) * rho3 == zeros(ComplexF64, 4, 4)

            println("1-number site2")
            @test (c2 * c2_dag) * rho0 == rho0
            @test (c2 * c2_dag) * rho1 == rho1
            @test (c2 * c2_dag) * rho2 == zeros(ComplexF64, 4, 4)
            @test (c2 * c2_dag) * rho3 == zeros(ComplexF64, 4, 4)
        end

        @testset "commutation" begin
            println("commutation")
            @test c1 * c2 == -c2 * c1
            @test c1_dag * c2 == -c2 * c1_dag
            @test c1 * c2_dag == -c2_dag * c1
            @test c1_dag * c2_dag == -c2_dag * c1_dag
        end
    end

    @testset "Vectorization to Liouville space" begin
        d = 4
        @testset "loss" begin
            println("loss site1")
            @test gksl.spost(c1_dag, d) * gksl.spre(c1, d) * vec(rho0) == zeros(ComplexF64, 16)
            @test gksl.spre(c1, d) * gksl.spost(c1_dag, d) * vec(rho0) == zeros(ComplexF64, 16)
            
            @test gksl.spost(c1_dag, d) * gksl.spre(c1, d) * vec(rho1) == vec(rho0)
            @test gksl.spre(c1, d) * gksl.spost(c1_dag, d) * vec(rho1) == vec(rho0)

            @test gksl.spost(c1_dag, d) * gksl.spre(c1, d) * vec(rho2) == zeros(ComplexF64, 16)
            @test gksl.spre(c1, d) * gksl.spost(c1_dag, d) * vec(rho2) == zeros(ComplexF64, 16)

            @test gksl.spost(c1_dag, d) * gksl.spre(c1, d) * vec(rho3) == vec(rho2)
            @test gksl.spre(c1, d) * gksl.spost(c1_dag, d) * vec(rho3) == vec(rho2)

            println("loss site2")
            @test gksl.spost(c2_dag, d) * gksl.spre(c2, d) * vec(rho0) == zeros(ComplexF64, 16)
            @test gksl.spre(c2, d) * gksl.spost(c2_dag, d) * vec(rho0) == zeros(ComplexF64, 16)
            @test gksl.spost(c2_dag, d) * gksl.spre(c2, d) * vec(rho1) == zeros(ComplexF64, 16)
            @test gksl.spre(c2, d) * gksl.spost(c2_dag, d) * vec(rho1) == zeros(ComplexF64, 16)
            @test gksl.spost(c2_dag, d) * gksl.spre(c2, d) * vec(rho2) == vec(rho0)
            @test gksl.spre(c2, d) * gksl.spost(c2_dag, d) * vec(rho2) == vec(rho0)
            @test gksl.spost(c2_dag, d) * gksl.spre(c2, d) * vec(rho3) == vec(rho1)
            @test gksl.spre(c2, d) * gksl.spost(c2_dag, d) * vec(rho3) == vec(rho1)
        end

        @testset "gain" begin
            println("gain site1")
            @test gksl.spost(c1, d) * gksl.spre(c1_dag, d) * vec(rho0) == vec(rho1)
            @test gksl.spre(c1_dag, d) * gksl.spost(c1, d) * vec(rho0) == vec(rho1)
            @test gksl.spost(c1, d) * gksl.spre(c1_dag, d) * vec(rho1) == zeros(ComplexF64, 16)
            @test gksl.spre(c1_dag, d) * gksl.spost(c1, d) * vec(rho1) == zeros(ComplexF64, 16)
            @test gksl.spost(c1, d) * gksl.spre(c1_dag, d) * vec(rho2) == vec(rho3)
            @test gksl.spre(c1_dag, d) * gksl.spost(c1, d) * vec(rho2) == vec(rho3)
            @test gksl.spost(c1, d) * gksl.spre(c1_dag, d) * vec(rho3) == zeros(ComplexF64, 16)
            @test gksl.spre(c1_dag, d) * gksl.spost(c1, d) * vec(rho3) == zeros(ComplexF64, 16)

            println("gain site2")
            @test gksl.spost(c2, d) * gksl.spre(c2_dag, d) * vec(rho0) == vec(rho2)
            @test gksl.spre(c2_dag, d) * gksl.spost(c2, d) * vec(rho0) == vec(rho2)
            @test gksl.spost(c2, d) * gksl.spre(c2_dag, d) * vec(rho1) == vec(rho3)
            @test gksl.spre(c2_dag, d) * gksl.spost(c2, d) * vec(rho1) == vec(rho3)
            @test gksl.spost(c2, d) * gksl.spre(c2_dag, d) * vec(rho2) == zeros(ComplexF64, 16)
            @test gksl.spre(c2_dag, d) * gksl.spost(c2, d) * vec(rho2) == zeros(ComplexF64, 16)
            @test gksl.spost(c2, d) * gksl.spre(c2_dag, d) * vec(rho3) == zeros(ComplexF64, 16)
            @test gksl.spre(c2_dag, d) * gksl.spost(c2, d) * vec(rho3) == zeros(ComplexF64, 16)
        end

        @testset "number" begin
            println("number site1")
            @test gksl.spre(c1_dag, d) * gksl.spre(c1, d) * vec(rho0) == zeros(ComplexF64, 16)
            @test gksl.spost(c1, d) * gksl.spost(c1_dag, d) * vec(rho0) == zeros(ComplexF64, 16)
            @test gksl.spre(c1, d) * gksl.spre(c1_dag, d) * vec(rho0) == vec(rho0)
            @test gksl.spost(c1_dag, d) * gksl.spost(c1, d) * vec(rho0) == vec(rho0)

            @test gksl.spre(c1_dag, d) * gksl.spre(c1, d) * vec(rho1) == vec(rho1)
            @test gksl.spost(c1, d) * gksl.spost(c1_dag, d) * vec(rho1) == vec(rho1)
            @test gksl.spre(c1, d) * gksl.spre(c1_dag, d) * vec(rho1) == zeros(ComplexF64, 16)
            @test gksl.spost(c1_dag, d) * gksl.spost(c1, d) * vec(rho1) == zeros(ComplexF64, 16)

            @test gksl.spre(c1_dag, d) * gksl.spre(c1, d) * vec(rho2) == zeros(ComplexF64, 16)
            @test gksl.spost(c1, d) * gksl.spost(c1_dag, d) * vec(rho2) == zeros(ComplexF64, 16)
            @test gksl.spre(c1, d) * gksl.spre(c1_dag, d) * vec(rho2) == vec(rho2)
            @test gksl.spost(c1_dag, d) * gksl.spost(c1, d) * vec(rho2) == vec(rho2)

            @test gksl.spre(c1_dag, d) * gksl.spre(c1, d) * vec(rho3) == vec(rho3)
            @test gksl.spost(c1, d) * gksl.spost(c1_dag, d) * vec(rho3) == vec(rho3)
            @test gksl.spre(c1, d) * gksl.spre(c1_dag, d) * vec(rho3) == zeros(ComplexF64, 16)
            @test gksl.spost(c1_dag, d) * gksl.spost(c1, d) * vec(rho3) == zeros(ComplexF64, 16)

            println("number site2")
            @test gksl.spre(c2_dag, d) * gksl.spre(c2, d) * vec(rho0) == zeros(ComplexF64, 16)
            @test gksl.spost(c2, d) * gksl.spost(c2_dag, d) * vec(rho0) == zeros(ComplexF64, 16)
            @test gksl.spre(c2, d) * gksl.spre(c2_dag, d) * vec(rho0) == vec(rho0)
            @test gksl.spost(c2_dag, d) * gksl.spost(c2, d) * vec(rho0) == vec(rho0)

            @test gksl.spre(c2_dag, d) * gksl.spre(c2, d) * vec(rho1) == zeros(ComplexF64, 16)
            @test gksl.spost(c2, d) * gksl.spost(c2_dag, d) * vec(rho1) == zeros(ComplexF64, 16)
            @test gksl.spre(c2, d) * gksl.spre(c2_dag, d) * vec(rho1) == vec(rho1)
            @test gksl.spost(c2_dag, d) * gksl.spost(c2, d) * vec(rho1) == vec(rho1)

            @test gksl.spre(c2_dag, d) * gksl.spre(c2, d) * vec(rho2) == vec(rho2)
            @test gksl.spost(c2, d) * gksl.spost(c2_dag, d) * vec(rho2) == vec(rho2)
            @test gksl.spre(c2, d) * gksl.spre(c2_dag, d) * vec(rho2) == zeros(ComplexF64, 16)
            @test gksl.spost(c2_dag, d) * gksl.spost(c2, d) * vec(rho2) == zeros(ComplexF64, 16)

            @test gksl.spre(c2_dag, d) * gksl.spre(c2, d) * vec(rho3) == vec(rho3)
            @test gksl.spost(c2, d) * gksl.spost(c2_dag, d) * vec(rho3) == vec(rho3)
            @test gksl.spre(c2, d) * gksl.spre(c2_dag, d) * vec(rho3) == zeros(ComplexF64, 16)
            @test gksl.spost(c2_dag, d) * gksl.spost(c2, d) * vec(rho3) == zeros(ComplexF64, 16)
        end

        @testset "commutation" begin
            println("commutation")
            @test gksl.spre(c1, d) * gksl.spre(c2, d) == -gksl.spre(c2, d) * gksl.spre(c1, d)
            @test gksl.spost(c1, d) * gksl.spost(c2, d) == -gksl.spost(c2, d) * gksl.spost(c1, d)
            @test gksl.spre(c1_dag, d) * gksl.spre(c2, d) == -gksl.spre(c2, d) * gksl.spre(c1_dag, d)
            @test gksl.spost(c1_dag, d) * gksl.spost(c2, d) == -gksl.spost(c2, d) * gksl.spost(c1_dag, d)
            @test gksl.spre(c1, d) * gksl.spre(c2_dag, d) == -gksl.spre(c2_dag, d) * gksl.spre(c1, d)
            @test gksl.spost(c1, d) * gksl.spost(c2_dag, d) == -gksl.spost(c2_dag, d) * gksl.spost(c1, d)
            @test gksl.spre(c1_dag, d) * gksl.spre(c2_dag, d) == -gksl.spre(c2_dag, d) * gksl.spre(c1_dag, d)
            @test gksl.spost(c1_dag, d) * gksl.spost(c2_dag, d) == -gksl.spost(c2_dag, d) * gksl.spost(c1_dag, d)
        end
    end
end