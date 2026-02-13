using LinearAlgebra
using Random

const C64 = ComplexF64

struct SystemParams
    t::Float64
    omega::Float64
end

struct BathSpectrum
    Sigma::Function      # Σ(ω)
    SigmaTilde::Function # \tilde{Σ}(ω)
end

struct TwoBathModel
    sys::SystemParams
    bath1::BathSpectrum
    bath2::BathSpectrum
end

"""
Two-site spinless fermion operators in the explicit Fock basis order
[|00>, |01>, |10>, |11>].
"""
function fermion_ops_2site()
    I2 = Matrix{C64}(I, 2, 2)
    # Local basis is [|0>, |1>].
    c = C64[0 1; 0 0]
    sigma_z = C64[1 0; 0 -1]
    c1 = kron(c, I2)
    c2 = kron(sigma_z, c) # Jordan-Wigner string on site 1
    return c1, c2, c1', c2'
end

function system_hamiltonian(p::SystemParams)
    c1, c2, cd1, cd2 = fermion_ops_2site()
    n1 = cd1 * c1
    n2 = cd2 * c2
    # Match main.tex convention: H_S = -t(c1†c2 + c2†c1) - (omega/2)(n1 - n2)
    return -p.t * (cd1 * c2 + cd2 * c1) - 0.5 * p.omega * (n1 - n2)
end

function mixing_angles(p::SystemParams)
    r = sqrt(p.t^2 + (p.omega / 2)^2)
    den = sqrt((r + p.omega / 2)^2 + p.t^2)
    c = (r + p.omega / 2) / den
    s = p.t / den
    return r, c, s
end

function normal_modes(p::SystemParams)
    c1, c2, _, _ = fermion_ops_2site()
    _, c, s = mixing_angles(p)
    # Match main.tex rotation: [c+; c-] = [c -s; -s -c] [c1; c2]
    cplus = c * c1 - s * c2
    cminus = -s * c1 - c * c2
    return cplus, cminus
end

function local_to_eigen_coeff(site::Int, c::Float64, s::Float64)
    if site == 1
        return c, -s
    elseif site == 2
        return -s, -c
    else
        error("site must be 1 or 2")
    end
end

"""
Decompose Hermitian system operators (eqs. 62-64, 87-90) into Bohr frequencies ±r.
Returns Dict((p, sign) => c_p[sign*r]), p in (:alpha, :beta), sign in (+1, -1).
"""
function coupling_ops_by_frequency(p::SystemParams, site::Int)
    cplus, cminus = normal_modes(p)
    _, c, s = mixing_angles(p)
    a, b = local_to_eigen_coeff(site, c, s)

    # For ci = a c+ + b c-, positive-frequency part carries c+ and c-†.
    # c_alpha = i(ci - ci†)/√2, c_beta = (ci† + ci)/√2.
    c_alpha_plus = im * (a * cplus - b * cminus') / sqrt(2)
    c_beta_plus = (a * cplus + b * cminus') / sqrt(2)

    ops = Dict{Tuple{Symbol, Int}, Matrix{C64}}()
    ops[(:alpha, +1)] = c_alpha_plus
    ops[(:alpha, -1)] = c_alpha_plus'
    ops[(:beta, +1)] = c_beta_plus
    ops[(:beta, -1)] = c_beta_plus'
    return ops
end

# Vectorization helpers: vec(A ρ B) = (B^T ⊗ A) vec(ρ)
function spre(A::AbstractMatrix{C64}, d::Int)
    return kron(Matrix{C64}(I, d, d), A)
end

function spost(B::AbstractMatrix{C64}, d::Int)
    return kron(transpose(B), Matrix{C64}(I, d, d))
end

function commutator_super(H::Matrix{C64})
    d = size(H, 1)
    return -im * (spre(H, d) - spost(H, d))
end

function lindblad_super(J::Matrix{C64})
    d = size(J, 1)
    JdJ = J' * J
    return spost(J', d) * spre(J, d) - 0.5 * spre(JdJ, d) - 0.5 * spost(JdJ, d)
end

"""
Bath correlation Φ_pq(ω) from eqs. (72)-(73).
Sign convention for off-diagonal part follows Φ_{βα} = -Φ_{αβ}.
"""
function Phi_from_sigma(b::BathSpectrum, p::Symbol, q::Symbol, w::Float64)
    s1 = b.Sigma(w)
    s2 = conj(b.SigmaTilde(-w))
    if p == q
        # eqs. (70),(72): Φ_{αα}=Φ_{ββ}= (i/2)(Σ - \tilde{Σ}^\dagger)
        return 0.5im * (s1 - s2)
    elseif p == :alpha && q == :beta
        # eqs. (66),(73): Φ_{αβ}= (1/2)(Σ + \tilde{Σ}^\dagger)
        return 0.5 * (s1 + s2)
    elseif p == :beta && q == :alpha
        return -0.5 * (s1 + s2)
    else
        error("p,q must be :alpha or :beta")
    end
end

"""
Compute gamma_pq(ω) and eta_pq(ω) from Φ_pq(ω).
Uses relations: γ_pq = Φ_pq + Φ_qp* , η_pq = -(i/2)(Φ_pq - Φ_qp*).
"""
function gamma_eta_from_phi(b::BathSpectrum, p::Symbol, q::Symbol, w::Float64)
    phi_pq = Phi_from_sigma(b, p, q, w)
    phi_qp = Phi_from_sigma(b, q, p, w)
    gamma = phi_pq + conj(phi_qp)
    eta = -0.5im * (phi_pq - conj(phi_qp))
    return gamma, eta
end

"""
Redfield Liouvillian (eqs. 94-95) for the two-bath model.
"""
function redfield_liouvillian(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    L = commutator_super(H)
    pq = (:alpha, :beta)

    for (site, bath) in ((1, m.bath1), (2, m.bath2))
        ops = coupling_ops_by_frequency(m.sys, site)
        for p in pq, q in pq, sgn_w in (+1, -1), sgn_wp in (+1, -1)
            w = sgn_w * r
            wp = sgn_wp * r
            Ap = ops[(p, sgn_w)]   # c_p[w]
            Aq = ops[(q, sgn_wp)]  # c_q[w']
            pref = (p == q) ? -1.0 : 1.0  # (1 - 2δ_pq)

            phi_pq = Phi_from_sigma(bath, p, q, wp)
            phi_qp = Phi_from_sigma(bath, q, p, w)

            # Φ_pq(ω') [c_p†(ω), c_q(ω') ρ]
            t1 = phi_pq * (spre(Ap' * Aq, d) - spost(Ap', d) * spre(Aq, d))

            # - Φ_qp*(ω) [c_q(ω'), ρ c_p†(ω)]
            t2 = -conj(phi_qp) * (spost(Ap', d) * spre(Aq, d) - spost(Ap' * Aq, d))

            L += pref * (t1 + t2)
        end
    end
    return L
end

"""
γ-matrix square root (eqs. 84-86, 100) at frequency w.
"""
function gammahat_sqrt(b::BathSpectrum, w::Float64)
    sw = b.Sigma(w)
    st = conj(b.SigmaTilde(-w))

    gamma = imag(sw + st)     # eq. (82)
    gamma_p = imag(sw - st)   # eq. (83)

    gamma_m = gamma + gamma_p
    gamma_s = gamma - gamma_p

    gm = sqrt(complex(gamma_m))
    gs = sqrt(complex(gamma_s))

    return (1 / sqrt(2)) * C64[
        (gm + gs)         im * (gm - gs)
        -im * (gm - gs)   (gm + gs)
    ]
end

"""
Nathan-Rudner type GKSL Liouvillian (eqs. 100-109).
"""
function nr_gksl_liouvillian(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    r, _, _ = mixing_angles(m.sys)
    Hls_total = zeros(C64, d, d)
    pset = (:alpha, :beta)
    pidx = Dict(:alpha => 1, :beta => 2)
    lfac = Dict(:alpha => -im, :beta => im) # l_{α/β} = ∓ i

    # Accumulate Lamb shift from each bath
    for (site, bath) in ((1, m.bath1), (2, m.bath2))
        ops = coupling_ops_by_frequency(m.sys, site)
        Hls = zeros(C64, d, d)
        for p in pset, q in pset, sgn_w in (+1, -1), sgn_wp in (+1, -1)
            w = sgn_w * r
            wp = sgn_wp * r
            Ap = ops[(p, sgn_w)]   # c_p[w]
            Aq = ops[(q, sgn_wp)]  # c_q[w']
            pref = (p == q) ? -1.0 : 1.0  # (1 - 2δ_pq)

            gamma_w, eta_w = gamma_eta_from_phi(bath, p, q, w)
            gamma_wp, eta_wp = gamma_eta_from_phi(bath, p, q, wp)

            coeff = 0.5 * (eta_w + eta_wp) + 0.25im * (gamma_w - gamma_wp)
            Hls += pref * coeff * (Ap' * Aq)
        end
        Hls = 0.5 * (Hls + Hls')
        Hls_total += Hls
    end

    L = commutator_super(H + Hls_total)

    # Dissipator (Nathan-Rudner GKSL)
    for (site, bath) in ((1, m.bath1), (2, m.bath2))
        ops = coupling_ops_by_frequency(m.sys, site)
        jumps = Matrix{C64}[]
        for p in pset
            J = zeros(C64, d, d)
            for sgn in (+1, -1)
                sqg = gammahat_sqrt(bath, sgn * r)
                for q in pset
                    J += sqg[pidx[p], pidx[q]] * lfac[q] * ops[(q, sgn)]
                end
            end
            push!(jumps, J)
        end
        for J in jumps
            L += lindblad_super(J)
        end
    end
    return L
end

"""
Return jump operators for each bath/site in the Nathan-Rudner GKSL construction.
"""
function nr_jumps_by_site(m::TwoBathModel)
    r, _, _ = mixing_angles(m.sys)
    pset = (:alpha, :beta)
    pidx = Dict(:alpha => 1, :beta => 2)
    lfac = Dict(:alpha => -im, :beta => im) # l_{α/β} = ∓ i

    jumps_by_site = Dict{Int, Vector{Matrix{C64}}}()
    for (site, bath) in ((1, m.bath1), (2, m.bath2))
        ops = coupling_ops_by_frequency(m.sys, site)
        jumps = Matrix{C64}[]
        for p in pset
            J = zeros(C64, size(ops[(p, +1)]))
            for sgn in (+1, -1)
                sqg = gammahat_sqrt(bath, sgn * r)
                for q in pset
                    J += sqg[pidx[p], pidx[q]] * lfac[q] * ops[(q, sgn)]
                end
            end
            push!(jumps, J)
        end
        jumps_by_site[site] = jumps
    end
    return jumps_by_site
end

"""
Compute particle gain/loss rates for each site using GKSL jump operators.
Returns a Dict(site => (gain, loss)) for a given density matrix rho.
"""
function hs_coeffs(op::AbstractMatrix{C64}, basis::Dict{Symbol, AbstractMatrix{C64}})
    coeffs = Dict{Symbol, C64}()
    for (k, B) in basis
        denom = tr(B' * B)
        coeffs[k] = denom == 0 ? 0 + 0im : tr(B' * op) / denom
    end
    return coeffs
end

function redfield_gain_loss_local(
    bath::BathSpectrum,
    ops,
    r::Float64,
    rho::Matrix{C64},
    site::Int,
    basis::Dict{Symbol, AbstractMatrix{C64}}
)
    pq = (:alpha, :beta)
    c_key = site == 1 ? :c1 : :c2
    cd_key = site == 1 ? :cd1 : :cd2
    n = basis[cd_key] * basis[c_key]

    gain = 0.0
    loss = 0.0

    for p in pq, q in pq, sgn_w in (+1, -1), sgn_wp in (+1, -1)
        w = sgn_w * r
        wp = sgn_wp * r
        Ap = ops[(p, sgn_w)]   # c_p[w]
        Aq = ops[(q, sgn_wp)]  # c_q[w']
        pref = (p == q) ? -1.0 : 1.0  # (1 - 2δ_pq)

        phi_pq = Phi_from_sigma(bath, p, q, wp)
        phi_qp = Phi_from_sigma(bath, q, p, w)

        # Sandwich terms only:
        # -phi_pq [Ap' ρ Aq]  and  -phi_qp* [Aq ρ Ap']
        for (X, Y, coeff) in ((Ap', Aq, -phi_pq), (Aq, Ap', -conj(phi_qp)))
            coeffs_X = hs_coeffs(X, basis)
            coeffs_Y = hs_coeffs(Y, basis)

            cg = coeffs_X[cd_key] * coeffs_Y[c_key]  # c† ρ c
            cl = coeffs_X[c_key] * coeffs_Y[cd_key]  # c ρ c†

            if cg != 0
                gain += real(pref * coeff * cg * tr(n * (basis[cd_key] * rho * basis[c_key])))
            end
            if cl != 0
                loss += real(pref * coeff * cl * tr(n * (basis[c_key] * rho * basis[cd_key])))
            end
        end
    end

    return gain, loss
end

"""
Compute particle gain/loss and net flow for each site.
method = :gksl uses NR-GKSL jumps; method = :redfield uses Redfield superoperator split.
Returns Dict(site => (gain, loss, net)).
"""
function site_gain_loss(m::TwoBathModel, rho::Matrix{C64}; method::Symbol = :gksl)
    c1, c2, cd1, cd2 = fermion_ops_2site()
    nops = Dict(1 => cd1 * c1, 2 => cd2 * c2)
    results = Dict{Int, Tuple{Float64, Float64, Float64}}()

    if method == :gksl
        jumps_by_site = nr_jumps_by_site(m)
        for site in (1, 2)
            n = nops[site]
            gain = 0.0
            loss = 0.0
            for J in jumps_by_site[site]
                JdJ = J' * J
                gain += real(tr(n * (J * rho * J')))
                loss += real(tr(n * (0.5 * (JdJ * rho + rho * JdJ))))
            end
            results[site] = (gain, loss, gain - loss)
        end
        return results
    elseif method == :redfield
        r, _, _ = mixing_angles(m.sys)
        basis = Dict(:c1 => c1, :cd1 => cd1, :c2 => c2, :cd2 => cd2)
        for (site, bath) in ((1, m.bath1), (2, m.bath2))
            ops = coupling_ops_by_frequency(m.sys, site)
            gain, loss = redfield_gain_loss_local(bath, ops, r, rho, site, basis)
            results[site] = (gain, loss, gain - loss)
        end
        return results
    else
        error("method must be :gksl or :redfield")
    end
end

function steady_state_from_liouvillian(L::Matrix{C64}; d::Int)
    I_d = Matrix{C64}(I, d, d)
    tr_row = reshape(vec(transpose(I_d)), 1, :)
    A = [L; tr_row]
    b = zeros(C64, size(L, 1) + 1)
    b[end] = 1 + 0im
    rho_vec = A \ b
    rho = reshape(rho_vec, d, d)
    rho = 0.5 * (rho + rho')
    rho ./= real(tr(rho))
    return rho
end

function steady_states(m::TwoBathModel)
    H = system_hamiltonian(m.sys)
    d = size(H, 1)
    Lr = redfield_liouvillian(m)
    Lg = nr_gksl_liouvillian(m)
    return steady_state_from_liouvillian(Lr; d), steady_state_from_liouvillian(Lg; d), H
end

function fermi(e::Float64, mu::Float64, beta::Float64)
    return 1.0 / (exp(beta * (e - mu)) + 1.0)
end

function test_c1_on_11(; verbose::Bool = true)
    c1, c2, _, _ = fermion_ops_2site()
    v11 = C64[0, 0, 0, 1]
    v01 = C64[0, 1, 0, 0]

    got_c1_11 = c1 * v11
    exp_c1_11 = C64[0, 1, 0, 0]   # c1|11> = |01>
    ok1 = isapprox(got_c1_11, exp_c1_11; atol = 1e-12, rtol = 0.0)

    got_c1_01 = c1 * v01
    exp_c1_01 = C64[0, 0, 0, 0]   # c1|01> = 0
    ok2 = isapprox(got_c1_01, exp_c1_01; atol = 1e-12, rtol = 0.0)

    got_c2_11 = c2 * v11
    exp_c2_11 = C64[0, 0, -1, 0]  # c2|11> = -|10> (Jordan-Wigner sign)
    ok3 = isapprox(got_c2_11, exp_c2_11; atol = 1e-12, rtol = 0.0)

    ok = ok1 && ok2 && ok3
    if verbose
        println("basis order = [|00>, |01>, |10>, |11>]")
        println("input |11> vector: ", v11)
        println("input |01> vector: ", v01)
        println()
        println("c1 * |11> = ", got_c1_11, " ; expected = ", exp_c1_11, " ; pass = ", ok1)
        println("c1 * |01> = ", got_c1_01, " ; expected = ", exp_c1_01, " ; pass = ", ok2)
        println("c2 * |11> = ", got_c2_11, " ; expected = ", exp_c2_11, " ; pass = ", ok3)
        println("all tests pass = ", ok)
    end
    return ok, (got_c1_11, got_c1_01, got_c2_11), (exp_c1_11, exp_c1_01, exp_c2_11)
end

function random_density_matrix(d::Int; seed::Int = 1234)
    rng = MersenneTwister(seed)
    x = randn(rng, C64, d, d)
    rho = x * x'
    rho ./= real(tr(rho))
    return rho
end

function test_redfield_trace_preservation(; dt::Float64 = 0.05, nsteps::Int = 40, verbose::Bool = true)
    sys = SystemParams(1.0, 0.8)
    bathL = make_bath(Gamma = 0.15, mu = 0.6, beta = 8.0, tau_c = 0.2)
    bathR = make_bath(Gamma = 0.15, mu = -0.6, beta = 8.0, tau_c = 0.2)
    model = TwoBathModel(sys, bathL, bathR)

    L = redfield_liouvillian(model)
    d = 4
    rho0 = random_density_matrix(d)
    rho_vec = vec(rho0)

    # Check generator-level trace conservation: Tr(Lρ) = 0
    drho0 = reshape(L * rho_vec, d, d)
    trdot0 = tr(drho0)

    # Check finite-time propagator E = exp(dt * L)
    E = exp(dt * L)
    max_trace_dev = 0.0
    last_trace = tr(rho0)
    for _ in 1:nsteps
        rho_vec = E * rho_vec
        rho = reshape(rho_vec, d, d)
        last_trace = tr(rho)
        max_trace_dev = max(max_trace_dev, abs(last_trace - 1))
    end

    ok = isapprox(trdot0, 0 + 0im; atol = 1e-10, rtol = 0.0) &&
         isapprox(last_trace, 1 + 0im; atol = 1e-10, rtol = 0.0) &&
         max_trace_dev < 1e-9

    if verbose
        println("Redfield trace-preservation test")
        println("dt = ", dt, ", nsteps = ", nsteps)
        println("Tr(rho0) = ", tr(rho0))
        println("Tr(L*rho0) = ", trdot0)
        println("Tr(rho_final) = ", last_trace)
        println("max |Tr(rho_t)-1| = ", max_trace_dev)
        println("test pass = ", ok)
    end
    return ok, trdot0, last_trace, max_trace_dev
end

function thermal_grand_canonical_state(H::Matrix{C64}, Nop::Matrix{C64}, beta::Float64, mu::Float64)
    rho = exp(-beta * (H - mu * Nop))
    rho ./= tr(rho)
    return rho
end

function mode_occupations(rho::Matrix{C64}, p::SystemParams)
    cplus, cminus = normal_modes(p)
    nplus = real(tr((cplus' * cplus) * rho))
    nminus = real(tr((cminus' * cminus) * rho))
    return nplus, nminus
end

function current_operator(p::SystemParams)
    c1, c2, cd1, cd2 = fermion_ops_2site()
    return im * p.t * (cd1 * c2 - cd2 * c1)
end

function current_expectation(rho::Matrix{C64}, p::SystemParams)
    Jop = current_operator(p)
    return real(tr(Jop * rho))
end

function test_equilibrium_fermi_distribution(;
    omega::Float64 = 0.8,
    mu::Float64 = 0.0,
    beta::Float64 = 8.0,
    Gamma::Float64 = 0.15,
    tau_c::Float64 = 0.2,
    occ_tol::Float64 = 5e-2,
    rho_tol::Float64 = 1e-1,
    verbose::Bool = true
)
    sys = SystemParams(1.0, omega)
    bathL = make_bath(Gamma = Gamma, mu = mu, beta = beta, tau_c = tau_c)
    bathR = make_bath(Gamma = Gamma, mu = mu, beta = beta, tau_c = tau_c)
    model = TwoBathModel(sys, bathL, bathR)

    rho_red, rho_gksl, H = steady_states(model)
    c1, c2, cd1, cd2 = fermion_ops_2site()
    Nop = cd1 * c1 + cd2 * c2
    rho_th = thermal_grand_canonical_state(H, Nop, beta, mu)

    r, _, _ = mixing_angles(sys)
    f_plus = fermi(r, mu, beta)
    f_minus = fermi(-r, mu, beta)

    npr, nmr = mode_occupations(rho_red, sys)
    npg, nmg = mode_occupations(rho_gksl, sys)

    occ_err_red = max(abs(npr - f_plus), abs(nmr - f_minus))
    occ_err_gksl = max(abs(npg - f_plus), abs(nmg - f_minus))

    rho_err_red = opnorm(rho_red - rho_th, 1)
    rho_err_gksl = opnorm(rho_gksl - rho_th, 1)

    ok_red = occ_err_red < occ_tol && rho_err_red < rho_tol
    ok_gksl = occ_err_gksl < occ_tol && rho_err_gksl < rho_tol
    ok = ok_red && ok_gksl

    if verbose
        println("Equilibrium Fermi-distribution test (omega=$(omega), mu_L=mu_R=$(mu), beta=$(beta))")
        println("target Fermi occupations: n+(r) = $(f_plus), n-(-r) = $(f_minus)")
        println("Redfield occupations: n+ = $(npr), n- = $(nmr), max occ error = $(occ_err_red)")
        println("NR-GKSL occupations: n+ = $(npg), n- = $(nmg), max occ error = $(occ_err_gksl)")
        println("||rho_red - rho_th||_1  = $(rho_err_red)")
        println("||rho_gksl - rho_th||_1 = $(rho_err_gksl)")
        println("thresholds: occ_tol=$(occ_tol), rho_tol=$(rho_tol)")
        println("pass Redfield = $(ok_red), pass GKSL = $(ok_gksl), all pass = $(ok)")
    end

    return ok, occ_err_red, occ_err_gksl, rho_err_red, rho_err_gksl
end

function test_current_zero_and_finite(;
    beta::Float64 = 8.0,
    Gamma::Float64 = 0.15,
    tau_c::Float64 = 0.2,
    mu_eq::Float64 = 0.0,
    muL_bias::Float64 = 0.6,
    muR_bias::Float64 = -0.6,
    zero_tol::Float64 = 1e-10,
    finite_tol::Float64 = 1e-4,
    verbose::Bool = true
)
    # Case 1: equilibrium (omega=0, muL=muR=0) -> current should vanish.
    sys_eq = SystemParams(1.0, 0.0)
    bathL_eq = make_bath(Gamma = Gamma, mu = mu_eq, beta = beta, tau_c = tau_c)
    bathR_eq = make_bath(Gamma = Gamma, mu = mu_eq, beta = beta, tau_c = tau_c)
    model_eq = TwoBathModel(sys_eq, bathL_eq, bathR_eq)
    rho_red_eq, _, _ = steady_states(model_eq)
    j_eq = current_expectation(rho_red_eq, sys_eq)
    ok_zero = abs(j_eq) < zero_tol

    # Case 2: biased chemical potentials (Redfield) -> finite current.
    sys_bias = SystemParams(1.0, 0.0)
    bathL_bias = make_bath(Gamma = Gamma, mu = muL_bias, beta = beta, tau_c = tau_c)
    bathR_bias = make_bath(Gamma = Gamma, mu = muR_bias, beta = beta, tau_c = tau_c)
    model_bias = TwoBathModel(sys_bias, bathL_bias, bathR_bias)
    rho_red_bias, _, _ = steady_states(model_bias)
    j_bias = current_expectation(rho_red_bias, sys_bias)
    ok_finite = abs(j_bias) > finite_tol

    ok = ok_zero && ok_finite
    if verbose
        println("Current test with J = i t (c1†c2 - c2†c1)")
        println("equilibrium case: omega=0, muL=muR=$(mu_eq)")
        println("  Redfield current = $(j_eq), |J|<$(zero_tol) => $(ok_zero)")
        println("biased case: omega=0, muL=$(muL_bias), muR=$(muR_bias)")
        println("  Redfield current = $(j_bias), |J|>$(finite_tol) => $(ok_finite)")
        println("all pass = $(ok)")
    end
    return ok, j_eq, j_bias
end

function test_gain_loss_zero_equilibrium(;
    beta::Float64 = 8.0,
    Gamma::Float64 = 0.15,
    tau_c::Float64 = 0.2,
    mu::Float64 = 0.0,
    omega::Float64 = 0.0,
    tol::Float64 = 1e-10,
    verbose::Bool = true
)
    sys = SystemParams(1.0, omega)
    bathL = make_bath(Gamma = Gamma, mu = mu, beta = beta, tau_c = tau_c)
    bathR = make_bath(Gamma = Gamma, mu = mu, beta = beta, tau_c = tau_c)
    model = TwoBathModel(sys, bathL, bathR)

    rho_red, rho_gksl, _ = steady_states(model)

    gl_red = site_gain_loss(model, rho_red; method = :redfield)
    gl_gksl = site_gain_loss(model, rho_gksl; method = :gksl)

    ok_red = all(abs(gl_red[site][3]) < tol for site in (1, 2))
    ok_gksl = all(abs(gl_gksl[site][3]) < tol for site in (1, 2))
    ok = ok_red && ok_gksl

    if verbose
        println("Gain-Loss net flow test (muL=muR=$(mu), omega=$(omega))")
        println("tolerance = $(tol)")
        println("Redfield net: site1=$(gl_red[1][3]), site2=$(gl_red[2][3]) => pass=$(ok_red)")
        println("GKSL net:    site1=$(gl_gksl[1][3]), site2=$(gl_gksl[2][3]) => pass=$(ok_gksl)")
        println("all pass = $(ok)")
    end

    return ok, gl_red, gl_gksl
end

"""
Example finite-correlation bath self-energy model.
"""
function make_bath(; Gamma::Float64, mu::Float64, beta::Float64, tau_c::Float64)
    Sigma = (w::Float64) -> begin
        f = fermi(w, mu, beta)
        -im * Gamma * (1 - f) / (1 - im * w * tau_c)
    end
    SigmaTilde = (w::Float64) -> begin
        f = fermi(w, mu, beta)
        -im * Gamma * f / (1 - im * w * tau_c)
    end
    return BathSpectrum(Sigma, SigmaTilde)
end

function main()
    sys = SystemParams(1.0, 0.8)
    bathL = make_bath(Gamma = 0.15, mu = 0.6, beta = 8.0, tau_c = 0.2)
    bathR = make_bath(Gamma = 0.15, mu = -0.6, beta = 8.0, tau_c = 0.2)
    model = TwoBathModel(sys, bathL, bathR)

    rho_red, rho_gksl, H = steady_states(model)

    c1, c2, cd1, cd2 = fermion_ops_2site()
    n1 = real(tr((cd1 * c1) * rho_red))
    n2 = real(tr((cd2 * c2) * rho_red))
    n1g = real(tr((cd1 * c1) * rho_gksl))
    n2g = real(tr((cd2 * c2) * rho_gksl))

    println("=== Steady state (Redfield) ===")
    println(rho_red)
    println("occupations: n1 = $(n1), n2 = $(n2)")
    println()
    println("=== Steady state (Nathan-Rudner GKSL) ===")
    println(rho_gksl)
    println("occupations: n1 = $(n1g), n2 = $(n2g)")
    println()
    println("trace check: Redfield=$(tr(rho_red)), GKSL=$(tr(rho_gksl))")
    println("Hermiticity check: Redfield=$(norm(rho_red-rho_red')), GKSL=$(norm(rho_gksl-rho_gksl'))")
    _ = H
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
