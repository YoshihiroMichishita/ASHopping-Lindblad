function central_block(C::AbstractMatrix{<:Complex})
    n = size(C, 1)
    n == size(C, 2) || error("central_block requires a square matrix")
    iseven(n) || error("central_block requires an even number of sites")
    left = n ÷ 2
    right = left + 1
    return Matrix{ComplexF64}(C[left:right, left:right]), (left, right)
end

function local_current_from_Cmid(Cmid::AbstractMatrix{<:Complex}, t::Real)
    size(Cmid) == (2, 2) || error("local_current_from_Cmid requires a 2x2 matrix")
    return 2 * t * imag(Cmid[1, 2])
end

function local_beta_mu_from_Cmid(Cmid::AbstractMatrix{<:Complex}, t::Real; eps::Float64 = 1e-12)
    size(Cmid) == (2, 2) || error("local_beta_mu_from_Cmid requires a 2x2 matrix")
    abs(t) > eps || error("local_beta_mu_from_Cmid requires nonzero hopping t")

    d1 = real(Cmid[1, 1])
    d2 = real(Cmid[2, 2])
    r = real(Cmid[1, 2])
    yim = imag(Cmid[1, 2])

    d = 0.5 * (d1 + d2)
    n_minus = clamp(d + r, eps, 1 - eps)
    n_plus = clamp(d - r, eps, 1 - eps)

    a = log((1 - n_plus) / n_plus)
    b = log((1 - n_minus) / n_minus)
    beta_eff = (a - b) / (2 * t)
    abs(beta_eff) > eps || error("local_beta_mu_from_Cmid produced near-zero beta_eff")
    mu_eff = -(a + b) / (2 * beta_eff)

    return (
        beta_eff = beta_eff,
        temp_eff = 1 / beta_eff,
        mu_eff = mu_eff,
        n_minus = n_minus,
        n_plus = n_plus,
        residual_diag = abs(d1 - d2),
        residual_imag = abs(yim),
    )
end

function local_beta_from_real_offdiag(Cmid::AbstractMatrix{<:Complex}, t::Real; eps::Float64 = 1e-12)
    size(Cmid) == (2, 2) || error("local_beta_from_real_offdiag requires a 2x2 matrix")
    abs(t) > eps || error("local_beta_from_real_offdiag requires nonzero hopping t")

    x = clamp(2 * real(Cmid[1, 2]), -1 + eps, 1 - eps)
    beta_eff = (2 / t) * atanh(x)
    return (
        beta_eff = beta_eff,
        temp_eff = 1 / beta_eff,
    )
end

function local_energy_temperature_from_Cmid(Cmid::AbstractMatrix{<:Complex}, t::Real; eps::Float64 = 1e-12)
    size(Cmid) == (2, 2) || error("local_energy_temperature_from_Cmid requires a 2x2 matrix")
    abs(t) > eps || error("local_energy_temperature_from_Cmid requires nonzero hopping t")

    d1 = real(Cmid[1, 1])
    d2 = real(Cmid[2, 2])
    yim = imag(Cmid[1, 2])
    energy_loc = -2 * t * real(Cmid[1, 2])
    number_loc = d1 + d2

    n_minus_raw = 0.5 * (number_loc - energy_loc / t)
    n_plus_raw = 0.5 * (number_loc + energy_loc / t)
    n_minus = clamp(n_minus_raw, eps, 1 - eps)
    n_plus = clamp(n_plus_raw, eps, 1 - eps)

    a = log((1 - n_plus) / n_plus)
    b = log((1 - n_minus) / n_minus)
    beta_eff = (a - b) / (2 * t)
    abs(beta_eff) > eps || error("local_energy_temperature_from_Cmid produced near-zero beta_eff")
    mu_eff = -(a + b) / (2 * beta_eff)

    return (
        beta_eff = beta_eff,
        temp_eff = 1 / beta_eff,
        mu_eff = mu_eff,
        energy_loc = energy_loc,
        number_loc = number_loc,
        n_minus_energy = n_minus,
        n_plus_energy = n_plus,
        residual_diag = abs(d1 - d2),
        residual_imag = abs(yim),
        occupation_clamp_error = max(abs(n_minus - n_minus_raw), abs(n_plus - n_plus_raw)),
    )
end
