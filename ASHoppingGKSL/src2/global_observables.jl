function global_effective_temperature(C::AbstractMatrix{<:Complex},
                                      sys::gksl.SystemParams;
                                      fix_mu_zero::Bool = true,
                                      eps::Float64 = 1e-12)
    fix_mu_zero || error("global_effective_temperature currently supports only fix_mu_zero=true")

    eps_modes, U = gksl.single_particle_eigensystem(sys)
    Cmode = ComplexF64.(U' * C * U)
    occupations = clamp.(real.(diag(Cmode)), eps, 1 - eps)
    offdiag_norm = norm(Cmode - Diagonal(diag(Cmode)))

    betas = Float64[]
    weights = Float64[]
    for a in eachindex(eps_modes)
        abs(eps_modes[a]) > eps || continue
        push!(betas, log((1 - occupations[a]) / occupations[a]) / eps_modes[a])
        push!(weights, abs(eps_modes[a]))
    end

    isempty(betas) && error("No nonzero mode energies available for global_effective_temperature")
    beta_eff = sum(weights .* betas) / sum(weights)
    fit_occ = 1.0 ./ (exp.(beta_eff .* eps_modes) .+ 1.0)
    fit_residual = norm(occupations - fit_occ)

    return (
        beta_eff = beta_eff,
        temp_eff = 1 / beta_eff,
        mu_eff = 0.0,
        mode_energies = eps_modes,
        mode_occupations = occupations,
        offdiag_norm = offdiag_norm,
        fit_residual = fit_residual,
    )
end
