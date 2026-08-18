function load_correlator_jld2(path::AbstractString)
    return jldopen(path, "r") do file
        if haskey(file, "matrix")
            Matrix{ComplexF64}(file["matrix"])
        elseif haskey(file, "C")
            Matrix{ComplexF64}(file["C"])
        else
            error("JLD2 file must contain \"matrix\" or \"C\"")
        end
    end
end

function load_correlator_csv_pair(real_path::AbstractString, imag_path::AbstractString)
    real_df = CSV.read(real_path, DataFrame)
    imag_df = CSV.read(imag_path, DataFrame)
    names(real_df) == names(imag_df) || error("CSV real/imag columns do not match")
    ncols = ncol(real_df)
    ncols >= 2 || error("CSV correlator must have at least one data column")

    real_mat = Matrix{Float64}(real_df[:, 2:ncols])
    imag_mat = Matrix{Float64}(imag_df[:, 2:ncols])
    return ComplexF64.(real_mat .+ 1im .* imag_mat)
end

function load_correlator(path::AbstractString; format::Symbol = :auto)
    if format == :auto
        if endswith(path, ".jld2")
            return load_correlator_jld2(path)
        elseif endswith(path, ".csv")
            error("For CSV input, specify format=:csv_pair and pass the basename without _real/_imag suffix")
        else
            return load_correlator_csv_pair(path * "_real.csv", path * "_imag.csv")
        end
    elseif format == :jld2
        return load_correlator_jld2(path)
    elseif format == :csv_pair
        return load_correlator_csv_pair(path * "_real.csv", path * "_imag.csv")
    else
        error("format must be :auto, :jld2, or :csv_pair")
    end
end

function local_observables_from_C(C::AbstractMatrix{<:Complex}, t::Real)
    Cmid, inds = central_block(C)
    thermo = local_beta_mu_from_Cmid(Cmid, t)
    energy_thermo = local_energy_temperature_from_Cmid(Cmid, t)
    current = local_current_from_Cmid(Cmid, t)
    return (
        central_indices = inds,
        Cmid = Cmid,
        beta_eff = thermo.beta_eff,
        temp_eff = thermo.temp_eff,
        mu_eff = thermo.mu_eff,
        n_minus = thermo.n_minus,
        n_plus = thermo.n_plus,
        residual_diag = thermo.residual_diag,
        residual_imag = thermo.residual_imag,
        beta_eff_energy = energy_thermo.beta_eff,
        temp_eff_energy = energy_thermo.temp_eff,
        mu_eff_energy = energy_thermo.mu_eff,
        energy_loc = energy_thermo.energy_loc,
        number_loc = energy_thermo.number_loc,
        n_minus_energy = energy_thermo.n_minus_energy,
        n_plus_energy = energy_thermo.n_plus_energy,
        occupation_clamp_error_energy = energy_thermo.occupation_clamp_error,
        current = current,
    )
end

function local_observables_from_file(path::AbstractString, t::Real; format::Symbol = :auto)
    C = load_correlator(path; format=format)
    return local_observables_from_C(C, t)
end
