function gamma_from_path(path::AbstractString)
    dirname = basename(path)
    startswith(dirname, "Gamma_") || error("Expected directory name starting with Gamma_: $dirname")
    return parse(Float64, dirname[length("Gamma_") + 1:end])
end

function collect_local_observables_over_gamma(rootdir::AbstractString;
                                              filename::AbstractString = "nr_correlator.jld2",
                                              t::Real = 1.0,
                                              format::Symbol = :jld2)
    gamma_dirs = filter(path -> isdir(path) && startswith(basename(path), "Gamma_"),
                        readdir(rootdir; join=true))

    records = NamedTuple[]
    for dir in gamma_dirs
        gamma = gamma_from_path(dir)
        path = joinpath(dir, filename)
        obs = local_observables_from_file(path, t; format=format)
        push!(records, (
            Gamma = gamma,
            beta_eff = obs.beta_eff,
            temp_eff = obs.temp_eff,
            mu_eff = obs.mu_eff,
            current = obs.current,
        ))
    end

    sort!(records; by = x -> x.Gamma)

    return (
        Gamma = [r.Gamma for r in records],
        beta_eff = [r.beta_eff for r in records],
        temp_eff = [r.temp_eff for r in records],
        mu_eff = [r.mu_eff for r in records],
        current = [r.current for r in records],
    )
end
