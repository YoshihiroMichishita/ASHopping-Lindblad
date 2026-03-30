abstract type BoundaryCondition end

struct PBC <: BoundaryCondition end

struct OBC <: BoundaryCondition end

struct ModelParams
    L::Int
    t::Float64
    J::Float64
    alpha::Float64
    beta::Float64
    mu::Float64
    gamma_plus::Float64
    gamma_minus::Float64
end

function ModelParams(
    L::Int,
    t::Real,
    J::Real,
    alpha::Real,
    beta::Real;
    mu::Real = 0.0,
    gamma_plus::Union{Nothing, Real} = nothing,
    gamma_minus::Union{Nothing, Real} = nothing,
)
    t_a = float(t) * sinh(float(J))
    gp = gamma_plus === nothing ? max(2 * t_a, 0.0) : float(gamma_plus)
    gm = gamma_minus === nothing ? max(-2 * t_a, 0.0) : float(gamma_minus)
    return ModelParams(
        L,
        float(t),
        float(J),
        float(alpha),
        float(beta),
        float(mu),
        gp,
        gm,
    )
end

struct SteadyStateResult
    boundary::Symbol
    energies::Vector{Float64}
    occupations::Vector{Float64}
    density::Vector{Float64}
    mode_labels::Union{Nothing, Vector{Float64}}
    basis_transform::Matrix{ComplexF64}
end
