abstract type BoundaryCondition end

struct PBC <: BoundaryCondition end

struct OBC <: BoundaryCondition end

struct ModelParams
    L::Int
    t::Float64
    J::Float64
end

function ModelParams(L::Int, t::Real, J::Real)
    return ModelParams(L, float(t), float(J))
end

struct SteadyStateResult
    boundary::Symbol
    correlation::Matrix{ComplexF64}
    hamiltonian::Matrix{ComplexF64}
    gamma_gain::Matrix{ComplexF64}
    gamma_loss::Matrix{ComplexF64}
    density::Vector{Float64}
    currents::Vector{Float64}
    residual::Float64
end
