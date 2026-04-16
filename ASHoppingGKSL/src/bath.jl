#include("ops_space.jl")

struct BathSpectrum
    μ::Float64
    β::Float64
    Sigma::Function      # Σ(ω)
    SigmaTilde::Function # \tilde{Σ}(ω)
end

struct TwoBathModel
    sys::SystemParams
    bath1::BathSpectrum
    bath2::BathSpectrum
end

function fermi(e::Float64, mu::Float64, beta::Float64)
    return 1.0 / (exp(beta * (e - mu)) + 1.0)
end

"""
Example finite-correlation bath self-energy model.
"""
# mu is chemical potential
function make_bath(; Gamma::Float64, mu::Float64, beta::Float64, tau_c::Float64)
    Sigma = (w::Float64) -> begin
        f = fermi(w, mu, beta)
        #-im * Gamma * (1 - f) / (1 - im * w * tau_c)
        -im * Gamma * (1 - f) / (1 - im * (w - mu) * tau_c)
    end
    #TODO:SigmaTildeの符号あってるか確認
    SigmaTilde = (w::Float64) -> begin
        f = fermi(w, mu, beta)
        #-im * Gamma * f / (1 - im * w * tau_c)
        -im * Gamma * f / (1 - im * (w - mu) * tau_c)
    end
    return BathSpectrum(mu, beta, Sigma, SigmaTilde)
end

function make_bath_flat(; Gamma::Float64, mu::Float64, beta::Float64, tau_c::Float64)
    Sigma = (w::Float64) -> begin
        f = fermi(w, mu, beta)
        #-im * Gamma * (1 - f) / (1 - im * w * tau_c)
        -im * Gamma * (1 - f)
    end
    #TODO:SigmaTildeの符号あってるか確認
    SigmaTilde = (w::Float64) -> begin
        f = fermi(w, mu, beta)
        #-im * Gamma * f / (1 - im * w * tau_c)
        -im * Gamma * f
    end
    return BathSpectrum(mu, beta, Sigma, SigmaTilde)
end

# mu is energy potential
function make_bath2(; Gamma::Float64, mu::Float64, beta::Float64, tau_c::Float64)
    Sigma = (w::Float64) -> begin
        f = fermi(w, 0.0, beta)
        -im * Gamma * (1 - f) / (1 - im * (w - mu) * tau_c)
    end
    SigmaTilde = (w::Float64) -> begin
        f = fermi(w, 0.0, beta)
        -im * Gamma * f / (1 - im * (w - mu) * tau_c)
    end
    return BathSpectrum(mu, beta, Sigma, SigmaTilde)
end