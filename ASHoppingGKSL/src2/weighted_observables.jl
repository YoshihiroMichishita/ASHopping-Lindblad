function loss_corr_from_C(C::AbstractMatrix{<:Complex})
    return transpose(C)
end

function gain_corr_from_C(C::AbstractMatrix{<:Complex})
    n = size(C, 1)
    return Matrix{ComplexF64}(I, n, n) - Matrix{ComplexF64}(C)
end

function loss_gain_weighted_from_C(loss_coeff::AbstractMatrix{<:Complex},
                                   gain_coeff::AbstractMatrix{<:Complex},
                                   C::AbstractMatrix{<:Complex})
    loss_corr = loss_corr_from_C(C)
    gain_corr = gain_corr_from_C(C)
    return (
        loss_weighted = loss_coeff .* loss_corr,
        gain_weighted = gain_coeff .* gain_corr,
    )
end
