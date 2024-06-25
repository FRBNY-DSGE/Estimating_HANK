function util(c::AbstractArray, γ::Real)
    if γ == 1.0
        util = log.(c)
    else
        util = c .^ (1.0 - γ) ./ (1.0 - γ)
    end
    return util
end

function util(c::AbstractArray, θ::NamedTuple)
    return util(c, θ[:γ])
end
