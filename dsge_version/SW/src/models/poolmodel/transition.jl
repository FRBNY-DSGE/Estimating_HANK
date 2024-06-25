"""
```
transition(m::PoolModel{T}) where {T<:AbstractFloat}
```

Assign transition equation

```
x_t = (1-ρ) μ + ρ x_{t-1} + sqrt{1 - ρ^2} σ ϵ_t
ϵ_t ∼ iid N(0,1), x_0 ∼ N(μ,σ^2)
λ_t = Φ(x_t)
```
where Φ(⋅) is the cdf of a N(0,1) random variable, F_ϵ is the distribution of ϵ_t,
and F_λ is the distribution of λ(x_0).
"""
function transition(m::PoolModel{T}) where {T<:AbstractFloat}
    transition_necessary = :ρ in [m.parameters[i].key for i in 1:length(m.parameters)]
    ### If false, transition fn never used so arbitrarily chosen.
    ρ::Float64 = transition_necessary ? m[:ρ].value : 0.0
    μ::Float64 = transition_necessary ? m[:μ].value : m[:λ].value
    σ::Float64 = transition_necessary ? m[:σ].value : 0.0

    @inline Φ(x::Vector{Float64}, ϵ::Vector{Float64}) = ((1.0 - ρ) * μ .+ ρ .* x .+
                                                         sqrt(1.0 - ρ^2) * σ .* ϵ)

    @inline Φ(x::Float64, ϵ::Float64) = (1.0 - ρ) * μ + ρ *
        x + sqrt(1 - ρ^2) * σ * ϵ

    F_ϵ = Normal(0.,1.)
    F_λ = Uniform(0.,1.)
    return Φ, F_ϵ, F_λ
end
