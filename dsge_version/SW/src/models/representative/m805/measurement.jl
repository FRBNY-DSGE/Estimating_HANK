"""
```
measurement{T<:AbstractFloat}(m::Model805{T}, TTT::Matrix{T}, RRR::Matrix{T},
                              CCC::Vector{T})
```

Assign measurement equation

```
y_t = ZZ*s_t + DD + u_t
```

where

```
Var(ϵ_t) = QQ
Var(u_t) = EE
Cov(ϵ_t, u_t) = 0
```
"""
function measurement(m::Model805{T},
                                       TTT::Matrix{T},
                                       RRR::Matrix{T},
                                       CCC::Vector{T}) where {T<:AbstractFloat}
    endo      = m.endogenous_states
    endo_addl = m.endogenous_states_augmented
    exo       = m.exogenous_shocks
    obs       = m.observables

    _n_observables = n_observables(m)
    _n_states = n_states_augmented(m)
    _n_shocks_exogenous = n_shocks_exogenous(m)

    ZZ = zeros(_n_observables, _n_states)
    DD = zeros(_n_observables)
    EE = zeros(_n_observables, _n_observables)
    QQ = zeros(_n_shocks_exogenous, _n_shocks_exogenous)

    ## Output growth - Quarterly!
    ZZ[obs[:obs_gdp], endo[:y_t]]       = 1.0
    ZZ[obs[:obs_gdp], endo_addl[:y_t1]] = -1.0
    ZZ[obs[:obs_gdp], endo[:z_t]]       = 1.0
    DD[obs[:obs_gdp]]                   = 100*(exp(m[:zstar])-1)

    ## Hours growth
    ZZ[obs[:obs_hours], endo[:L_t]] = 1.0
    DD[obs[:obs_hours]]             = m[:Lmean]

    ## Labor Share/real wage growth
    ZZ[obs[:obs_wages], endo[:w_t]]       = 1.0
    ZZ[obs[:obs_wages], endo_addl[:w_t1]] = -1.0
    ZZ[obs[:obs_wages], endo[:z_t]]       = 1.0
    DD[obs[:obs_wages]]                   = 100*(exp(m[:zstar])-1)

    ## Inflation (GDP Deflator)
    ZZ[obs[:obs_gdpdeflator], endo[:π_t]]  = 1.0
    DD[obs[:obs_gdpdeflator]]              = 100*(m[:π_star]-1)

    ## Nominal interest rate
    ZZ[obs[:obs_nominalrate], endo[:R_t]] = 1.0
    DD[obs[:obs_nominalrate]]             = m[:Rstarn]

    ## Consumption Growth
    ZZ[obs[:obs_consumption], endo[:c_t]]       = 1.0
    ZZ[obs[:obs_consumption], endo_addl[:c_t1]] = -1.0
    ZZ[obs[:obs_consumption], endo[:z_t]]       = 1.0
    DD[obs[:obs_consumption]]                   = 100*(exp(m[:zstar])-1)

    ## Investment Growth
    ZZ[obs[:obs_investment], endo[:i_t]]       = 1.0
    ZZ[obs[:obs_investment], endo_addl[:i_t1]] = -1.0
    ZZ[obs[:obs_investment], endo[:z_t]]       = 1.0
    DD[obs[:obs_investment]]                   = 100*(exp(m[:zstar])-1)

    ## 10 yr Inflation Expectations
    TTT10 = (1/40)*((Matrix{Float64}(I, size(TTT, 1), size(TTT,1)) - TTT) \ (TTT - TTT^41))
    ZZ[obs[:obs_longinflation], :] = TTT10[endo[:π_t], :]
    DD[obs[:obs_longinflation]]    = 100*(m[:π_star] - 1)


    QQ[exo[:g_sh], exo[:g_sh]]            = m[:σ_g]^2
    QQ[exo[:b_sh], exo[:b_sh]]            = m[:σ_b]^2
    QQ[exo[:μ_sh], exo[:μ_sh]]            = m[:σ_μ]^2
    QQ[exo[:z_sh], exo[:z_sh]]            = m[:σ_z]^2
    QQ[exo[:λ_f_sh], exo[:λ_f_sh]]        = m[:σ_λ_f]^2
    QQ[exo[:λ_w_sh], exo[:λ_w_sh]]        = m[:σ_λ_w]^2
    QQ[exo[:rm_sh], exo[:rm_sh]]          = m[:σ_rm]^2
    QQ[exo[:π_star_sh], exo[:π_star_sh]]  = m[:σ_π_star]^2


    # These lines set the standard deviations for the anticipated
    # shocks
    for i = 1:n_anticipated_shocks(m)
        ZZ[obs[Symbol("obs_nominalrate$i")], :] = ZZ[obs[:obs_nominalrate], :]'*(TTT^i)
        DD[obs[Symbol("obs_nominalrate$i")]]    = m[:Rstarn]
        if subspec(m) == "ss1"
            QQ[exo[Symbol("rm_shl$i")], exo[Symbol("rm_shl$i")]] = m[:σ_rm]^2 / n_anticipated_shocks(m)
       else
            QQ[exo[Symbol("rm_shl$i")], exo[Symbol("rm_shl$i")]] = m[:σ_rm]^2
        end
    end

    # Adjustment to DD because measurement equation assumes CCC is the zero vector
    if any(CCC .!= 0)
        DD += ZZ*((UniformScaling(1) - TTT)\CCC)
    end

    return Measurement(ZZ, DD, QQ, EE)
end