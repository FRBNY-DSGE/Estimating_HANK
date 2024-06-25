"""
```
measurement(m::HetDSGELag{T}, TTT::Matrix{T}, RRR::Matrix{T},
            CCC::Vector{T}) where {T<:AbstractFloat}
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
function measurement(m::HetDSGELag{T},
                     TTT::Matrix{T},
                     RRR::Matrix{T},
                     CCC::Vector{T}) where {T<:AbstractFloat}
    endo     = m.endogenous_states
    endo_unnorm = m.endogenous_states_unnormalized
    exo      = m.exogenous_shocks
    obs      = m.observables

    _n_model_states = n_model_states(m)
    _n_states = n_backward_looking_states(m)
    _n_jumps = n_jumps(m)

    _n_observables = n_observables(m)
    _n_shocks_exogenous = n_shocks_exogenous(m)

    ZZ = zeros(_n_observables, _n_model_states)
    DD = zeros(_n_observables)
    EE = zeros(_n_observables, _n_observables)
    QQ = zeros(_n_shocks_exogenous, _n_shocks_exogenous)

    ## GDP growth - Quarterly!
    ZZ[obs[:obs_gdp], endo[:y′_t]]          = 1.0
    ZZ[obs[:obs_gdp], endo[:y′_t1]]     = -1.0
    ZZ[obs[:obs_gdp], endo[:z′_t]]          = 1.0
    #ZZ[obs[:obs_gdp], endo_new[:e_gdp_t]]  = 1.0
    #ZZ[obs[:obs_gdp], endo_new[:e_gdp_t1]] = -m[:me_level]
    DD[obs[:obs_gdp]]                      = 100*(exp(m[:γ])-1) #100*(exp(m[:zstar])-1)

    ## GDI growth- Quarterly!
    #=ZZ[obs[:obs_gdi], endo[:y_t]]          = m[:γ_gdi]
    ZZ[obs[:obs_gdi], endo_new[:y_t1]]     = -m[:γ_gdi]
    ZZ[obs[:obs_gdi], endo[:z_t]]          = m[:γ_gdi]
    ZZ[obs[:obs_gdi], endo_new[:e_gdi_t]]  = 1.0
    ZZ[obs[:obs_gdi], endo_new[:e_gdi_t1]] = -m[:me_level]
    DD[obs[:obs_gdi]]                      = 100*(exp(m[:z_star])-1) + m[:δ_gdi]

    ## Hours growth
    ZZ[obs[:obs_hours], endo[:L_t]] = 1.0
    DD[obs[:obs_hours]]             = m[:Lmean]

    ## Labor Share/real wage growth
    ZZ[obs[:obs_wages], endo[:w_t]]      = 1.0
    ZZ[obs[:obs_wages], endo_new[:w_t1]] = -1.0
    ZZ[obs[:obs_wages], endo[:z_t]]      = 1.0
    DD[obs[:obs_wages]]                  = 100*(exp(m[:z_star])-1)

    ## Inflation (GDP Deflator)
    ZZ[obs[:obs_gdpdeflator], endo[:π_t]]            = m[:Γ_gdpdef]
    ZZ[obs[:obs_gdpdeflator], endo_new[:e_gdpdef_t]] = 1.0
    DD[obs[:obs_gdpdeflator]]                        = 100*(m[:π_star]-1) + m[:δ_gdpdef]

    ## Inflation (Core PCE)
    ZZ[obs[:obs_corepce], endo[:π_t]]             = 1.0
    ZZ[obs[:obs_corepce], endo_new[:e_corepce_t]] = 1.0
    DD[obs[:obs_corepce]]                         = 100*(m[:π_star]-1)

    ## Nominal interest rate
    ZZ[obs[:obs_nominalrate], endo[:R_t]] = 1.0
    DD[obs[:obs_nominalrate]]             = m[:Rstarn]

    ## Consumption Growth
    ZZ[obs[:obs_consumption], endo[:c_t]]      = 1.0
    ZZ[obs[:obs_consumption], endo_new[:c_t1]] = -1.0
    ZZ[obs[:obs_consumption], endo[:z_t]]      = 1.0
    DD[obs[:obs_consumption]]                  = 100*(exp(m[:z_star])-1)

    ## Investment Growth
    ZZ[obs[:obs_investment], endo[:i_t]]      = 1.0
    ZZ[obs[:obs_investment], endo_new[:i_t1]] = -1.0
    ZZ[obs[:obs_investment], endo[:z_t]]      = 1.0
    DD[obs[:obs_investment]]                  = 100*(exp(m[:z_star])-1)

    ## Spreads
    ZZ[obs[:obs_spread], endo[:ERtil_k_t]] = 1.0
    ZZ[obs[:obs_spread], endo[:R_t]]       = -1.0
    DD[obs[:obs_spread]]                   = 100*log(m[:spr])

    ## 10 yrs infl exp

    TTT10                          = (1/40)*((Matrix{Float64}(I, size(TTT, 1), size(TTT,1))
                                              - TTT)\(TTT - TTT^41))
    ZZ[obs[:obs_longinflation], :] = TTT10[endo[:π_t], :]
    DD[obs[:obs_longinflation]]    = 100*(m[:π_star]-1)

    ## Long Rate
    ZZ[obs[:obs_longrate], :]               = ZZ[6, :]' * TTT10
    ZZ[obs[:obs_longrate], endo_new[:lr_t]] = 1.0
    DD[obs[:obs_longrate]]                  = m[:Rstarn]

    ## TFP
    ZZ[obs[:obs_tfp], endo[:z_t]]       = (1-m[:α])*m[:Iendoα] + 1*(1-m[:Iendoα])
    ZZ[obs[:obs_tfp], endo_new[:tfp_t]] = 1.0
    ZZ[obs[:obs_tfp], endo[:u_t]]       = m[:α]/( (1-m[:α])*(1-m[:Iendoα]) + 1*m[:Iendoα] )
    ZZ[obs[:obs_tfp], endo_new[:u_t1]]  = -(m[:α]/( (1-m[:α])*(1-m[:Iendoα]) + 1*m[:Iendoα]) ) =#

    QQ[exo[:g_sh], exo[:g_sh]]            = m[:σ_g]^2
    QQ[exo[:b_sh], exo[:b_sh]]            = m[:σ_b]^2
    QQ[exo[:μ_sh], exo[:μ_sh]]            = m[:σ_μ]^2
    QQ[exo[:z_sh], exo[:z_sh]]            = m[:σ_z]^2
    QQ[exo[:λ_f_sh], exo[:λ_f_sh]]        = m[:σ_λ_f]^2
    QQ[exo[:λ_w_sh], exo[:λ_w_sh]]        = m[:σ_λ_w]^2
    QQ[exo[:MON_sh], exo[:MON_sh]]        = m[:σ_MON]^2

    # These lines set the standard deviations for the anticipated shocks
   #= for i = 1:n_anticipated_shocks(m)
        ZZ[obs[Symbol("obs_nominalrate$i")], :] = ZZ[obs[:obs_nominalrate], :]' * (TTT^i)
        DD[obs[Symbol("obs_nominalrate$i")]]    = m[:Rstarn]
        if subspec(m) == "ss11"
            QQ[exo[Symbol("rm_shl$i")], exo[Symbol("rm_shl$i")]] = m[:σ_r_m]^2 / n_anticipated_shocks(m)
        else
            QQ[exo[Symbol("rm_shl$i")], exo[Symbol("rm_shl$i")]] = m[Symbol("σ_r_m$i")]^2
        end
    end =#

    # Adjustment to DD because measurement equation assumes CCC is the zero vector
    if any(CCC .!= 0)
        DD += ZZ*((UniformScaling(1) - TTT)\CCC)
    end

    return Measurement(ZZ, DD, QQ, EE)
end
