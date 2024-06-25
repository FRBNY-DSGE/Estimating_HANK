"""
```
Fsys_agg(F, X, XPrime, θ, grids, id, nt, eq)
```
Return deviations from aggregate equilibrium conditions.

### Inputs
- `F::AbstractVector`: vector holding deviations for aggregate equilibrium conditions
- `X::AbstractArray`,`XPrime::AbstractArray`: deviations from steady state in periods t [`X`] and t+1 [`XPrime`]
- `θ::NamedTuple`: maps parameter name to value
- `grids::OrderedDict`: maps names of quantities related to the idiosyncratic state space to their values (e.g. `m_grid`)
- `id::OrderedDict{Symbol, Int or UnitRange{Int}}`: maps variable name (e.g. `Y_t` and `Y′_t`)
    to its index/indices in `X` and `XPrime`.
- `nt::NamedTuple`: maps variable name (e.g. `Y_t` and `Y′_t`) to steady-state log level
- `eq::OrderedDict{Symbol, Int or UnitRange{Int}}`: maps name of equilibrium conditions
    to its index/indices in `F`
"""
function Fsys_agg(F::AbstractVector, X::AbstractArray, XPrime::AbstractArray,
                  θ::NamedTuple, grids::OrderedDict, id::OrderedDict{Symbol, I1}, nt::NamedTuple,
                  eq::OrderedDict{Symbol, I2}) where {I1 <: Union{Int, UnitRange}, I2 <: Union{Int, UnitRange}}

    ############################################################################
    #            I. Read out argument values                                   #
    ############################################################################

    ############################################################################
    # I.1. Generate code that reads aggregate states/controls
    #      from steady state deviations. Equations take the form of:
    # r       = exp.(Xss[indexes.rSS] .+ X[indexes.r])
    # rPrime  = exp.(Xss[indexes.rSS] .+ XPrime[indexes.r])
    ############################################################################

    # X => vector of steady state deviations
    # id => Dictionary mapping variable names (e.g. :union_retained_t) to indices (of X)
    # nt => NamedTuple mapping variable names (e.g. :union_retained_t) to steady state level
    #       i.e. nt[:union_retained_t] returns the steady state level of :union_retained_t
    # sslogdeviations2levels takes a log deviation and returns the level of the variable

    # Today
    @sslogdeviations2levels union_retained_t, retained_t = X, id, nt
    @sslogdeviations2levels Y_t1, B_t1, T_t1, I_t1, w_t1, q_t1 = X, id, nt
    @sslogdeviations2levels C_t1, avg_tax_rate_t1, τ_prog_t1 = X, id, nt
    @sslogdeviations2levels A_t, Z_t, Ψ_t, RB_t, μ_p_t, μ_w_t = X, id, nt
    @sslogdeviations2levels σ_t, G_sh_t, P_sh_t, R_sh_t, S_sh_t = X, id, nt
    @sslogdeviations2levels rk_t, w_t, K_t, π_t, π_w_t = X, id, nt
    @sslogdeviations2levels Y_t, C_t, q_t, N_t, mc_t, mc_w_t = X, id, nt
    @sslogdeviations2levels u_t, Ht_t, avg_tax_rate_t, T_t, I_t = X, id, nt
    @sslogdeviations2levels B_t, BD_t, BY_t, TY_t, mc_w_w_t = X, id, nt
    @sslogdeviations2levels G_t, τ_level_t, τ_prog_t = X, id, nt
    @sslogdeviations2levels Gini_C_t, Gini_X_t, sd_log_y_t = X, id, nt
    @sslogdeviations2levels I90_share_t, I90_share_net_t, W90_share_t = X, id, nt
    @sslogdeviations2levels Ygrowth_t, Bgrowth_t, Igrowth_t, wgrowth_t = X, id, nt
    @sslogdeviations2levels Cgrowth_t, Tgrowth_t, LP_t, LP_XA_t = X, id, nt
    @sslogdeviations2levels tot_retained_Y_t, union_firm_profits_t = X, id, nt
    @sslogdeviations2levels union_profits_t, firm_profits_t, profits_t = X, id, nt

    # Tomorrow # NOTE that we use XPrime, so id[:C_t] and id[:C_t] should point to the same indices
    # sslogdeviatiosn2levels_unprimekeys uses the fact that today's steady state parameter
    # equals tomorrow's steady state parameter to avoid extra allocations
    @sslogdeviations2levels_unprimekeys union_retained′_t, retained′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys Y′_t1, B′_t1, T′_t1, I′_t1, w′_t1, q′_t1 = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys C′_t1, avg_tax_rate′_t1, τ_prog′_t1 = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys A′_t, Z′_t, Ψ′_t, RB′_t, μ_p′_t, μ_w′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys σ′_t, G_sh′_t, P_sh′_t, R_sh′_t, S_sh′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys rk′_t, w′_t, K′_t, π′_t, π_w′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys Y′_t, C′_t, q′_t, N′_t, mc′_t, mc_w′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys u′_t, Ht′_t, avg_tax_rate′_t, T′_t, I′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys B′_t, BD′_t, BY′_t, TY′_t, mc_w_w′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys G′_t, τ_level′_t, τ_prog′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys Gini_C′_t, Gini_X′_t, sd_log_y′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys I90_share′_t, I90_share_net′_t, W90_share′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys Ygrowth′_t, Bgrowth′_t, Igrowth′_t, wgrowth′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys Cgrowth′_t, Tgrowth′_t, LP′_t, LP_XA′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys tot_retained_Y′_t, union_firm_profits′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys union_profits′_t, firm_profits′_t, profits′_t = XPrime, id, nt

    # Some special handling for type stability
    y_grid   = get_gridpts(grids, :y_grid)::Vector{Float64}
    y_ndgrid = grids[:y_ndgrid]::Array{Float64, 3}
    H        = grids[:H]::Float64

    # Read out equation scalar indices
    # @unpack from UnPack.jl allows you to write @unpack v = dictionary => v == dictionary[:v]
    @unpack_and_first eq_mp, eq_tax_progressivity, eq_tax_level = eq
    @unpack_and_first eq_tax_revenue, eq_avg_tax_rate, eq_deficit_rule = eq
    @unpack_and_first eq_gov_budget_constraint, eq_price_phillips_curve = eq
    @unpack_and_first eq_wage_phillips_curve, eq_real_wage_inflation, eq_capital_util = eq
    @unpack_and_first eq_capital_return, eq_received_wages, eq_wages_firms_pay = eq
    @unpack_and_first eq_union_firm_profits, eq_union_profits, eq_union_retained = eq
    @unpack_and_first eq_firm_profits, eq_profits_distr_to_hh, eq_retained = eq
    @unpack_and_first eq_tobins_q, eq_expost_liquidity_premium = eq
    @unpack_and_first eq_exante_liquidity_premium, eq_capital_accum, eq_labor_supply = eq
    @unpack_and_first eq_output, eq_resource_constraint, eq_capital_market_clear = eq
    @unpack_and_first eq_debt_market_clear, eq_bond_market_clear, eq_bond_output_ratio = eq
    @unpack_and_first eq_tax_output_ratio, eq_retained_earnings_gdp_ratio, eq_Ht = eq
    @unpack_and_first eq_Ygrowth, eq_Tgrowth, eq_Bgrowth = eq
    @unpack_and_first eq_Igrowth, eq_wgrowth, eq_Cgrowth = eq
    @unpack_and_first eq_LY, eq_LB, eq_LI, eq_Lw, eq_LT, eq_Lq, eq_LC = eq
    @unpack_and_first eq_Lavg_tax_rate, eq_Lτ_prog = eq
    @unpack_and_first eq_A, eq_Z, eq_Ψ, eq_μ_p, eq_μ_w, eq_σ, eq_G, eq_P, eq_R, eq_S = eq

# Elasticities and steepness from target markups for Phillips Curves
η_p                       = μ_p_t / (μ_p_t - 1.0)                                 # demand elasticity
κ_p                       = η_p * (θ[:κ_p] / θ[:μ_p]) * (θ[:μ_p] - 1.0)     # implied steepness of phillips curve
η_w                      = μ_w_t / (μ_w_t - 1.0)                               # demand elasticity wages
κ_w                      = η_w * (θ[:κ_w] / θ[:μ_w]) * (θ[:μ_w] - 1.0) # implied steepness of wage phillips curve

# Capital Utilization
MPK_SS                  = exp(nt[:rk_t]) - 1.0 + θ[:δ_0]       # stationary equil. marginal productivity of capital
δ_1                     = MPK_SS                                        # normailzation of utilization to 1 in stationary equilibrium
δ_2                     = δ_1 * θ[:δ_s]                              # express second utilization coefficient in relative terms
# Auxiliary variables
Kserv                   = K_t * u_t                                         # Effective capital
MPKserv                 = mc_t * Z_t * θ[:α] * (Kserv ./ N_t) .^(θ[:α] - 1.0)      # marginal product of Capital
depr                    = θ[:δ_0] + δ_1 * (u_t - 1.0) + δ_2 / 2.0 * (u_t - 1.0)^2.0   # depreciation

Wagesum                 = N_t * w_t                                         # Total wages in economy t
Wagesum′            = N′_t * w′_t                               # Total wages in economy t+1

# Efficient ouput and employment
# Note, need the DSGE tag at the beginning b/c aggregate_equations is not included
# as part of the package but called via @include
N_GAP                   = _bbl_employment(K_t, Z_t ./ (θ[:μ_p] * θ[:μ_w]), θ[:α], θ[:τ_lev], θ[:τ_prog], θ[:γ])
Y_GAP                   = _bbl_output(K_t, Z_t, N_GAP, θ[:α])

# tax progressivity variabels used to calculate e.g. total taxes
tax_prog_scale          = (θ[:γ] + θ[:τ_prog]) / ((θ[:γ] + τ_prog_t))                        # scaling of labor disutility including tax progressivity
# TODO: check if we can avoid doing this inc = [...] stuff b/c it seems it's just allocating one vector??
inc                     = [τ_level_t .* ((y_ndgrid ./ H).^tax_prog_scale .*
                                         mc_w_t .* w_t .* N_t ./ Ht_t).^(1.0 - τ_prog_t)]                                 # capital liquidation Income (q=1 in steady state)
inc[1][:,:,end]        .= τ_level_t .* (y_ndgrid[:, :, end] .* profits_t).^(1.0 - τ_prog_t)             # profit income net of taxes

incgross                = [((y_ndgrid ./ H).^tax_prog_scale .* mc_w_t .* w_t .* N_t ./ Ht_t)]   # capital liquidation Income (q=1 in steady state)
incgross[1][:,:,end]   .= (y_ndgrid[:, :, end] .* profits_t)                                    # gross profit income

taxrev                  = incgross[1] .- inc[1]                                                 # tax revenues
incgrossaux             = incgross[1]
# Summary for aggregate human capital
# distr_y                 = sum(nt[:distr_t], dims=(1,2)) # TODO: can we replace this with nt[:marginal_pdf_y_t]??
distr_y                 = nt[:marginal_pdf_y_t]
Htact                   = dot(distr_y[1:end-1], (y_grid[1:end-1] ./ H).^((θ[:γ] + θ[:τ_prog]) / (θ[:γ] + τ_prog_t)))
# Htact                   = sum(distr_y[1:end-1] .* (y_grid[1:end-1] ./ H).^((θ[:γ] + θ[:τ_prog]) / (θ[:γ] + τ_prog_t)))

############################################################################
#           Error term calculations (i.e. model starts here)          #
############################################################################

#-------- States -----------#
# Error Term on exogeneous States
# Shock processes
F[eq_G]       = log(G_sh′_t)         - θ[:ρ_G] * log(G_sh_t)     # primary deficit shock
F[eq_P]   = log(P_sh′_t)         - θ[:ρ_P_sh] * log(P_sh_t) # tax shock

F[eq_R]       = log(R_sh′_t)         - θ[:ρ_R_sh] * log(R_sh_t)     # Taylor rule shock
F[eq_S]       = log(S_sh′_t)         - θ[:ρ_S_sh] * log(S_sh_t)     # uncertainty shock

# Stochastic states that can be directly moved (no feedback)
F[eq_A]            = log(A′_t)              - θ[:ρ_A] * log(A_t)               # (unobserved) Private bond return fed-funds spread (produces goods out of nothing if negative)
F[eq_Z]            = log(Z′_t)              - θ[:ρ_Z] * log(Z_t)               # TFP
F[eq_Ψ]           = log(Ψ′_t)              - θ[:ρ_Ψ] * log(Ψ_t)             # Investment-good productivity

F[eq_μ_p]            = log(μ_p′_t / θ[:μ_p])   - θ[:ρ_μ_p] * log(μ_p_t / θ[:μ_p])      # Process for markup target
F[eq_μ_w]           = log(μ_w′_t / θ[:μ_w])   - θ[:ρ_μ_w] * log(μ_w_t / θ[:μ_w])   # Process for w-markup target

# Endogeneous States (including Lags)
F[eq_σ]            = log(σ′_t)              - (θ[:ρ_S] * log(σ_t) + (1.0 - θ[:ρ_S]) *
                                               θ[:Σ_n] * log(Ygrowth_t) + log(S_sh_t)) # Idiosyncratic income risk (contemporaneous reaction to business cycle)

F[eq_LY]         = log(Y′_t1)    - log(Y_t)
F[eq_LB]         = log(B′_t1)    - log(B_t)
F[eq_LI]         = log(I′_t1)    - log(I_t)
F[eq_Lw]         = log(w′_t1)    - log(w_t)
F[eq_LT]         = log(T′_t1)    - log(T_t)
F[eq_Lq]         = log(q′_t1)    - log(q_t)
F[eq_LC]         = log(C′_t1)    - log(C_t)
F[eq_Lavg_tax_rate] = log(avg_tax_rate′_t1) - log(avg_tax_rate_t)
F[eq_Lτ_prog]     = log(τ_prog′_t1) - log(τ_prog_t)

# Growth rates
F[eq_Ygrowth]      = log(Ygrowth_t)      - log(Y_t / Y_t1)
F[eq_Tgrowth]      = log(Tgrowth_t)      - log(T_t / T_t1)
F[eq_Bgrowth]      = log(Bgrowth_t)      - log(B_t / B_t1)
F[eq_Igrowth]      = log(Igrowth_t)      - log(I_t / I_t1)
F[eq_wgrowth]      = log(wgrowth_t)      - log(w_t / w_t1)
F[eq_Cgrowth]      = log(Cgrowth_t)      - log(C_t / C_t1)

#  Taylor rule and interest rates
F[eq_mp]           = log(RB′_t) - nt[:RB_t] -
    ((1 - θ[:ρ_R]) * θ[:θ_π]) * log(π_t) -
    ((1 - θ[:ρ_R]) * θ[:θ_Y]) * log(Y_t / Y_GAP) -
    θ[:ρ_R] * (log(RB_t) - nt[:RB_t])  - log(R_sh_t)

# Tax rule
F[eq_tax_progressivity]        = log(τ_prog_t) - θ[:ρ_P] * log(τ_prog_t1)  - # TODO: find correct name of τ_prog_lag_t1
    (1.0 - θ[:ρ_P]) * (nt[:τ_prog_t]) -
    (1.0 - θ[:ρ_P]) * θ[:γ_Y_P] * log(Y_t / Y_GAP) -
    (1.0 - θ[:ρ_P]) * θ[:γ_B_P] * (log(B_t)- nt[:B_t]) -
    log(P_sh_t)

F[eq_tax_level]         = avg_tax_rate_t - dot(nt[:distr_t], taxrev) / dot(nt[:distr_t], incgrossaux) # Union profits are taxed at average tax rate

F[eq_tax_revenue]            = log(T_t) - log(dot(nt[:distr_t], taxrev) + avg_tax_rate_t * union_profits_t)
#=F[eq_tax_level]         = avg_tax_rate_t - sum(nt[:distr_t] .* taxrev) / sum(nt[:distr_t] .* incgrossaux) # Union profits are taxed at average tax rate

F[eq_tax_revenue]            = log(T_t) - log(sum(nt[:distr_t] .* taxrev) + avg_tax_rate_t * union_profits_t)=#

F[eq_avg_tax_rate]  = log(avg_tax_rate_t) - θ[:ρ_τ] * log(avg_tax_rate_t1)  -
    (1.0 - θ[:ρ_τ]) * nt[:avg_tax_rate_t] -
    (1.0 - θ[:ρ_τ]) * θ[:γ_Y_τ] * log(Y_t / Y_GAP) -
    (1.0 - θ[:ρ_τ]) * θ[:γ_B_τ] * (log(B_t) - nt[:B_t])

# --------- Controls ------------
# Deficit rule
F[eq_deficit_rule]            = log(Bgrowth′_t) + θ[:γ_B] * (log(B_t) - nt[:B_t])  -
    θ[:γ_Y] * log(Y_t / Y_GAP)  - θ[:γ_π] * log(π_t) - log(G_sh_t)

F[eq_gov_budget_constraint]            = log(G_t) - log(B′_t + T_t - RB_t / π_t * B_t)             # Government Budget Constraint

# Phillips Curve to determine equilibrium markup, output, factor incomes
F[eq_price_phillips_curve]           = (log(π_t) - nt[:π_t]) - κ_p * (mc_t - 1 / μ_p_t ) -
    θ[:β] * ((log(π′_t) - nt[:π_t]) * Y′_t / Y_t)

# Wage Phillips Curve
F[eq_wage_phillips_curve]          = (log(π_w_t) - nt[:π_w_t]) - (κ_w * (mc_w_t - 1 / μ_w_t) +
                                                                  θ[:β] * ((log(π_w′_t) - nt[:π_w_t]) * Wagesum′ / Wagesum))
# worker's wage = mcw * firm's wage
# Wage Dynamics
F[eq_real_wage_inflation]           = log(w_t / w_t1) - log(π_w_t / π_t)                   # Definition of real wage inflation

# Capital utilisation
F[eq_capital_util]            = MPKserv  -  q_t * (δ_1 + δ_2 * (u_t - 1.0))           # Optimality condition for utilization

# Prices
F[eq_capital_return]            = log(rk_t) - log(1. + MPKserv * u_t - q_t * depr)       # rate of return on capital

F[eq_received_wages]         = log(mc_w_w_t) - log(mc_w_t * w_t)                        # wages that workers receive

F[eq_wages_firms_pay]            = log(w_t) - log(_bbl_wage(Kserv, Z_t * mc_t, N_t, θ[:α]))     # wages that firms pay

F[eq_union_firm_profits]   = log(union_firm_profits_t)  - log(w_t * N_t * (1.0 - mc_w_t))  # profits of the monopolistic unions
F[eq_union_profits]         = log(union_profits_t)        - log((1.0 - exp(nt[:mc_w_t])) * w_t * N_t  + θ[:ω_U] *
                                                                (union_firm_profits_t - (1.0 - exp(nt[:mc_w_t])) * w_t * N_t + log(union_retained_t))) # distributed profits to households
F[eq_union_retained]       = log(union_retained′_t) - (union_firm_profits_t - union_profits_t + log(union_retained_t) * (RB_t / π_t)) # Accumulation equation, retained is in levels

F[eq_firm_profits]         = log(firm_profits_t)  - log(Y_t * (1.0 - mc_t))                                               # profits of the monopolistic resellers
F[eq_profits_distr_to_hh]              = log(profits_t)       - log((1.0 - exp(nt[:mc_t])) * Y_t + θ[:ω_F] *
                                                                    (firm_profits_t - (1.0 - exp(nt[:mc_t])) * Y_t + log(retained_t))) # distributed profits to households
F[eq_retained]             = log(retained′_t) - (firm_profits_t - profits_t + log(retained_t) * (RB_t / π_t))            # Accumulation equation (levels)

F[eq_tobins_q]            = 1.0 - Ψ_t * q_t * (1.0 - θ[:ϕ] / 2.0 * (Igrowth_t - 1.0)^2.0 - # price of capital investment adjustment costs
                                               θ[:ϕ] * (Igrowth_t - 1.0) * Igrowth_t)  -
                                               θ[:β] * Ψ′_t * q′_t * θ[:ϕ] * (Igrowth′_t - 1.0) * (Igrowth′_t)^2.0

# Asset market premia
F[eq_expost_liquidity_premium]           = log(LP_t)                  - (log((q_t + rk_t - 1.0) / q_t1) - log(RB_t / π_t))                   # Ex-post liquidity premium
F[eq_exante_liquidity_premium]         = log(LP_XA_t)                - (log((q′_t + rk′_t - 1.0) / q_t) - log(RB′_t / π′_t))  # ex-ante liquidity premium


# Aggregate Quantities
F[eq_capital_accum]            = K′_t -  K_t * (1.0 - depr)  - Ψ_t * I_t * (1.0 - θ[:ϕ] / 2.0 * (Igrowth_t -1.0).^2.0)           # Capital accumulation equation
F[eq_labor_supply]            = log(N_t) - log(((1.0 - τ_prog_t) * τ_level_t * (mc_w_t * w_t).^(1.0 - τ_prog_t)).^(1.0 / (θ[:γ] + τ_prog_t)) * Ht_t)   # labor supply
F[eq_output]            = log(Y_t) - log(Z_t * N_t .^(1.0 - θ[:α]) * Kserv .^ θ[:α])                                          # production function
F[eq_resource_constraint]            = log(Y_t - G_t - I_t - BD_t * θ[:Rbar] + (A_t - 1.0) * RB_t * B_t / π_t -
                                           (δ_1 * (u_t - 1.0) + δ_2 / 2.0 * (u_t - 1.0)^2.0) * K_t ) - log(C_t) # Resource constraint

# Error Term on prices/aggregate summary vars (logarithmic, controls), here difference to SS value averages
F[eq_capital_market_clear]            = log(K_t)     - nt[:K_t]                                                            # Capital market clearing
F[eq_debt_market_clear]           = log(BD_t)    - nt[:BD_t]                                                        # IOUs
F[eq_bond_market_clear]            = log(B_t)     - log(exp(nt[:B_t]) + log(retained_t) + log(union_retained_t) )   # Bond market clearing
F[eq_bond_output_ratio]           = log(BY_t)    - log(B_t / Y_t)                                                               # Bond to Output ratio
F[eq_tax_output_ratio]           = log(TY_t)    - log(T_t / Y_t)                                                               # Tax to output ratio
F[eq_retained_earnings_gdp_ratio] = log(tot_retained_Y_t) - ((log(retained_t) + log(union_retained_t)) / Y_t)                      # retained Earnings to GDP

# Add distributional summary stats that do change with other aggregate controls/prices so that the stationary
F[eq_Ht]           = log(Ht_t) - log(Htact)

return F
end

# This is meant to remove the unnecessary states. It is not done!
function _Fsys_agg(F::AbstractVector, X::AbstractArray, XPrime::AbstractArray,
                  θ::NamedTuple, grids::OrderedDict, id::OrderedDict{Symbol, I1}, nt::NamedTuple,
                  eq::OrderedDict{Symbol, I2}) where {I1 <: Union{Int, UnitRange}, I2 <: Union{Int, UnitRange}}

    ############################################################################
    #            I. Read out argument values                                   #
    ############################################################################

    ############################################################################
    # I.1. Generate code that reads aggregate states/controls
    #      from steady state deviations. Equations take the form of:
    # r       = exp.(Xss[indexes.rSS] .+ X[indexes.r])
    # rPrime  = exp.(Xss[indexes.rSS] .+ XPrime[indexes.r])
    ############################################################################

    # X => vector of steady state deviations
    # id => Dictionary mapping variable names (e.g. :union_retained_t) to indices (of X)
    # nt => NamedTuple mapping variable names (e.g. :union_retained_t) to steady state level
    #       i.e. nt[:union_retained_t] returns the steady state level of :union_retained_t
    # sslogdeviations2levels takes a log deviation and returns the level of the variable

    # Today
    @sslogdeviations2levels union_retained_t, retained_t = X, id, nt
#=    @sslogdeviations2levels Y_t1, B_t1, T_t1, I_t1, w_t1, q_t1 = X, id, nt
    @sslogdeviations2levels C_t1, avg_tax_rate_t1, τ_prog_t1 = X, id, nt=#
    @sslogdeviations2levels Y_t1, B_t1, I_t1, w_t1, q_t1 = X, id, nt
    @sslogdeviations2levels avg_tax_rate_t1, τ_prog_t1 = X, id, nt
    @sslogdeviations2levels A_t, Z_t, Ψ_t, RB_t, μ_p_t, μ_w_t = X, id, nt
    @sslogdeviations2levels σ_t, G_sh_t, P_sh_t, R_sh_t, S_sh_t = X, id, nt
    @sslogdeviations2levels rk_t, w_t, K_t, π_t, π_w_t = X, id, nt
    @sslogdeviations2levels Y_t, C_t, q_t, N_t, mc_t, mc_w_t = X, id, nt
    @sslogdeviations2levels u_t, Ht_t, avg_tax_rate_t, T_t, I_t = X, id, nt
#     @sslogdeviations2levels B_t, BD_t, BY_t, TY_t, mc_w_w_t = X, id, nt
    @sslogdeviations2levels B_t, BD_t, mc_w_w_t = X, id, nt
    @sslogdeviations2levels G_t, τ_level_t, τ_prog_t = X, id, nt
    @sslogdeviations2levels Gini_C_t, Gini_X_t, sd_log_y_t = X, id, nt
    @sslogdeviations2levels I90_share_t, I90_share_net_t, W90_share_t = X, id, nt
#=    @sslogdeviations2levels Ygrowth_t, Bgrowth_t, Igrowth_t, wgrowth_t = X, id, nt
    @sslogdeviations2levels Cgrowth_t, Tgrowth_t, LP_t, LP_XA_t = X, id, nt=#
    @sslogdeviations2levels Ygrowth_t, Bgrowth_t, Igrowth_t = X, id, nt
    @sslogdeviations2levels LP_t, LP_XA_t = X, id, nt
    @sslogdeviations2levels tot_retained_Y_t, union_firm_profits_t = X, id, nt
    @sslogdeviations2levels union_profits_t, firm_profits_t, profits_t = X, id, nt

    # Tomorrow # NOTE that we use XPrime, so id[:C_t] and id[:C_t] should point to the same indices
    # sslogdeviatiosn2levels_unprimekeys uses the fact that today's steady state parameter
    # equals tomorrow's steady state parameter to avoid extra allocations
    @sslogdeviations2levels_unprimekeys union_retained′_t, retained′_t = XPrime, id, nt
#=    @sslogdeviations2levels_unprimekeys Y′_t1, B′_t1, T′_t1, I′_t1, w′_t1, q′_t1 = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys C′_t1, avg_tax_rate′_t1, τ_prog′_t1 = XPrime, id, nt=#
    @sslogdeviations2levels_unprimekeys Y′_t1, B′_t1, I′_t1, w′_t1, q′_t1 = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys avg_tax_rate′_t1, τ_prog′_t1 = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys A′_t, Z′_t, Ψ′_t, RB′_t, μ_p′_t, μ_w′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys σ′_t, G_sh′_t, P_sh′_t, R_sh′_t, S_sh′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys rk′_t, w′_t, K′_t, π′_t, π_w′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys Y′_t, C′_t, q′_t, N′_t, mc′_t, mc_w′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys u′_t, Ht′_t, avg_tax_rate′_t, T′_t, I′_t = XPrime, id, nt
#     @sslogdeviations2levels_unprimekeys B′_t, BD′_t, BY′_t, TY′_t, mc_w_w′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys B′_t, BD′_t, mc_w_w′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys G′_t, τ_level′_t, τ_prog′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys Gini_C′_t, Gini_X′_t, sd_log_y′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys I90_share′_t, I90_share_net′_t, W90_share′_t = XPrime, id, nt
#=    @sslogdeviations2levels_unprimekeys Ygrowth′_t, Bgrowth′_t, Igrowth′_t, wgrowth′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys Cgrowth′_t, Tgrowth′_t, LP′_t, LP_XA′_t = XPrime, id, nt=#
    @sslogdeviations2levels_unprimekeys Ygrowth′_t, Bgrowth′_t, Igrowth′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys LP′_t, LP_XA′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys tot_retained_Y′_t, union_firm_profits′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys union_profits′_t, firm_profits′_t, profits′_t = XPrime, id, nt

    # Some special handling for type stability
    y_grid   = get_gridpts(grids, :y_grid)::Vector{Float64}
    y_ndgrid = grids[:y_ndgrid]::Array{Float64, 3}
    H        = grids[:H]::Float64

    # Read out equation scalar indices
    # @unpack from UnPack.jl allows you to write @unpack v = dictionary => v == dictionary[:v]
    @unpack_and_first eq_mp, eq_tax_progressivity, eq_tax_level = eq
    @unpack_and_first eq_tax_revenue, eq_avg_tax_rate, eq_deficit_rule = eq
    @unpack_and_first eq_gov_budget_constraint, eq_price_phillips_curve = eq
    @unpack_and_first eq_wage_phillips_curve, eq_real_wage_inflation, eq_capital_util = eq
    @unpack_and_first eq_capital_return, eq_received_wages, eq_wages_firms_pay = eq
    @unpack_and_first eq_union_firm_profits, eq_union_profits, eq_union_retained = eq
    @unpack_and_first eq_firm_profits, eq_profits_distr_to_hh, eq_retained = eq
    @unpack_and_first eq_tobins_q, eq_expost_liquidity_premium = eq
    @unpack_and_first eq_exante_liquidity_premium, eq_capital_accum, eq_labor_supply = eq
    @unpack_and_first eq_output, eq_resource_constraint, eq_capital_market_clear = eq
#=    @unpack_and_first eq_debt_market_clear, eq_bond_market_clear, eq_bond_output_ratio = eq
    @unpack_and_first eq_tax_output_ratio, eq_retained_earnings_gdp_ratio, eq_Ht = eq=#
    @unpack_and_first eq_debt_market_clear, eq_bond_market_clear = eq
    @unpack_and_first eq_retained_earnings_gdp_ratio, eq_Ht = eq
#=    @unpack_and_first eq_Ygrowth, eq_Tgrowth, eq_Bgrowth = eq
    @unpack_and_first eq_Igrowth, eq_wgrowth, eq_Cgrowth = eq=#
    @unpack_and_first eq_Ygrowth, eq_Bgrowth, eq_Igrowth = eq
#=    @unpack_and_first eq_LY, eq_LB, eq_LI, eq_Lw, eq_LT, eq_Lq, eq_LC = eq
    @unpack_and_first eq_Lavg_tax_rate, eq_Lτ_prog = eq=#
    @unpack_and_first eq_LY, eq_LB, eq_LI, eq_Lw, eq_Lq = eq
    @unpack_and_first eq_Lavg_tax_rate, eq_Lτ_prog = eq
    @unpack_and_first eq_A, eq_Z, eq_Ψ, eq_μ_p, eq_μ_w, eq_σ, eq_G, eq_P, eq_R, eq_S = eq

# Elasticities and steepness from target markups for Phillips Curves
η_p                       = μ_p_t / (μ_p_t - 1.0)                                 # demand elasticity
κ_p                       = η_p * (θ[:κ_p] / θ[:μ_p]) * (θ[:μ_p] - 1.0)     # implied steepness of phillips curve
η_w                      = μ_w_t / (μ_w_t - 1.0)                               # demand elasticity wages
κ_w                      = η_w * (θ[:κ_w] / θ[:μ_w]) * (θ[:μ_w] - 1.0) # implied steepness of wage phillips curve

# Capital Utilization
MPK_SS                  = exp(nt[:rk_t]) - 1.0 + θ[:δ_0]       # stationary equil. marginal productivity of capital
δ_1                     = MPK_SS                                        # normailzation of utilization to 1 in stationary equilibrium
δ_2                     = δ_1 * θ[:δ_s]                              # express second utilization coefficient in relative terms
# Auxiliary variables
Kserv                   = K_t * u_t                                         # Effective capital
MPKserv                 = mc_t * Z_t * θ[:α] * (Kserv ./ N_t) .^(θ[:α] - 1.0)      # marginal product of Capital
depr                    = θ[:δ_0] + δ_1 * (u_t - 1.0) + δ_2 / 2.0 * (u_t - 1.0)^2.0   # depreciation

Wagesum                 = N_t * w_t                                         # Total wages in economy t
Wagesum′            = N′_t * w′_t                               # Total wages in economy t+1

# Efficient ouput and employment
# Note, need the DSGE tag at the beginning b/c aggregate_equations is not included
# as part of the package but called via @include
N_GAP                   = _bbl_employment(K_t, Z_t ./ (θ[:μ_p] * θ[:μ_w]), θ[:α], θ[:τ_lev], θ[:τ_prog], θ[:γ])
Y_GAP                   = _bbl_output(K_t, Z_t, N_GAP, θ[:α])

# tax progressivity variabels used to calculate e.g. total taxes
tax_prog_scale          = (θ[:γ] + θ[:τ_prog]) / ((θ[:γ] + τ_prog_t))                        # scaling of labor disutility including tax progressivity
# TODO: check if we can avoid doing this inc = [...] stuff b/c it seems it's just allocating one vector??
inc                     = [τ_level_t .* ((y_ndgrid ./ H).^tax_prog_scale .*
                                         mc_w_t .* w_t .* N_t ./ Ht_t).^(1.0 - τ_prog_t)]                                 # capital liquidation Income (q=1 in steady state)
inc[1][:,:,end]        .= τ_level_t .* (y_ndgrid[:, :, end] .* profits_t).^(1.0 - τ_prog_t)             # profit income net of taxes

incgross                = [((y_ndgrid ./ H).^tax_prog_scale .* mc_w_t .* w_t .* N_t ./ Ht_t)]   # capital liquidation Income (q=1 in steady state)
incgross[1][:,:,end]   .= (y_ndgrid[:, :, end] .* profits_t)                                    # gross profit income

taxrev                  = incgross[1] .- inc[1]                                                 # tax revenues
incgrossaux             = incgross[1]
# Summary for aggregate human capital
# distr_y                 = sum(nt[:distr_t], dims=(1,2)) # TODO: can we replace this with nt[:marginal_pdf_y_t]??
distr_y                 = nt[:marginal_pdf_y_t]
Htact                   = dot(distr_y[1:end-1], (y_grid[1:end-1] ./ H).^((θ[:γ] + θ[:τ_prog]) / (θ[:γ] + τ_prog_t)))
# Htact                   = sum(distr_y[1:end-1] .* (y_grid[1:end-1] ./ H).^((θ[:γ] + θ[:τ_prog]) / (θ[:γ] + τ_prog_t)))

############################################################################
#           Error term calculations (i.e. model starts here)          #
############################################################################

#-------- States -----------#
# Error Term on exogeneous States
# Shock processes
F[eq_G]       = log(G_sh′_t)         - θ[:ρ_G] * log(G_sh_t)     # primary deficit shock
F[eq_P]   = log(P_sh′_t)         - θ[:ρ_P_sh] * log(P_sh_t) # tax shock

F[eq_R]       = log(R_sh′_t)         - θ[:ρ_R_sh] * log(R_sh_t)     # Taylor rule shock
F[eq_S]       = log(S_sh′_t)         - θ[:ρ_S_sh] * log(S_sh_t)     # uncertainty shock

# Stochastic states that can be directly moved (no feedback)
F[eq_A]            = log(A′_t)              - θ[:ρ_A] * log(A_t)               # (unobserved) Private bond return fed-funds spread (produces goods out of nothing if negative)
F[eq_Z]            = log(Z′_t)              - θ[:ρ_Z] * log(Z_t)               # TFP
F[eq_Ψ]           = log(Ψ′_t)              - θ[:ρ_Ψ] * log(Ψ_t)             # Investment-good productivity

F[eq_μ_p]            = log(μ_p′_t / θ[:μ_p])   - θ[:ρ_μ_p] * log(μ_p_t / θ[:μ_p])      # Process for markup target
F[eq_μ_w]           = log(μ_w′_t / θ[:μ_w])   - θ[:ρ_μ_w] * log(μ_w_t / θ[:μ_w])   # Process for w-markup target

# Endogeneous States (including Lags)
F[eq_σ]            = log(σ′_t)              - (θ[:ρ_S] * log(σ_t) + (1.0 - θ[:ρ_S]) *
                                               θ[:Σ_n] * log(Ygrowth_t) + log(S_sh_t)) # Idiosyncratic income risk (contemporaneous reaction to business cycle)

# Lags
F[eq_LY]         = log(Y′_t1)    - log(Y_t)
F[eq_LB]         = log(B′_t1)    - log(B_t)
F[eq_LI]         = log(I′_t1)    - log(I_t)
F[eq_Lw]         = log(w′_t1)    - log(w_t)
# F[eq_LT]         = log(T′_t1)    - log(T_t)
F[eq_Lq]         = log(q′_t1)    - log(q_t)
# F[eq_LC]         = log(C′_t1)    - log(C_t)
F[eq_Lavg_tax_rate] = log(avg_tax_rate′_t1) - log(avg_tax_rate_t)
F[eq_Lτ_prog]     = log(τ_prog′_t1) - log(τ_prog_t)

# Growth rates
F[eq_Ygrowth]      = log(Ygrowth_t)      - log(Y_t / Y_t1)
# F[eq_Tgrowth]      = log(Tgrowth_t)      - log(T_t / T_t1)
F[eq_Bgrowth]      = log(Bgrowth_t)      - log(B_t / B_t1)
F[eq_Igrowth]      = log(Igrowth_t)      - log(I_t / I_t1)
# F[eq_wgrowth]      = log(wgrowth_t)      - log(w_t / w_t1)
# F[eq_Cgrowth]      = log(Cgrowth_t)      - log(C_t / C_t1)

#  Taylor rule and interest rates
F[eq_mp]           = log(RB′_t) - nt[:RB_t] -
    ((1 - θ[:ρ_R]) * θ[:θ_π]) * log(π_t) -
    ((1 - θ[:ρ_R]) * θ[:θ_Y]) * log(Y_t / Y_GAP) -
    θ[:ρ_R] * (log(RB_t) - nt[:RB_t])  - log(R_sh_t)

# Tax rule
F[eq_tax_progressivity]        = log(τ_prog_t) - θ[:ρ_P] * log(τ_prog_t1)  - # TODO: find correct name of τ_prog_lag_t1
    (1.0 - θ[:ρ_P]) * (nt[:τ_prog_t]) -
    (1.0 - θ[:ρ_P]) * θ[:γ_Y_P] * log(Y_t / Y_GAP) -
    (1.0 - θ[:ρ_P]) * θ[:γ_B_P] * (log(B_t)- nt[:B_t]) -
    log(P_sh_t)

F[eq_tax_level]         = avg_tax_rate_t - dot(nt[:distr_t], taxrev) / dot(nt[:distr_t], incgrossaux) # Union profits are taxed at average tax rate

F[eq_tax_revenue]            = log(T_t) - log(dot(nt[:distr_t], taxrev) + avg_tax_rate_t * union_profits_t)

F[eq_avg_tax_rate]  = log(avg_tax_rate_t) - θ[:ρ_τ] * log(avg_tax_rate_t1)  -
    (1.0 - θ[:ρ_τ]) * nt[:avg_tax_rate_t] -
    (1.0 - θ[:ρ_τ]) * θ[:γ_Y_τ] * log(Y_t / Y_GAP) -
    (1.0 - θ[:ρ_τ]) * θ[:γ_B_τ] * (log(B_t) - nt[:B_t])

# --------- Controls ------------
# Deficit rule
F[eq_deficit_rule]            = log(Bgrowth′_t) + θ[:γ_B] * (log(B_t) - nt[:B_t])  -
    θ[:γ_Y] * log(Y_t / Y_GAP)  - θ[:γ_π] * log(π_t) - log(G_sh_t)

F[eq_gov_budget_constraint]            = log(G_t) - log(B′_t + T_t - RB_t / π_t * B_t)             # Government Budget Constraint

# Phillips Curve to determine equilibrium markup, output, factor incomes
F[eq_price_phillips_curve]           = (log(π_t) - nt[:π_t]) - κ_p * (mc_t - 1 / μ_p_t ) -
    θ[:β] * ((log(π′_t) - nt[:π_t]) * Y′_t / Y_t)

# Wage Phillips Curve
F[eq_wage_phillips_curve]          = (log(π_w_t) - nt[:π_w_t]) - (κ_w * (mc_w_t - 1 / μ_w_t) +
                                                                  θ[:β] * ((log(π_w′_t) - nt[:π_w_t]) * Wagesum′ / Wagesum))
# worker's wage = mcw * firm's wage
# Wage Dynamics
F[eq_real_wage_inflation]           = log(w_t / w_t1) - log(π_w_t / π_t)                   # Definition of real wage inflation

# Capital utilisation
F[eq_capital_util]            = MPKserv  -  q_t * (δ_1 + δ_2 * (u_t - 1.0))           # Optimality condition for utilization

# Prices
F[eq_capital_return]            = log(rk_t) - log(1. + MPKserv * u_t - q_t * depr)       # rate of return on capital

F[eq_received_wages]         = log(mc_w_w_t) - log(mc_w_t * w_t)                        # wages that workers receive

F[eq_wages_firms_pay]            = log(w_t) - log(_bbl_wage(Kserv, Z_t * mc_t, N_t, θ[:α]))     # wages that firms pay

F[eq_union_firm_profits]   = log(union_firm_profits_t)  - log(w_t * N_t * (1.0 - mc_w_t))  # profits of the monopolistic unions
F[eq_union_profits]         = log(union_profits_t)        - log((1.0 - exp(nt[:mc_w_t])) * w_t * N_t  + θ[:ω_U] *
                                                                (union_firm_profits_t - (1.0 - exp(nt[:mc_w_t])) * w_t * N_t + log(union_retained_t))) # distributed profits to households
F[eq_union_retained]       = log(union_retained′_t) - (union_firm_profits_t - union_profits_t + log(union_retained_t) * (RB_t / π_t)) # Accumulation equation, retained is in levels

F[eq_firm_profits]         = log(firm_profits_t)  - log(Y_t * (1.0 - mc_t))                                               # profits of the monopolistic resellers
F[eq_profits_distr_to_hh]              = log(profits_t)       - log((1.0 - exp(nt[:mc_t])) * Y_t + θ[:ω_F] *
                                                                    (firm_profits_t - (1.0 - exp(nt[:mc_t])) * Y_t + log(retained_t))) # distributed profits to households
F[eq_retained]             = log(retained′_t) - (firm_profits_t - profits_t + log(retained_t) * (RB_t / π_t))            # Accumulation equation (levels)

F[eq_tobins_q]            = 1.0 - Ψ_t * q_t * (1.0 - θ[:ϕ] / 2.0 * (Igrowth_t - 1.0)^2.0 - # price of capital investment adjustment costs
                                               θ[:ϕ] * (Igrowth_t - 1.0) * Igrowth_t)  -
                                               θ[:β] * Ψ′_t * q′_t * θ[:ϕ] * (Igrowth′_t - 1.0) * (Igrowth′_t)^2.0

# Asset market premia
F[eq_expost_liquidity_premium]           = log(LP_t)                  - (log((q_t + rk_t - 1.0) / q_t1) - log(RB_t / π_t))                   # Ex-post liquidity premium
F[eq_exante_liquidity_premium]         = log(LP_XA_t)                - (log((q′_t + rk′_t - 1.0) / q_t) - log(RB′_t / π′_t))  # ex-ante liquidity premium


# Aggregate Quantities
F[eq_capital_accum]            = K′_t -  K_t * (1.0 - depr)  - Ψ_t * I_t * (1.0 - θ[:ϕ] / 2.0 * (Igrowth_t -1.0).^2.0)           # Capital accumulation equation
F[eq_labor_supply]            = log(N_t) - log(((1.0 - τ_prog_t) * τ_level_t * (mc_w_t * w_t).^(1.0 - τ_prog_t)).^(1.0 / (θ[:γ] + τ_prog_t)) * Ht_t)   # labor supply
F[eq_output]            = log(Y_t) - log(Z_t * N_t .^(1.0 - θ[:α]) * Kserv .^ θ[:α])                                          # production function
F[eq_resource_constraint]            = log(Y_t - G_t - I_t - BD_t * θ[:Rbar] + (A_t - 1.0) * RB_t * B_t / π_t -
                                           (δ_1 * (u_t - 1.0) + δ_2 / 2.0 * (u_t - 1.0)^2.0) * K_t ) - log(C_t) # Resource constraint

# Error Term on prices/aggregate summary vars (logarithmic, controls), here difference to SS value averages
F[eq_capital_market_clear]            = log(K_t)     - nt[:K_t]                                                            # Capital market clearing
F[eq_debt_market_clear]           = log(BD_t)    - nt[:BD_t]                                                        # IOUs
F[eq_bond_market_clear]            = log(B_t)     - log(exp(nt[:B_t]) + log(retained_t) + log(union_retained_t) )   # Bond market clearing
# F[eq_bond_output_ratio]           = log(BY_t)    - log(B_t / Y_t)                                                               # Bond to Output ratio
# F[eq_tax_output_ratio]           = log(TY_t)    - log(T_t / Y_t)                                                               # Tax to output ratio
F[eq_retained_earnings_gdp_ratio] = log(tot_retained_Y_t) - ((log(retained_t) + log(union_retained_t)) / Y_t)                      # retained Earnings to GDP

# Add distributional summary stats that do change with other aggregate controls/prices so that the stationary
F[eq_Ht]           = log(Ht_t) - log(Htact)

return F
end
