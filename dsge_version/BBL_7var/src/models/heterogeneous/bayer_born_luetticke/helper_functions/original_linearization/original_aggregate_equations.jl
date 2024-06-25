#------------------------------------------------------------------------------
# THIS FILE CONTAINS THE "AGGREGATE" MODEL EQUATIONS, I.E. EVERYTHING  BUT THE
# HOUSEHOLD PLANNING PROBLEM. THE lATTER IS DESCRIBED BY ONE EGM BACKWARD STEP AND
# ONE FORWARD ITERATION OF THE DISTRIBUTION.
#
# AGGREGATE EQUATIONS TAKE THE FORM
# F[EQUATION NUMBER] = lhs - rhs
#
# EQUATION NUMBERS ARE GENEREATED AUTOMATICALLY AND STORED IN THE INDEX STRUCT
# FOR THIS THE "CORRESPONDING" VARIABLE NEEDS TO BE IN THE LIST OF STATES
# OR CONTROLS.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# AUXILIARY VARIABLES ARE DEFINED FIRST
#------------------------------------------------------------------------------


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

# Eficient ouput and employment
N_GAP                   = _bbl_employment(K_t, Z_t ./ (θ[:μ_p] * θ[:μ_w]), θ[:α], θ[:τ_lev], θ[:τ_prog], θ[:γ])
Y_GAP                   = _bbl_output(K_t, Z_t, N_GAP, θ[:α])

YREACTION = Ygrowth_t

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
distr_y                 = sum(nt[:distr_t], dims=(1,2)) # TODO: can we replace this with nt[:marginal_pdf_y_t]??
# Htact                   = dot(distr_y[1:end-1], (y_grid[1:end-1] ./ H).^((θ[:γ] + θ[:τ_prog]) / (θ[:γ] + τ_prog_t)))
Htact                   = sum(distr_y[1:end-1] .* (y_grid[1:end-1] ./ H).^((θ[:γ] + θ[:τ_prog]) / (θ[:γ] + τ_prog_t)))

############################################################################
#           Error term calculations (i.e. model starts here)          #
############################################################################

#-------- States -----------#
# Error Term on exogeneous States
# Shock processes
F[eq_G]       = log.(G_sh′_t)         - θ[:ρ_G] * log.(G_sh_t)     # primary deficit shock
F[eq_P]   = log.(P_sh′_t)         - θ[:ρ_P_sh] * log.(P_sh_t) # tax shock

F[eq_R]       = log.(R_sh′_t)         - θ[:ρ_R_sh] * log.(R_sh_t)     # Taylor rule shock
F[eq_S]       = log.(S_sh′_t)         - θ[:ρ_S_sh] * log.(S_sh_t)     # uncertainty shock

# Stochastic states that can be directly moved (no feedback)
F[eq_A]            = log.(A′_t)              - θ[:ρ_A] * log.(A_t)               # (unobserved) Private bond return fed-funds spread (produces goods out of nothing if negative)
F[eq_Z]            = log.(Z′_t)              - θ[:ρ_Z] * log.(Z_t)               # TFP
F[eq_Ψ]           = log.(Ψ′_t)              - θ[:ρ_Ψ] * log.(Ψ_t)             # Investment-good productivity

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
                         ((1 - θ[:ρ_R]) * θ[:θ_Y]) * log(YREACTION) -
                         θ[:ρ_R] * (log(RB_t) - nt[:RB_t])  - log(R_sh_t)

# Tax rule
F[eq_tax_progressivity]        = log(τ_prog_t) - θ[:ρ_P] * log(τ_prog_t1)  - # TODO: find correct name of τ_prog_lag_t1
                         (1.0 - θ[:ρ_P]) * (nt[:τ_prog_t]) -
                         (1.0 - θ[:ρ_P]) * θ[:γ_Y_P] * log(YREACTION) -
                         (1.0 - θ[:ρ_P]) * θ[:γ_B_P] * (log(B_t)- nt[:B_t]) -
                         log(P_sh_t)

#=F[eq_tax_level]         = avg_tax_rate_t - dot(nt[:distr_t], taxrev) / dot(nt[:distr_t], incgrossaux) # Union profits are taxed at average tax rate

F[eq_tax_revenue]            = log(T_t) - log(dot(nt[:distr_t], taxrev) + avg_tax_rate_t * union_profits_t)=#
F[eq_tax_level]         = avg_tax_rate_t - sum(nt[:distr_t] .* taxrev) / sum(nt[:distr_t] .* incgrossaux) # Union profits are taxed at average tax rate
#=
println("av tax rate")
println(avg_tax_rate_t)
println("union profits")
println(union_profits_t)
println("T_t")
println(T_t)
=#
F[eq_tax_revenue]            = log(T_t) - log(sum(nt[:distr_t] .* taxrev) + avg_tax_rate_t * union_profits_t)

F[eq_avg_tax_rate]  = log(avg_tax_rate_t) - θ[:ρ_τ] * log(avg_tax_rate_t1)  -
                            (1.0 - θ[:ρ_τ]) * nt[:avg_tax_rate_t] -
                            (1.0 - θ[:ρ_τ]) * θ[:γ_Y_τ] * log(YREACTION) -
                            (1.0 - θ[:ρ_τ]) * θ[:γ_B_τ] * (log(B_t) - nt[:B_t])

# --------- Controls ------------
# Deficit rule
F[eq_deficit_rule]            = log(Bgrowth′_t) + θ[:γ_B] * (log(B_t) - nt[:B_t])  -
                          θ[:γ_Y] * log(YREACTION)  - θ[:γ_π] * log(π_t) - log(G_sh_t)

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

#F[eq_union_firm_profits]   = log(union_firm_profits_t)  - log(w_t * N_t * (1.0 - mc_w_t))  # profits of the monopolistic unions
F[eq_union_profits]         = log(union_profits_t)        - log(w_t.*N_t.*(1-mc_w_t)) # distributed profits to households
#F[eq_union_retained]       = log(union_retained′_t) - (union_firm_profits_t - union_profits_t + log(union_retained_t) * (RB_t / π_t)) # Accumulation equation, retained is in levels

#F[eq_firm_profits]         = log(firm_profits_t)  - log(Y_t * (1.0 - mc_t))                                               # profits of the monopolistic resellers
#F[eq_profits_distr_to_hh]              = log(profits_t)       - log((1.0 - exp(nt[:mc_t])) * Y_t + θ[:ω_F] *


F[eq_profits_distr_to_hh]              = log(profits_t)       - log((1.0 - mc_t) .* Y_t .+ q_t .*(K′_t .- (1.0 - depr) .*K_t).- I_t) # distributed profits to households
#                                                        (firm_profits_t - (1.0 - exp(nt[:mc_t])) * Y_t + log(retained_t))) # distributed profits to households
#F[eq_retained]             = log(retained′_t) - (firm_profits_t - profits_t + log(retained_t) * (RB_t / π_t))            # Accumulation equation (levels)

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
#F[eq_resource_constraint]            = log(Y_t - G_t - I_t - BD_t * θ[:Rbar] + (A_t - 1.0) * RB_t * B_t / π_t -
                              # (δ_1 * (u_t - 1.0) + δ_2 / 2.0 * (u_t - 1.0)^2.0) * K_t ) - log(C_t) # Resource constraint
#println("testing new equation")
F[eq_resource_constraint]            = log(Y_t - G_t - I_t - BD_t * θ[:Rbar] + (A_t - 1.0) * RB_t * B_t / π_t)  - log(C_t) # Resource constraint

# Error Term on prices/aggregate summary vars (logarithmic, controls), here difference to SS value averages
F[eq_capital_market_clear]            = log(K_t)     - nt[:K_t]                                                            # Capital market clearing
F[eq_debt_market_clear]           = log(BD_t)    - nt[:BD_t]                                                        # IOUs
F[eq_bond_market_clear]            = log(B_t)     - nt[:B_t]   #+ log(retained_t) + log(union_retained_t) )   # Bond market clearing
F[eq_bond_output_ratio]           = log(BY_t)    - log(B_t / Y_t)                                                               # Bond to Output ratio
F[eq_tax_output_ratio]           = log(TY_t)    - log(T_t / Y_t)                                                               # Tax to output ratio
#F[eq_retained_earnings_gdp_ratio] = log(tot_retained_Y_t) - ((log(retained_t) + log(union_retained_t)) / Y_t)                      # retained Earnings to GDP

# Add distributional summary stats that do change with other aggregate controls/prices so that the stationary
F[eq_Ht]           = log(Ht_t) - log(Htact)
