function original_Fsys_agg(X::AbstractArray, XPrime::AbstractArray, # distrSS::AbstractArray, # TODO aggregate equations currently assume it's in nt
                  θ::NamedTuple, grids::OrderedDict, id::OrderedDict{Symbol, I1}, nt::NamedTuple,
                  eq::OrderedDict{Symbol, I2}) where {I1 <: Union{Int, UnitRange}, I2 <: Union{Int, UnitRange}}
              # The function call with Duals takes
              # Reserve space for error terms

    F = zeros(eltype(X),size(X)) # TODO replace with undef

    ############################################################################
    #            I. Read out argument values                                   #
    ############################################################################

    ############################################################################
    # I.1. Generate code that reads aggregate states/controls
    #      from steady state deviations. Equations take the form of:
    # r       = exp.(Xss[indexes.rSS] .+ X[indexes.r])
    # rPrime  = exp.(Xss[indexes.rSS] .+ XPrime[indexes.r])
    ############################################################################

    # Today
    #@sslogdeviations2levels union_retained_t, retained_t = X, id, nt
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
    #@sslogdeviations2levels tot_retained_Y_t, union_firm_profits_t = X, id, nt
    @sslogdeviations2levels union_profits_t, profits_t = X, id, nt

    # Tomorrow # NOTE that we use XPrime, so id[:C_t] and id[:C_t] should point to the same indices
    #@sslogdeviations2levels_unprimekeys union_retained′_t, retained′_t = XPrime, id, nt
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
    #@sslogdeviations2levels_unprimekeys tot_retained_Y′_t, union_firm_profits′_t = XPrime, id, nt
    @sslogdeviations2levels_unprimekeys union_profits′_t, profits′_t = XPrime, id, nt

    # Some special handling for type stability
    y_grid   = get_gridpts(grids, :y_grid)::Vector{Float64}
    y_ndgrid = grids[:y_ndgrid]::Array{Float64, 3}
    H        = grids[:H]::Float64

    # Read out equation scalar indices
    @unpack_and_first eq_mp, eq_tax_progressivity, eq_tax_level = eq
    @unpack_and_first eq_tax_revenue, eq_avg_tax_rate, eq_deficit_rule = eq
    @unpack_and_first eq_gov_budget_constraint, eq_price_phillips_curve = eq
    @unpack_and_first eq_wage_phillips_curve, eq_real_wage_inflation, eq_capital_util = eq
    @unpack_and_first eq_capital_return, eq_received_wages, eq_wages_firms_pay = eq
    @unpack_and_first eq_union_profits = eq
    @unpack_and_first eq_profits_distr_to_hh = eq
    @unpack_and_first eq_tobins_q, eq_expost_liquidity_premium = eq
    @unpack_and_first eq_exante_liquidity_premium, eq_capital_accum, eq_labor_supply = eq
    @unpack_and_first eq_output, eq_resource_constraint, eq_capital_market_clear = eq
    @unpack_and_first eq_debt_market_clear, eq_bond_market_clear, eq_bond_output_ratio = eq
    @unpack_and_first eq_tax_output_ratio, eq_Ht = eq
    @unpack_and_first eq_Ygrowth, eq_Tgrowth, eq_Bgrowth = eq
    @unpack_and_first eq_Igrowth, eq_wgrowth, eq_Cgrowth = eq
    @unpack_and_first eq_LY, eq_LB, eq_LI, eq_Lw, eq_LT, eq_Lq, eq_LC = eq
    @unpack_and_first eq_Lavg_tax_rate, eq_Lτ_prog = eq
    @unpack_and_first eq_A, eq_Z, eq_Ψ, eq_μ_p, eq_μ_w, eq_σ, eq_G, eq_P, eq_R, eq_S = eq

    #println("testing new modification y update")
    # Take aggregate model from equation file
    @include "original_aggregate_equations.jl"
    #@include("original_aggregate_equations.jl")
    #include("original_aggregate_equations.jl")

    return F
end
