function _bbl_incomes(θ::NamedTuple, grids::OrderedDict, KSS::Real, distrSS::Array{<: Real, 3})

    # Set up for type stability
    m_ndgrid = grids[:m_ndgrid]::Array{Float64, 3}
    k_ndgrid = grids[:k_ndgrid]::Array{Float64, 3}
    y_ndgrid = grids[:y_ndgrid]::Array{Float64, 3}
    H         = grids[:H]::Float64
    HW        = grids[:HW]::Float64

    NSS       = _bbl_employment(KSS, 1.0 / (θ[:μ_p] * θ[:μ_w]), θ[:α], θ[:τ_lev], θ[:τ_prog], θ[:γ])
    rkSS      = _bbl_interest(KSS, 1.0 / θ[:μ_p], NSS, θ[:α], θ[:δ_0])
    wSS       = _bbl_wage(KSS, 1.0 / θ[:μ_p], NSS, θ[:α])
    YSS       = _bbl_output(KSS, 1.0, NSS, θ[:α])
    ProfitsSS = (1.0 - 1.0 / θ[:μ_p]) * YSS
    ISS       = θ[:δ_0] * KSS
    RBSS      = θ[:RB]
    GHHFA     = ((θ[:γ] + θ[:τ_prog]) / (θ[:γ] + 1.0))      # transformation (scaling) for composite good

    neg_liq_r = RBSS + θ[:Rbar]
    eff_int   = [x <= 0. ? neg_liq_r : RBSS for x in m_ndgrid] # effective rate depending on assets

    incgross = Array{Array{Float64, 3}}(undef, 5) # gross income
    inc = Array{Array{Float64, 3}}(undef, 6)      # net (of taxes) income

    mcw = 1.0 ./ θ[:μ_w]

    gross_labor_income       = (y_ndgrid .* ((1. / θ[:μ_w]) * wSS * NSS / H))
    gross_labor_union_income = (1.0 - 1.0 / θ[:μ_w]) * wSS * NSS * HW
    incgross[1] = gross_labor_income .+ gross_labor_union_income # labor income (NEW)
    incgross[2] = rkSS .* k_ndgrid                               # rental income
    incgross[3] = eff_int .* m_ndgrid                            # liquid asset Income
    incgross[4] = k_ndgrid                                       # capital liquidation Income (q=1 in steady state)
    incgross[5] = gross_labor_income

    ny                     = size(y_ndgrid, 3)
    incgross[1][:, :, end] .= view(y_ndgrid, :, :, ny) .* ProfitsSS   # profit income net of taxes
    incgross[5][:, :, end] .= incgross[1][:, :, end]                   # this copies b/c [:, :, end] allocates a new matrix

    # TODO: remove this redundant calculation of inc[1] here and then recalculation later with union profits.
    # Just use calculation later on since inc[1] is not used at all
    labor_income_preGHHFA = θ[:τ_lev] .* gross_labor_income .^ (1.0 - θ[:τ_prog])
    labor_income = GHHFA .* labor_income_preGHHFA
    inc[1] = labor_income .+ gross_labor_union_income # labor income (NEW)
    inc[2] = incgross[2]                                                           # rental income
    inc[3] = incgross[3]                                                           # liquid asset income
    inc[4] = incgross[4]                                                           # points to ndgrid but won't be copying it
    inc[6] = labor_income_preGHHFA           # capital liquidation Income (q=1 in steady state)
    inc[5] = inc[6] .* ((1.0 - θ[:τ_prog]) / (θ[:γ] + 1))
    inc[1][:, :, end] .= θ[:τ_lev] .* view(incgross[1], :, :, ny) .^ (1.0 - θ[:τ_prog])             # profit income net of taxes
    inc[5][:, :, end] .= 0.0
    inc[6][:, :, end] .= inc[1][:, :, end]                                   # this copies b/c [:, :, end] allocates a new matrix

    taxrev        = incgross[5] - inc[6]
    tot_taxrev    = dot(distrSS, taxrev) # if a DimensionMismatch error occurs, it may be because you are using the wrong coarseness
    av_tax_rateSS = tot_taxrev / dot(distrSS, incgross[5]) # of the grid (e.g. you need to recall `init_grids!(m; coarse = ...)`

    # apply taxes to union profits
    labor_union_income = gross_labor_union_income .* (1. .- av_tax_rateSS)
    inc[1]             = labor_income .+ labor_union_income
    inc[1][:, :, end] .= inc[6][:, :, end]
    inc[6]             = labor_income_preGHHFA .+ labor_union_income
    inc[6][:, :, end] .= inc[1][:, :, end]

    return incgross, inc, NSS, rkSS, wSS, YSS, ProfitsSS, ISS, RBSS, taxrev, tot_taxrev, av_tax_rateSS, eff_int
end
