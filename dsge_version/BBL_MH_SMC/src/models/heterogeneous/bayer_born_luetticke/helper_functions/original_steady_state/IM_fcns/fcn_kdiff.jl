function original_Kdiff(K_guess::Float64, m::BayerBornLuetticke{T1},
                        initial::Bool = true, Vm_guess::AbstractArray = zeros(1, 1, 1),
                        Vk_guess::AbstractArray = zeros(1, 1, 1), distr_guess::AbstractArray = zeros(1, 1, 1);
                        verbose::Symbol = :none, coarse::Bool = false) where {T1 <: Real}
#=
    println("K_guess")
    println(K_guess)
    println("Vk_guess norm")
    println(norm(Vk_guess))
    println("distr_guess norm")
    println(norm(distr_guess))
    println("Vm_guess norm")
    println(norm(Vm_guess))
    #println("test change")
    #println("Transitions EigVal")
    #println(m.grids[:Π])
=#
    # Some type declarations b/c grids is an OrderedDict
    # => ensures type stability, or else unnecessary allocations are made
    Π                   = m.grids[:Π]::Matrix{T1}
    y_grid              = get_gridpts(m, :y_grid)::Vector{T1}
    m_ndgrid            = m.grids[:m_ndgrid]::Array{T1, 3}
    k_ndgrid            = m.grids[:k_ndgrid]::Array{T1, 3}
    y_ndgrid            = m.grids[:y_ndgrid]::Array{T1, 3}
    H                   = m.grids[:H]::T1
    HW                  = m.grids[:HW]::T1

    @save "save4.jld2" Π y_grid m_ndgrid k_ndgrid y_ndgrid H HW

    # TODO: check if there's a notable speed up by only passing settings, grids,
    #       and parameters into this function as kwargs
    #----------------------------------------------------------------------------
    # Calculate other prices from capital stock
    #----------------------------------------------------------------------------
   #= N           = _bbl_employment(K_guess, 1.0 / (m[:μ_p] * m[:μ_w]), m[:α],      # employment
                                  m[:τ_lev], m[:τ_prog], m[:γ])
    w           = _bbl_wage(K_guess, 1.0 / m[:μ_p], N, m[:α])                     # wages
    rk          = _bbl_interest(K_guess, 1.0 / m[:μ_p], N, m[:α], m[:δ_0])        # Return on illiquid asset
    profits     = (1.0 - 1.0 / m[:μ_p]) .* _bbl_output(K_guess, 1.0, N, m[:α])    # Profit income
    neg_liq_ret = RB + m[:Rbar]
    eff_int     = [x <= 0. ? neg_liq_ret : RB for x in m_ndgrid]        # effective rate depending on assets
    RB          = m[:RB] ./ m[:π]                                                  # Real return on liquid assets=#


    N           = _original_bbl_employment(K_guess, 1.0 ./ (m[:μ_p] * m[:μ_w]), m[:α],      # employment
                                  m[:τ_lev], m[:τ_prog], m[:γ])
    w           = _original_bbl_wage(K_guess, 1.0 ./ m[:μ_p], N, m[:α])                     # wages


#ADDED THE 1.0
    rk          = _original_bbl_interest(K_guess, 1.0 ./ m[:μ_p], N, m[:α], m[:δ_0])        # Return on illiquid asset
    profits     = (1.0 .- 1.0 ./ m[:μ_p]) .* _original_bbl_output(K_guess, 1.0, N, m[:α])    # Profit income
    RB          = m[:RB] ./ m[:π]                                                  # Real return on liquid assets

    eff_int     = (RB .+ m[:Rbar] .* (m_ndgrid .<= 0.0))

    GHHFA       = (m[:γ] + m[:τ_prog]) / (m[:γ] + 1.0)                            # transformation (scaling) for composite good

    #----------------------------------------------------------------------------
    # Array (inc) to store incomes
    # inc[1] = labor income , inc[2] = rental income,
    # inc[3]= liquid assets income, inc[4] = capital liquidation income
    #----------------------------------------------------------------------------
    # Paux            = m.grids[:Paux]::Matrix{T1}                                 # Grab ergodic income distribution from transitions
    Paux = Π^1000
    distr_y         = Paux[1, :]                                                 # stationary income distribution
    inc             = Array{Array{Float64, 3}}(undef, 4)                         # container for income
    # mcw             = 1.0 / m[:μ_w]                                              # wage markup
    mcw             = 1.0 ./ m[:μ_w]                                              # wage markup

    # gross (labor) incomes
#=    incgross        = y_grid .* (mcw * w * N / H)      # gross income workers (wages)
    incgross[end]   = y_grid[end] * profits                     # gross income entrepreneurs (profits)=#
    incgross        = y_grid .* mcw .* w .* N ./ H      # gross income workers (wages)
    incgross[end]   = y_grid[end] .* profits                     # gross income entrepreneurs (profits)

    # net (labor) incomes
    # incnet          = m[:τ_lev] * incgross .^ (1.0 - m[:τ_prog])
    incnet          = m[:τ_lev] .* (mcw .* w .* N ./ H .* y_grid).^(1. - m[:τ_prog])
    incnet[end]     = m[:τ_lev] .* (y_grid[end] .* profits).^(1. - m[:τ_prog])


    # average tax rate
    # av_tax_rate     = dot((incgross - incnet), distr_y) / dot(incgross, distr_y)
    av_tax_rate     = dot((incgross - incnet), distr_y) ./ dot(incgross, distr_y)


    # TODO: replace the y_ndgrid calculation with just repeating the incnet vector OR use list comprehension later on
    ny              = get_setting(m, coarse ? :coarse_ny : :ny)
#=    inc[1]          = (GHHFA * m[:τ_lev]) .* (y_ndgrid .* (mcw * w * N / H)) .^ (1.0 - m[:τ_prog]) .+
        ((1.0 - mcw) * w * N * (1.0 - av_tax_rate) * HW)         # labor income net of taxes incl. union profits
    inc[1][:,:,end] = m[:τ_lev] * (view(y_ndgrid, :, :, ny) * profits) .^ (1.0 - m[:τ_prog]) # profit income net of taxes=#
    inc[1]          = GHHFA .* m[:τ_lev] .* (y_ndgrid .* mcw .* w .* N ./ H) .^ (1.0 - m[:τ_prog]) .+
        (1.0 .- mcw) .* w .* N .* (1.0 .- av_tax_rate) .* HW         # labor income net of taxes incl. union profits
    inc[1][:,:,end] = m[:τ_lev] .* (y_ndgrid[:, :, end] * profits).^(1.0 - m[:τ_prog]) # profit income net of taxes

    # incomes out of wealth # TODO: replace these steps OR use list comprehension later on
    inc[2]          = rk .* k_ndgrid                                  # rental income
    inc[3]          = eff_int .* m_ndgrid                             # liquid asset income
    inc[4]          = k_ndgrid                                        # capital liquidation income (q=1 in steady state)

    #----------------------------------------------------------------------------
    # Initialize policy function (guess/stored values)
    #----------------------------------------------------------------------------

#=
var1 = inc[1]
var2 = inc[2]
var3 = inc[3]var4 = inc[4]

@save "save3.jld2" var1 var2 var3 var4
=#
    # initial guess consumption and marginal values (if not set)
    if initial # TODO: pass in m_ndgrid .> 0. if that's already calculated elsewhere
        # c_guess     = inc[1] .+ inc[2] .* (inc[2] .> 0) .+ inc[3] .* (m_ndgrid .> 0.)
        c_guess     = inc[1] .+ inc[2] .* (k_ndgrid .* rk .> 0.) .+ inc[3] .* (m_ndgrid .> 0.)
        if any(x -> x < 0., c_guess)
            @warn "negative consumption guess"
        end
        Vm          = eff_int .* _bbl_mutil(c_guess, m[:ξ])
        Vk          = (rk + m[:λ]) .* _bbl_mutil(c_guess, m[:ξ])
        distr       = get_untransformed_values(m[:distr_star])::Array{T1, 3}
    else
        Vm          = Vm_guess
        Vk          = Vk_guess
        distr       = distr_guess
    end

    #----------------------------------------------------------------------------
    # Calculate supply of funds for given prices
    #----------------------------------------------------------------------------
    KS              = original_Ksupply(RB, 1.0 + rk, m, Vm, Vk, distr,
                                       inc, eff_int; verbose = verbose, coarse = coarse)
    K               = KS[1]                                                     # capital
    Vm              = KS[end-2]                                                 # marginal value of liquid assets
    Vk              = KS[end-1]                                                 # marginal value of illiquid assets
    distr           = KS[end]                                                   # stationary distribution
    diff            = K - K_guess                                               # excess supply of funds
    return diff, Vm, Vk, distr
end
