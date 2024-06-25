"""
```
Fsys(X,XPrime,Xss,m_par,n_par,indexes,Γ,compressionIndexes,DC,IDC)
```
returns deviations from equilibrium around the steady state.

The computation is split into the aggregate block, handled by [`Fsys_agg()`](@ref),
and heterogeneous block.

### Inputs
- `X::AbstractArray`,`XPrime::AbstractArray`: deviations from steady state in periods t [`X`] and t+1 [`XPrime`]
- `θ::NamedTuple`: maps names of economic parameters to their values
- `grids::OrderedDict`: maps names of quantities related to the idiosyncratic state space to their values (e.g. `m_grid`)
- `id::OrderedDict{Symbol, UnitRange{Int}}`: maps variable name (e.g. `Y_t` and `Y′_t`)
    to its indices in `X` and `XPrime`.
- `nt::NamedTuple`: maps variable name (e.g. `Y_t` and `Y′_t`) to steady-state log level
- `eq::OrderedDict{Symbol, UnitRange{Int}}`: maps name of equilibrium conditions
    to its indices in `F`
- `Γ::Vector{Matrix{Float64}}`: transformation matrices to retrieve marginal distributions from deviations
- `DC::Vector{Matrix{Float64}}`,
    `IDC::Vector{Adjoint{Float64,Matrix{Float64}}}`: transformation matrices to retrieve marginal value functions from deviations
- `DCD::Vector{Matrix{Float64}}`,
    `IDCD::Vector{Adjoint{Float64,Matrix{Float64}}}`: transformation matrices to retrieve copula from deviations
"""
function Fsys(F::AbstractVector, X::AbstractArray, XPrime::AbstractArray, θ::NamedTuple, grids::OrderedDict,
              id::AbstractDict, nt::NamedTuple, eqconds::OrderedDict{Symbol, UnitRange{Int}},
              dct_compression_indices::Dict{Symbol, Vector{Int}},
              Γ::Array{Array{Float64,2},1}, DC::Array{Array{Float64,2},1},
              IDC::Array{Adjoint{Float64,Array{Float64,2}},1}, DCD::Array{Array{Float64,2},1},
              IDCD::Array{Adjoint{Float64,Array{Float64,2}},1})
              # The function call with Duals takes
              # Reserve space for error terms

    ############################################################################
    #            I. Read out argument values                                   #
    ############################################################################
    # rougly 10% of computing time, more if uncompress is actually called

    # Some initial set up
    m_ndgrid = grids[:m_ndgrid]::Array{Float64, 3}
    k_ndgrid = grids[:k_ndgrid]::Array{Float64, 3}
    y_ndgrid = grids[:y_ndgrid]::Array{Float64, 3}
    m_grid = get_gridpts(grids, :m_grid)::Vector{Float64}
    k_grid = get_gridpts(grids, :k_grid)::Vector{Float64}
    y_grid = get_gridpts(grids, :y_grid)::Vector{Float64}
    HW     = grids[:HW]::Float64
    y_bin_bounds = grids[:y_bin_bounds]::Vector{Float64}
    Π = grids[:Π]::Matrix{Float64}
    nm, nk, ny = size(m_ndgrid)

    ############################################################################
    # I.1. Generate code that reads aggregate states/controls
    #      from steady state deviations. Equations take the form of:
    # r       = exp.(Xss[indexes.rSS] .+ X[indexes.r])
    # rPrime  = exp.(Xss[indexes.rSS] .+ XPrime[indexes.r])
    ############################################################################

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

    # Tomorrow # NOTE that we use XPrime, so id[:C_t] and id[:C′_t] should point to the same indices
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

    ############################################################################
    # I.2. Distributions (Γ-multiplying makes sure that they are distributions)
    ############################################################################

    # Get perturbation to distribution implied by perturbation to DCT coefs
    θD      = uncompressD(dct_compression_indices[:copula], X[id[:copula_t]], DCD, IDCD, (nm, nk, ny))

    # Adjust distribution perturbation to ensure the distribution sums to one # TODO: check if we canuse view
    DISTRAUX = zeros(eltype(θD), nm, nk, ny)      # Use [nm, 1:end-1, 1:end-1], [:, nk, 1:end-1], [:, :, ny] to guarantee
    θDaux = reshape(θD, (nm - 1, nk - 1, ny - 1)) # the total change to the distribution is zero (hence distribution still sums to one).
    DISTRAUX[1:end-1,1:end-1,1:end-1] = θDaux     # Note that the change in the indices and the associated change in what the sum
    DISTRAUX[end,1:end-1,1:end-1] = -sum(θDaux, dims=(1)) # is computed over accounts for the previous adjustments,
    DISTRAUX[:,end,1:end-1] = -sum(DISTRAUX[:, :, 1:end-1], dims=(2)) # e.g. -sum(DISTRAUX[:, :, 1:end-1], dims=2) accounts
    DISTRAUX[:,:,end] = -sum(DISTRAUX, dims=(3)) # for the -sum(θDaux, dims=1) adjustment to avoid double-counting

    DISTR = nt[:distr_t] + DISTRAUX # DISTRAUX describes the perturbation to the copula implied by perturbations to DCT coefs
    DISTR = max.(DISTR, 1e-16)      # while ensuring the distribution sums to one
    DISTR = DISTR ./ sum(DISTR) # renormalize
    DISTR = cumsum(cumsum(cumsum(DISTR;dims=3);dims=2);dims=1) # compute CDF

    # Perturb the marginal PDFs, while using Γ to ensure marginals sum to one
    distr_m       = nt[:marginal_pdf_m_t] .+ Γ[1] * X[id[:marginal_pdf_m_t]]
    distr_m_Prime = nt[:marginal_pdf_m_t] .+ Γ[1] * XPrime[id[:marginal_pdf_m′_t]]
    distr_k       = nt[:marginal_pdf_k_t] .+ Γ[2] * X[id[:marginal_pdf_k_t]]
    distr_k_Prime = nt[:marginal_pdf_k_t] .+ Γ[2] * XPrime[id[:marginal_pdf_k′_t]]
    distr_y       = nt[:marginal_pdf_y_t] .+ Γ[3] * X[id[:marginal_pdf_y_t]]
    distr_y_Prime = nt[:marginal_pdf_y_t] .+ Γ[3] * XPrime[id[:marginal_pdf_y′_t]]

    # Joint distributions (uncompressing), needed to use the copula
    CDF_m         = cumsum([0.0; vec(distr_m)])
    CDF_k         = cumsum([0.0; vec(distr_k)])
    CDF_y         = cumsum([0.0; vec(distr_y)])

    # Construct the copula from the perturbed distribution => perturbed copula
    cum_zero = zeros(eltype(θD), nm + 1, nk + 1, ny + 1)
    cum_zero[2:end,2:end,2:end] = DISTR # add the CDF
    Copula1(x::AbstractVector,y::AbstractVector,z::AbstractVector) =
        mylinearinterpolate3(cum_zero[:,end,end], cum_zero[end,:,end], cum_zero[end,end,:], cum_zero, x, y, z) # TODO: check if I can use view
    #Copula1 = LinearInterpolation((cum_zero[:,end,end],cum_zero[end,:,end],cum_zero[end,end,:]),cum_zero,extrapolation_bc=Line())

    # Compute the distribution implied by the perturbed marginals and copula
    CDF_joint     = Copula1(vec(CDF_m), vec(CDF_k), vec(CDF_y)) # roughly 5% of time
    distr         = diff(diff(diff(CDF_joint; dims=3);dims=2);dims=1)

    ############################################################################
    # I.3 uncompressing policies/value functions
    ###########################################################################
    if any((tot_dual.(XPrime[id[:Vm′_t]]) + realpart.(XPrime[id[:Vm′_t]])) .!= 0.0)
        θm      = uncompress(dct_compression_indices[:Vm], XPrime[id[:Vm′_t]], DC, IDC, (nm, nk, ny))
        VmPrime = vec(nt[:Vm_t]) + θm
    else
         VmPrime = vec(nt[:Vm_t]) .+ zero(eltype(X))
    end
    VmPrime .= (exp.(VmPrime))

     if any((tot_dual.(XPrime[id[:Vk′_t]]) + realpart.(XPrime[id[:Vk′_t]])) .!= 0.0)
        θk      = uncompress(dct_compression_indices[:Vk], XPrime[id[:Vk′_t]], DC, IDC, (nm, nk, ny))
        VkPrime = vec(nt[:Vk_t]) + θk
     else
         VkPrime = vec(nt[:Vk_t]) .+ zero(eltype(X))
     end
    VkPrime .= (exp.(VkPrime))

    ############################################################################
    #           II. Auxiliary Variables                                        #
    ############################################################################
    # Transition Matrix Productivity
    if tot_dual.(σ_t .+ zero(eltype(X))) == 0.0
        if σ_t == 1.0
            Π                  = Π .+ zero(eltype(X))
        else
            Π                  = Π
            PP                 = ExTransition(θ[:ρ_h], y_bin_bounds, sqrt(σ_t))
            Π[1:end-1,1:end-1] = PP.*(1.0 - θ[:ζ])
        end # TODO: if using Rouwenhorst method, this step is not necessary b/c perturbing σ doesn't change the transition productivity
    else
        Π                  = Π .+ zero(eltype(X))
        PP                 = ExTransition(θ[:ρ_h], y_bin_bounds, sqrt(σ_t))
        Π[1:end-1,1:end-1] = PP.*(1.0 - θ[:ζ])
    end

    ############################################################################
    #           III. Error term calculations (i.e. model starts here)          #
    ############################################################################

    ############################################################################
    #           III. 1. Aggregate Part #
    ############################################################################
    #Fsys_agg(F, X, XPrime, θ, grids, id, nt, eqconds) # Fsys_agg(X, XPrime, θ, grids, id, nt, eqconds)

    F = Fsys_agg(F, X, XPrime, θ, grids, id, nt, eqconds)
    # Error Term on prices/aggregate summary vars (logarithmic, controls)
    KP           = dot(k_grid, distr_k)
    F[first(eqconds[:eq_capital_market_clear])] = log.(K_t)     - log.(KP)
    BP           = dot(m_grid, distr_m)
    F[first(eqconds[:eq_bond_market_clear])] = log.(B_t)     - log.(BP)

    BDact = -dot(distr_m, (m_grid .< 0.) .* m_grid)

    F[first(eqconds[:eq_debt_market_clear])] = log.(BD_t)  - log.(BDact)

    # Average Human Capital =
    # average productivity (at the productivity grid, used to normalize to 0)
    H       = dot(view(distr_y, 1:ny-1), view(y_grid, 1:ny-1))

    ############################################################################
    #               III. 2. Heterogeneous Agent Part                           #
    ############################################################################
    # Incomes
    nonpos_ret   = θ[:Rbar] .* (m_ndgrid .<= 0.0)
    eff_int      = ((RB_t * A_t) .+ nonpos_ret) ./ π_t # effective rate (need to check timing below and inflation)
    eff_intPrime = ((RB′_t * A′_t) .+ nonpos_ret) ./ π′_t

    GHHFA                    = ((θ[:γ] + τ_prog_t) / (θ[:γ] + 1.)) # transformation (scaling) for composite good
    tax_prog_scale           = (θ[:γ] + θ[:τ_prog]) / ((θ[:γ] + τ_prog_t))
    y_ndgrid_rel_H           = y_ndgrid ./ H
    wages_times_labor        = mc_w_t * w_t * N_t
    gross_labor_income       = (y_ndgrid_rel_H .^ tax_prog_scale) .* wages_times_labor
    gross_labor_income_rel_Ht       = gross_labor_income ./ Ht_t
    labor_income_rel_Ht = τ_level_t .* (gross_labor_income_rel_Ht) .^ (1. - τ_prog_t)
    entrep_profits           = view(y_ndgrid, :, :, ny) .* profits_t
    entrep_profits_net_taxes = τ_level_t .* entrep_profits.^(1.0 - τ_prog_t) # profit income net of taxes
    inc = [  GHHFA .* labor_income_rel_Ht .+ (union_profits_t) .* (1.0 - avg_tax_rate_t) .* HW, # labor income (NEW)
             (rk_t - 1.0) .* k_ndgrid, # rental income
             eff_int .* m_ndgrid, # liquid asset Income
             k_ndgrid .* q_t,
             τ_level_t .* (wages_times_labor .* y_ndgrid_rel_H).^(1.0 - τ_prog_t) .* ((1.0 - τ_prog_t) / (θ[:γ] + 1)),
             labor_income_rel_Ht] # capital liquidation Income (q=1 in steady state)
    inc[1][:,:,end] .= entrep_profits_net_taxes
    inc[5][:,:,end] .= 0.0
    inc[6][:,:,end] .= entrep_profits_net_taxes

    incgross =[  gross_labor_income_rel_Ht .+ union_profits_t,
                 (rk_t .- 1.0) .* k_ndgrid,                                      # rental income # TODO: can we copy from inc?
                 eff_int .* m_ndgrid,                                        # liquid asset Income
                 k_ndgrid .* q_t,
                 gross_labor_income_rel_Ht]           # capital liquidation Income (q=1 in steady state)
    incgross[1][:,:,end] .= entrep_profits
    incgross[5][:,:,end] .= entrep_profits

    taxrev      = incgross[5]-inc[6] # tax revenues w/o tax on union profits
    incgrossaux = incgross[5]
    tot_taxrev  = dot(distr, taxrev)
    F[first(eqconds[:eq_tax_level])] = avg_tax_rate_t - tot_taxrev / dot(distr, incgrossaux)
    F[first(eqconds[:eq_tax_revenue])]    = log(T_t) - log(tot_taxrev + avg_tax_rate_t * (union_profits_t))

    inc[6] = labor_income_rel_Ht .+ ((1.0 .- mc_w_t) .* w_t .* N_t) .* (1.0 .- avg_tax_rate_t)
    inc[6][:,:,end] .= entrep_profits_net_taxes

    # Calculate optimal policies
    # make allocations to populate later
    Vm_new   = Array{eltype(X),3}(undef, nm, nk, ny)
    Vk_new   = similar(Vm_new)
    c_a_star = similar(Vm_new)
    m_a_star = similar(Vm_new)
    k_a_star = similar(Vm_new)
    c_n_star = similar(Vm_new)
    m_n_star = similar(Vm_new)

    # expected marginal values
    joined_mk_dims = (nm * nk, ny)
    EVkPrime = reshape(VkPrime, (nm, nk, ny))
    EVmPrime = reshape(VmPrime, (nm, nk, ny))
    EVmPrime = reshape((reshape(eff_intPrime, joined_mk_dims) .*
                        reshape(EVmPrime, joined_mk_dims)) * Π', (nm, nk, ny))
    EVkPrime = reshape(reshape(EVkPrime, joined_mk_dims) * Π', (nm, nk, ny))

    # policy iteration
    EGM_policyupdate!(EVmPrime, EVkPrime, q_t, π_t, RB_t * A_t, 1.0, inc, θ, grids, false,
                      c_a_star, m_a_star, k_a_star, c_n_star, m_n_star)

    # Update marginal values
    updateV!(Vm_new, Vk_new, EVkPrime, c_a_star, c_n_star, m_n_star,
             rk_t - 1.0, q_t, θ, m_grid, Π) # update expected marginal values time t


@show size(DC)

    # Calculate error terms on marginal values
    Vm_err        = log.((Vm_new)) - nt[:Vm_t]
    Vm_thet       = compress(dct_compression_indices[:Vm], Vm_err, DC, IDC, (nm, nk, ny))
    F[eqconds[:eq_marginal_value_bonds]] = X[id[:Vm_t]] - Vm_thet

    Vk_err        = log.((Vk_new)) - nt[:Vk_t]
    Vk_thet       = compress(dct_compression_indices[:Vk], Vk_err, DC, IDC, (nm, nk, ny))
    F[eqconds[:eq_marginal_value_capital]] = X[id[:Vk_t]] - Vk_thet

    # Error Term on distribution (in levels, states)
    dPrime        = DirectTransition(m_a_star,  m_n_star, k_a_star, distr, θ[:λ],
                                     Π, (nm, nk, ny), m_grid, k_grid)
    dPrs          = reshape(dPrime, nm, nk, ny)
    temp          = dropdims(sum(dPrs,dims=(2,3)),dims=(2,3))
    cum_m         = cumsum(temp)
    F[eqconds[:eq_marginal_pdf_m]] = view(temp, 1:nm-1) - view(distr_m_Prime, 1:nm-1)
    temp          = dropdims(sum(dPrs,dims=(1,3)),dims=(1,3))
    cum_k         = cumsum(temp)
    F[eqconds[:eq_marginal_pdf_k]] = view(temp, 1:nk-1) - view(distr_k_Prime, 1:nk-1)
    temp          = distr_y' * Π # dropdims(sum(dPrs,dims=(1,2)),dims=(1,2))
    cum_h         = cumsum(temp')
    F[eqconds[:eq_marginal_pdf_y]] = view(temp, 1:ny-1) - view(distr_y_Prime, 1:ny-1)

    # Construct new copula of the distribution after accounting for effects of the initial perturbations
    cum_zero = zeros(eltype(θD), nm + 1, nk + 1, ny + 1)
    cum_dist_new = cumsum(cumsum(cumsum(dPrs; dims=3);dims=2);dims=1)
    cum_zero[2:end,2:end,2:end] = cum_dist_new
    Copula2(x::AbstractVector, y::AbstractVector, z::AbstractVector) = mylinearinterpolate3([0; cum_m], [0; cum_k], [0; cum_h],
                                                                                                 cum_zero, x, y, z)
    # Compute implied CDF # TODO: add marginal cdfs to the named tuple so we don't need to do this calculation all the time
    CDF_joint     = Copula2([0.0; cumsum(nt[:marginal_pdf_m_t])] .+ zero(eltype(θD)),
                            [0.0; cumsum(nt[:marginal_pdf_k_t])] .+ zero(eltype(θD)),
                            [0.0; cumsum(nt[:marginal_pdf_y_t])] .+ zero(eltype(θD))) # roughly 5% of time

    # Get implied distribution and compute the error relative to steady state
    distr_up         = diff(diff(diff(CDF_joint; dims=3);dims=2);dims=1)
    distr_err        = distr_up - nt[:distr_t]


    # Compute the DCT using the steady-state basis and calculate the change in free DCT coefficients
    D_thet       = compressD(dct_compression_indices[:copula], distr_err[1:end-1, 1:end-1, 1:end-1], DCD, IDCD, (nm, nk, ny))
    F[eqconds[:eq_copula]] =  D_thet - XPrime[id[:copula_t]]

    # Compute distributional variables
    distr_m_act, distr_k_act, distr_y_act, share_borroweract, GiniWact, I90shareact, I90sharenetact, GiniXact, #=
        =# sdlogxact, P9010Cact, GiniCact, sdlgCact, P9010Iact, GiniIact, sdlogyact, w90shareact, P10Cact, P50Cact, P90Cact =
        distrSummaries(distr, c_a_star, c_n_star, inc, incgross, θ, (nm, nk, ny), grids)

    Htact                   = dot(view(distr_y, 1:ny-1), (view(y_grid, 1:ny-1) ./ H) .^ (tax_prog_scale))
    F[first(eqconds[:eq_Ht])]           = log(Ht_t)            - log(Htact)
    F[first(eqconds[:eq_Gini_X])]        = log(Gini_X_t)         - log(GiniXact)
    F[first(eqconds[:eq_I90_share])]     = log(I90_share_t)     - log(I90shareact)
    F[first(eqconds[:eq_I90_share_net])]  = log(I90_share_net_t) - log(I90sharenetact)

    F[first(eqconds[:eq_W90_share])]     = log(W90_share_t)     - log(w90shareact)
    F[first(eqconds[:eq_sd_log_y])]       = log(sd_log_y_t)      - log(sdlogyact)
    F[first(eqconds[:eq_Gini_C])]        = log(Gini_C_t)        - log(GiniCact)

    return F
end

#=
# deprecated out-of-place version of Fsys
function Fsys(X::AbstractArray, XPrime::AbstractArray, θ::NamedTuple, grids::OrderedDict,
              id::AbstractDict, nt::NamedTuple, eqconds::OrderedDict{Symbol, UnitRange{Int}},
              dct_compression_indices::Dict{Symbol, Vector{Int}},
              Γ::Array{Array{Float64,2},1}, DC::Array{Array{Float64,2},1},
              IDC::Array{Adjoint{Float64,Array{Float64,2}},1}, DCD::Array{Array{Float64,2},1},
              IDCD::Array{Adjoint{Float64,Array{Float64,2}},1})
              # The function call with Duals takes
              # Reserve space for error terms

    F = zeros(eltype(X), size(X))

    ############################################################################
    #            I. Read out argument values                                   #
    ############################################################################
    # rougly 10% of computing time, more if uncompress is actually called

    # Some initial set up
    m_ndgrid = grids[:m_ndgrid]::Array{Float64, 3}
    k_ndgrid = grids[:k_ndgrid]::Array{Float64, 3}
    y_ndgrid = grids[:y_ndgrid]::Array{Float64, 3}
    m_grid = get_gridpts(grids, :m_grid)::Vector{Float64}
    k_grid = get_gridpts(grids, :k_grid)::Vector{Float64}
    y_grid = get_gridpts(grids, :y_grid)::Vector{Float64}
    HW     = grids[:HW]::Float64
    y_bin_bounds = grids[:y_bin_bounds]::Vector{Float64}
    Π = grids[:Π]::Matrix{Float64}
    nm, nk, ny = size(m_ndgrid)

    ############################################################################
    # I.1. Generate code that reads aggregate states/controls
    #      from steady state deviations. Equations take the form of:
    # r       = exp.(Xss[indexes.rSS] .+ X[indexes.r])
    # rPrime  = exp.(Xss[indexes.rSS] .+ XPrime[indexes.r])
    ############################################################################

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

    # Tomorrow # NOTE that we use XPrime, so id[:C_t] and id[:C′_t] should point to the same indices
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

    ############################################################################
    # I.2. Distributions (Γ-multiplying makes sure that they are distributions)
    ############################################################################

    # Get perturbation to distribution implied by perturbation to DCT coefs
    θD      = uncompressD(dct_compression_indices[:copula], X[id[:copula_t]], DCD, IDCD, (nm, nk, ny))

    # Adjust distribution perturbation to ensure the distribution sums to one # TODO: check if we canuse view
    DISTRAUX = zeros(eltype(θD), nm, nk, ny)      # Use [nm, 1:end-1, 1:end-1], [:, nk, 1:end-1], [:, :, ny] to guarantee
    θDaux = reshape(θD, (nm - 1, nk - 1, ny - 1)) # the total change to the distribution is zero (hence distribution still sums to one).
    DISTRAUX[1:end-1,1:end-1,1:end-1] = θDaux     # Note that the change in the indices and the associated change in what the sum
    DISTRAUX[end,1:end-1,1:end-1] = -sum(θDaux, dims=(1)) # is computed over accounts for the previous adjustments,
    DISTRAUX[:,end,1:end-1] = -sum(DISTRAUX[:, :, 1:end-1], dims=(2)) # e.g. -sum(DISTRAUX[:, :, 1:end-1], dims=2) accounts
    DISTRAUX[:,:,end] = -sum(DISTRAUX, dims=(3)) # for the -sum(θDaux, dims=1) adjustment to avoid double-counting

    DISTR = nt[:distr_t] + DISTRAUX # DISTRAUX describes the perturbation to the copula implied by perturbations to DCT coefs
    DISTR = max.(DISTR, 1e-16)      # while ensuring the distribution sums to one
    DISTR = DISTR ./ sum(DISTR) # renormalize
    DISTR = cumsum(cumsum(cumsum(DISTR;dims=3);dims=2);dims=1) # compute CDF

    # Perturb the marginal PDFs, while using Γ to ensure marginals sum to one
    distr_m       = nt[:marginal_pdf_m_t] .+ Γ[1] * X[id[:marginal_pdf_m_t]]
    distr_m_Prime = nt[:marginal_pdf_m_t] .+ Γ[1] * XPrime[id[:marginal_pdf_m′_t]]
    distr_k       = nt[:marginal_pdf_k_t] .+ Γ[2] * X[id[:marginal_pdf_k_t]]
    distr_k_Prime = nt[:marginal_pdf_k_t] .+ Γ[2] * XPrime[id[:marginal_pdf_k′_t]]
    distr_y       = nt[:marginal_pdf_y_t] .+ Γ[3] * X[id[:marginal_pdf_y_t]]
    distr_y_Prime = nt[:marginal_pdf_y_t] .+ Γ[3] * XPrime[id[:marginal_pdf_y′_t]]

    # Joint distributions (uncompressing), needed to use the copula
    CDF_m         = cumsum([0.0; vec(distr_m)])
    CDF_k         = cumsum([0.0; vec(distr_k)])
    CDF_y         = cumsum([0.0; vec(distr_y)])

    # Construct the copula from the perturbed distribution => perturbed copula
    cum_zero = zeros(eltype(θD), nm + 1, nk + 1, ny + 1)
    cum_zero[2:end,2:end,2:end] = DISTR # add the CDF
    Copula1(x::AbstractVector,y::AbstractVector,z::AbstractVector) =
        mylinearinterpolate3(cum_zero[:,end,end], cum_zero[end,:,end], cum_zero[end,end,:], cum_zero, x, y, z) # TODO: check if I can use view
    #Copula1 = LinearInterpolation((cum_zero[:,end,end],cum_zero[end,:,end],cum_zero[end,end,:]),cum_zero,extrapolation_bc=Line())

    # Compute the distribution implied by the perturbed marginals and copula
    CDF_joint     = Copula1(vec(CDF_m), vec(CDF_k), vec(CDF_y)) # roughly 5% of time
    distr         = diff(diff(diff(CDF_joint; dims=3);dims=2);dims=1)

    ############################################################################
    # I.3 uncompressing policies/value functions
    ###########################################################################
    if any((tot_dual.(XPrime[id[:Vm′_t]]) + realpart.(XPrime[id[:Vm′_t]])) .!= 0.0)
        θm      = uncompress(dct_compression_indices[:Vm], XPrime[id[:Vm′_t]], DC, IDC, (nm, nk, ny))
        VmPrime = vec(nt[:Vm_t]) + θm
    else
         VmPrime = vec(nt[:Vm_t]) .+ zero(eltype(X))
    end
    VmPrime .= (exp.(VmPrime))

     if any((tot_dual.(XPrime[id[:Vk′_t]]) + realpart.(XPrime[id[:Vk′_t]])) .!= 0.0)
        θk      = uncompress(dct_compression_indices[:Vk], XPrime[id[:Vk′_t]], DC, IDC, (nm, nk, ny))
        VkPrime = vec(nt[:Vk_t]) + θk
     else
         VkPrime = vec(nt[:Vk_t]) .+ zero(eltype(X))
     end
    VkPrime .= (exp.(VkPrime))

    ############################################################################
    #           II. Auxiliary Variables                                        #
    ############################################################################
    # Transition Matrix Productivity
    if tot_dual.(σ_t .+ zero(eltype(X))) == 0.0
        if σ_t == 1.0
            Π                  = Π .+ zero(eltype(X))
        else
            Π                  = Π
            PP                 = ExTransition(θ[:ρ_h], y_bin_bounds, sqrt(σ_t))
            Π[1:end-1,1:end-1] = PP.*(1.0 - θ[:ζ])
        end
    else
        Π                  = Π .+ zero(eltype(X))
        PP                 = ExTransition(θ[:ρ_h], y_bin_bounds, sqrt(σ_t))
        Π[1:end-1,1:end-1] = PP.*(1.0 - θ[:ζ])
    end

    ############################################################################
    #           III. Error term calculations (i.e. model starts here)          #
    ############################################################################

    ############################################################################
    #           III. 1. Aggregate Part #
    ############################################################################
    F            = Fsys_agg(X, XPrime, θ, grids, id, nt, eqconds) # Fsys_agg(X, XPrime, θ, grids, id, nt, eqconds)

    # Error Term on prices/aggregate summary vars (logarithmic, controls)
    KP           = dot(k_grid, distr_k)
    F[first(eqconds[:eq_capital_market_clear])] = log.(K_t)     - log.(KP)
    BP           = dot(m_grid, distr_m)
    F[first(eqconds[:eq_bond_market_clear])] = log.(B_t)     - log.(BP)

    BDact = -dot(distr_m, (m_grid .< 0.) .* m_grid)

    F[first(eqconds[:eq_debt_market_clear])] = log.(BD_t)  - log.(BDact)

    # Average Human Capital =
    # average productivity (at the productivity grid, used to normalize to 0)
    H       = dot(view(distr_y, 1:ny-1), view(y_grid, 1:ny-1))

    ############################################################################
    #               III. 2. Heterogeneous Agent Part                           #
    ############################################################################
    # Incomes
    nonpos_ret   = θ[:Rbar] .* (m_ndgrid .<= 0.0)
    eff_int      = ((RB_t * A_t) .+ nonpos_ret) ./ π_t # effective rate (need to check timing below and inflation)
    eff_intPrime = ((RB′_t * A′_t) .+ nonpos_ret) ./ π′_t

    GHHFA                    = ((θ[:γ] + τ_prog_t) / (θ[:γ] + 1.)) # transformation (scaling) for composite good
    tax_prog_scale           = (θ[:γ] + θ[:τ_prog]) / ((θ[:γ] + τ_prog_t))
    y_ndgrid_rel_H           = y_ndgrid ./ H
    wages_times_labor        = mc_w_t * w_t * N_t
    gross_labor_income       = (y_ndgrid_rel_H .^ tax_prog_scale) .* wages_times_labor
    gross_labor_income_rel_Ht       = gross_labor_income ./ Ht_t
    labor_income_rel_Ht = τ_level_t .* (gross_labor_income_rel_Ht) .^ (1. - τ_prog_t)
    entrep_profits           = view(y_ndgrid, :, :, ny) .* profits_t
    entrep_profits_net_taxes = τ_level_t .* entrep_profits.^(1.0 - τ_prog_t) # profit income net of taxes
    inc = [  GHHFA .* labor_income_rel_Ht .+ (union_profits_t) .* (1.0 - avg_tax_rate_t) .* HW, # labor income (NEW)
             (rk_t - 1.0) .* k_ndgrid, # rental income
             eff_int .* m_ndgrid, # liquid asset Income
             k_ndgrid .* q_t,
             τ_level_t .* (wages_times_labor .* y_ndgrid_rel_H).^(1.0 - τ_prog_t) .* ((1.0 - τ_prog_t) / (θ[:γ] + 1)),
             labor_income_rel_Ht] # capital liquidation Income (q=1 in steady state)
    inc[1][:,:,end] .= entrep_profits_net_taxes
    inc[5][:,:,end] .= 0.0
    inc[6][:,:,end] .= entrep_profits_net_taxes

    incgross =[  gross_labor_income_rel_Ht .+ union_profits_t,
                 (rk_t .- 1.0) .* k_ndgrid,                                      # rental income # TODO: can we copy from inc?
                 eff_int .* m_ndgrid,                                        # liquid asset Income
                 k_ndgrid .* q_t,
                 gross_labor_income_rel_Ht]           # capital liquidation Income (q=1 in steady state)
    incgross[1][:,:,end] .= entrep_profits
    incgross[5][:,:,end] .= entrep_profits

    taxrev      = incgross[5]-inc[6] # tax revenues w/o tax on union profits
    incgrossaux = incgross[5]
    tot_taxrev  = dot(distr, taxrev)
    F[first(eqconds[:eq_tax_level])] = avg_tax_rate_t - tot_taxrev / dot(distr, incgrossaux)
    F[first(eqconds[:eq_tax_revenue])]    = log(T_t) - log(tot_taxrev + avg_tax_rate_t * (union_profits_t))

    inc[6] = labor_income_rel_Ht .+ ((1.0 .- mc_w_t) .* w_t .* N_t) .* (1.0 .- avg_tax_rate_t)
    inc[6][:,:,end] .= entrep_profits_net_taxes

    # Calculate optimal policies
    # expected marginal values
    EVkPrime = reshape(VkPrime, (nm, nk, ny))
    EVmPrime = reshape(VmPrime, (nm, nk, ny))

    @views @inbounds begin
        for mm = 1:nm
            EVkPrime[mm,:,:] .= EVkPrime[mm,:,:]*Π' # TODO: get rid of broadcasting, I believe it makes an unnecessary allocation relative to =
            EVmPrime[mm,:,:] .= eff_intPrime[mm,:,:].*(EVmPrime[mm,:,:]*Π')
        end
    end
    c_a_star, m_a_star, k_a_star, c_n_star, m_n_star =
                    EGM_policyupdate(EVmPrime, EVkPrime, q_t, π_t, RB_t * A_t, 1.0, inc, θ, grids, false) # policy iteration

    # Update marginal values
    Vk_new, Vm_new = updateV(EVkPrime, c_a_star, c_n_star, m_n_star, rk_t - 1.0, q_t, θ, m_grid, Π) # update expected marginal values time t

    # Calculate error terms on marginal values
    Vm_err        = log.((Vm_new)) - nt[:Vm_t]
    Vm_thet       = compress(dct_compression_indices[:Vm], Vm_err, DC, IDC, (nm, nk, ny))
    F[eqconds[:eq_marginal_value_bonds]] = X[id[:Vm_t]] - Vm_thet

    Vk_err        = log.((Vk_new)) - nt[:Vk_t]
    Vk_thet       = compress(dct_compression_indices[:Vk], Vk_err, DC, IDC, (nm, nk, ny))
    F[eqconds[:eq_marginal_value_capital]] = X[id[:Vk_t]] - Vk_thet

    # Error Term on distribution (in levels, states)
    dPrime        = DirectTransition(m_a_star,  m_n_star, k_a_star, distr, θ[:λ],
                                     Π, (nm, nk, ny), m_grid, k_grid)
    dPrs          = reshape(dPrime, nm, nk, ny)
    temp          = dropdims(sum(dPrs,dims=(2,3)),dims=(2,3))
    cum_m         = cumsum(temp)
    F[eqconds[:eq_marginal_pdf_m]] = view(temp, 1:nm-1) - view(distr_m_Prime, 1:nm-1)
    temp          = dropdims(sum(dPrs,dims=(1,3)),dims=(1,3))
    cum_k         = cumsum(temp)
    F[eqconds[:eq_marginal_pdf_k]] = view(temp, 1:nk-1) - view(distr_k_Prime, 1:nk-1)
    temp          = distr_y' * Π # dropdims(sum(dPrs,dims=(1,2)),dims=(1,2))
    cum_h         = cumsum(temp')
    F[eqconds[:eq_marginal_pdf_y]] = view(temp, 1:ny-1) - view(distr_y_Prime, 1:ny-1)

    # Construct new copula of the distribution after accounting for effects of the initial perturbations
    cum_zero = zeros(eltype(θD), nm + 1, nk + 1, ny + 1)
    cum_dist_new = cumsum(cumsum(cumsum(dPrs; dims=3);dims=2);dims=1)
    cum_zero[2:end,2:end,2:end] = cum_dist_new
    Copula2(x::AbstractVector, y::AbstractVector, z::AbstractVector) = mylinearinterpolate3([0; cum_m], [0; cum_k], [0; cum_h],
                                                                                                 cum_zero, x, y, z)
    # Compute implied CDF # TODO: add marginal cdfs to the named tuple so we don't need to do this calculation all the time
    CDF_joint     = Copula2([0.0; cumsum(nt[:marginal_pdf_m_t])] .+ zero(eltype(θD)),
                            [0.0; cumsum(nt[:marginal_pdf_k_t])] .+ zero(eltype(θD)),
                            [0.0; cumsum(nt[:marginal_pdf_y_t])] .+ zero(eltype(θD))) # roughly 5% of time

    # Get implied distribution and compute the error relative to steady state
    distr_up         = diff(diff(diff(CDF_joint; dims=3);dims=2);dims=1)
    distr_err        = distr_up - nt[:distr_t]

    # Compute the DCT using the steady-state basis and calculate the change in free DCT coefficients
    D_thet       = compressD(dct_compression_indices[:copula], distr_err[1:end-1, 1:end-1, 1:end-1], DCD, IDCD, (nm, nk, ny))
    F[eqconds[:eq_copula]] =  D_thet - XPrime[id[:copula_t]]

    # Compute distributional variables
    distr_m_act, distr_k_act, distr_y_act, share_borroweract, GiniWact, I90shareact, I90sharenetact, GiniXact, #=
        =# sdlogxact, P9010Cact, GiniCact, sdlgCact, P9010Iact, GiniIact, sdlogyact, w90shareact, P10Cact, P50Cact, P90Cact =
        distrSummaries(distr, c_a_star, c_n_star, inc, incgross, θ, (nm, nk, ny), grids)

    Htact                   = dot(view(distr_y, 1:ny-1), (view(y_grid, 1:ny-1) ./ H) .^ (tax_prog_scale))
    F[first(eqconds[:eq_Ht])]           = log(Ht_t)            - log(Htact)
    F[first(eqconds[:eq_Gini_X])]        = log(Gini_X_t)         - log(GiniXact)
    F[first(eqconds[:eq_I90_share])]     = log(I90_share_t)     - log(I90shareact)
    F[first(eqconds[:eq_I90_share_net])]  = log(I90_share_net_t) - log(I90sharenetact)

    F[first(eqconds[:eq_W90_share])]     = log(W90_share_t)     - log(w90shareact)
    F[first(eqconds[:eq_sd_log_y])]       = log(sd_log_y_t)      - log(sdlogyact)
    F[first(eqconds[:eq_Gini_C])]        = log(Gini_C_t)        - log(GiniCact)

    return F
end
=#
