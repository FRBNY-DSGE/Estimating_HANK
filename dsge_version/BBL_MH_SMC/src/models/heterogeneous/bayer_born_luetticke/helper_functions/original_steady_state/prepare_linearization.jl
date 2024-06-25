using Roots
function original_prepare_linearization(m::BayerBornLuetticke, KSS::T, VmSS::AbstractArray{T, 3}, VkSS::AbstractArray{T, 3},
                                        distrSS::AbstractArray{T, 3}; verbose::Symbol = :none) where {T <: Real}

    # Set up
    if verbose in [:low, :high]
        println("Running reduction step to prepare linearization")
    end
    θ  = parameters2namedtuple(m)
    nm, nk, ny = get_idiosyncratic_dims(m)

    # Calculate other equilibrium quantities
    incgross, incnet, NSS, rkSS, wSS, YSS, ProfitsSS, ISS, RBSS, taxrev, tot_taxrev, avg_tax_rateSS, eff_int = _original_bbl_incomes(θ, m.grids, KSS, distrSS)


    # obtain other steady state variables
    KSS, BSS, TransitionMatSS, TransitionMat_aSS, TransitionMat_nSS,
        c_a_starSS, m_a_starSS, k_a_starSS, c_n_starSS, m_n_starSS, VmSS, VkSS, distrSS =
            original_Ksupply(RBSS, 1.0 + rkSS, m, VmSS, VkSS, distrSS, incnet, eff_int)


    KSS1 = KSS
    VmSS1 = VmSS
    VkSS1 = VkSS
    distrSS1 = distrSS

            # not passing verbose to Ksupply b/c any print statements in Ksupply are redundant


    VmSS                = log.(1.0 ./ (sqrt.(sqrt.(VmSS))))
    VkSS                = log.(1.0 ./ (sqrt.(sqrt.(VkSS))))


    # Calculate taxes and government expenditures
    # TSS                 = (tot_taxrev + avg_tax_rateSS * ((1.0 .- 1.0 ./ θ[:μ_w]) .* wSS .* NSS))
    TSS                 = (distrSS[:]' * taxrev[:] + avg_tax_rateSS * ((1. .- 1. ./ θ[:μ_w]) .* wSS .* NSS))
    GSS                 = TSS - (θ[:RB] ./ θ[:π] - 1.0) * BSS

    # Produce distributional summary statistics
    distr_m_SS, distr_k_SS, distr_y_SS, share_borrowerSS, GiniWSS, I90shareSS,I90sharenetSS, GiniXSS,
            sdlogxSS, P9010CSS, GiniCSS, sdlogCSS, P9010ISS, GiniISS, sdlogySS, w90shareSS, P10CSS, P50CSS, P90CSS =
            original_distrSummaries(distrSS, 1.0, c_a_starSS, c_n_starSS, incnet, incgross, θ, get_idiosyncratic_dims(m), m.grids)

    ## Store quantities in m

    # Aggregate scalars
    m[:K_star] = log(KSS)
    m[:N_star] = log(NSS)
    m[:Y_star] = log(YSS)
    m[:G_star] = log(GSS)
    m[:w_star] = log(wSS)
    m[:T_star] = log(TSS)
    m[:I_star] = log(ISS)
    m[:B_star] = log(BSS)
    m[:avg_tax_rate_star] = log(avg_tax_rateSS)

    # Scalar summary statistics
    m[:share_borrower_star]       = log(share_borrowerSS)
    m[:Gini_wealth_star]          = log(GiniWSS)
    m[:W90_share_star]            = log(w90shareSS)
    m[:I90_share_star]            = log(I90shareSS)
    m[:I90_share_net_star]        = log(I90sharenetSS)
    m[:Gini_C_star]               = log(GiniCSS)
    m[:P90_minus_P10_income_star] = log(P9010ISS)
    m[:sd_log_y_star]             = log(sdlogySS)
    m[:Gini_X_star]               = log(GiniXSS) # TODO: figure out exactly what this X is
    m[:sd_log_X_star]             = log(sdlogxSS)
    m[:P90_minus_P10_C_star]      = log(P9010CSS)
    m[:Gini_C_star]               = log(GiniCSS)
    m[:sd_log_C_star]             = log(sdlogCSS)
    m[:P10_C_star]                = log(P10CSS)
    m[:P50_C_star]                = log(P50CSS)
    m[:P90_C_star]                = log(P90CSS)

    # Functional/distributional variables
    m[:distr_star]          = distrSS    # note the distributional variables aren't logged
    m[:marginal_pdf_m_star] = distr_m_SS
    m[:marginal_pdf_k_star] = distr_k_SS
    m[:marginal_pdf_y_star] = distr_y_SS
    m[:Vm_star]             = VmSS       # note that VmSS and VkSS are both already logged
    m[:Vk_star]             = VkSS


    # ------------------------------------------------------------------------------
    ## STEP 2: Dimensionality reduction
    # ------------------------------------------------------------------------------
    # 2 a.) Discrete cosine transformation of marginal value functions
    # ------------------------------------------------------------------------------
    ThetaVm             = vec(dct(VmSS)) # Discrete cosine transformation of marginal liquid asset value


    ind                 = sortperm(abs.(vec(ThetaVm)); rev = true)   # Indexes of coefficients sorted by their absolute size
    coeffs              = 1                     # Container to store the number of retained coefficients



    # Find the important basis functions (discrete cosine) for VmSS (in L2 norm)
    while norm(view(ThetaVm, view(ind, 1:coeffs))) / norm(ThetaVm) < 1. - get_setting(m, :dct_energy_loss)
            coeffs     += 1                                          # add retained coefficients until only some share of energy is lost
    end
    compressionIndexesVm = ind[1:coeffs]                             # store indexes of retained coefficients


    ThetaVk             = vec(dct(VkSS))# Discrete cosine transformation of marginal liquid asset value


    ind                 = sortperm(abs.(vec(ThetaVk)); rev = true)   # Indexes of coefficients sorted by their absolute size
    coeffs              = 1



    # Find the important basis functions (discrete cosine) for VkSS
    while norm(view(ThetaVk, view(ind, 1:coeffs))) / norm(ThetaVk) < 1. - get_setting(m, :dct_energy_loss)
            coeffs     += 1                                          # add retained coefficients until only some share of energy is lost
    end
    compressionIndexesVk = ind[1:coeffs]                             # store indexes of retained coefficients
#=
    ind                 = sortperm(abs.(vec(ThetaD)); rev = true)    # Indexes of coefficients sorted by their absolute size
    n_copula_coefs      = get_setting(m, :n_copula_dct_coefficients) # keep n_copula_coefs coefficients, but
    compressionIndexesD = ind[2:1+n_copula_coefs]                    # leave out index no. 1 as this shifts the constant
=#

#Copula Coefficients from BBL, NEEDED TO USE n_dct_copula coeffs instead if wanted to restrict to 10 arbitrrily
SELECT = [ ((i+j+k) <= get_setting(m, :reduc_copula)) & (!((i == 1) & (j == 1)) & !((k == 1) & (j == 1)) & !((k == 1) & (i == 1))) for i = 1:get_setting(m, :nm_copula), j = 1:get_setting(m, :nk_copula), k = 1:get_setting(m, :ny_copula)]

compressionIndexesD = findall(SELECT[:])


    distr_LOL           = view(distrSS, 1:get_setting(m, :nm_copula)-1, 1:get_setting(m, :nk_copula)-1, 1:get_setting(m, :ny_copula)-1)      # Leave out last entry of histogramm (b/c it integrates to 1)
    ThetaD              = vec(dct(distr_LOL))                        # Discrete cosine transformation of Copula



    compressionIndexes  = Array{Array{Int, 1}, 1}(undef, 3)          # Container to store all retained coefficients in one array
    compressionIndexes[1] = compressionIndexesVm
    compressionIndexes[2] = compressionIndexesVk
    compressionIndexes[3] = compressionIndexesD

@show size(compressionIndexesVm)
@show size(compressionIndexesVk)
@show size(compressionIndexesD)

    # Store reduction parameters (coefficients go as SteadyStateParameterGrid, indices go as settings)
    m[:dct_Vm_star]     = ThetaVm
    m[:dct_Vk_star]     = ThetaVk
    m[:dct_copula_star] = ThetaD

    update_compression_indices!(m, [:Vm, :Vk, :copula],
                                compressionIndexesVm, compressionIndexesVk, compressionIndexesD)

    # TODO: move this step to the indices/dimensions update (setting is n_backward_looking_states)
    # add to no. of states the coefficients that perturb the copula
    # n_par.nstates = n_par.ny + n_par.nk + n_par.nm + n_par.naggrstates - 3 + length(compressionIndexes[3])


#nstates = get_setting(m, :nm) + get_setting(m, :nk) + get_setting(m, :ny) - 3 +

    # ------------------------------------------------------------------------------
    # 2b.) Produce the Copula as an interpolant on the distribution function
    #      and its marginals
    # ------------------------------------------------------------------------------
    # TODO: use our general CDF and marginal CDF quadrature here rather than this implementation
    # CDF_SS              = zeros(nm + 1, nk + 1, ny + 1) # Produce CDF of asset-income distribution (container here)
    CDF_SS                    = Array{Float64}(undef, nm + 1, nk + 1, ny + 1)        # Produce CDF of asset-income distribution (container here)
    CDF_SS[1, :, :]          .= 0.
    CDF_SS[:, 1, :]          .= 0.
    CDF_SS[:, :, 1]          .= 0.
    CDF_SS[2:end,2:end,2:end] = cumsum(cumsum(cumsum(distrSS,dims=1),dims=2),dims=3) # Calculate CDF from PDF
    CDF_m                     = cumsum([0.0; vec(distr_m_SS)])                       # Marginal distribution (cdf) of liquid assets
    CDF_k                     = cumsum([0.0; vec(distr_k_SS)])                       # Marginal distribution (cdf) of illiquid assets
    CDF_y                     = cumsum([0.0; vec(distr_y_SS)])                       # Marginal distribution (cdf) of income

    # TODO: create notion of "steady-state function parameter" and store this function there b/c we will need it for MCMC/SMC
    # Calculate interpolation nodes for the copula as those elements of the marginal distribution
    # that yield close to equal aggregate shares in liquid wealth, illiquid wealth and income.
    # Entrepreneur state treated separately.
    copula_marginal_m = copula_marg_equi(distr_m_SS, m.grids[:m_grid], get_setting(m,:nm_copula))
    copula_marginal_k = copula_marg_equi(distr_k_SS, m.grids[:k_grid], get_setting(m,:nk_copula))
    copula_marginal_y = copula_marg_equi_y(distr_y_SS, m.grids[:y_grid], get_setting(m,:ny_copula))
    m <= Setting(:copula_marginal_m, copula_marginal_m)
    m <= Setting(:copula_marginal_k, copula_marginal_k)
    m <= Setting(:copula_marginal_y, copula_marginal_y)




    Copula(x::Vector, y::Vector, z::Vector) =
        mylinearinterpolate3(CDF_m, CDF_k, CDF_y, CDF_SS, x, y, z) # Define Copula as a function (used only if not perturbed)

    # Store reduction parameters (cdfs go as SteadyStateParameterGrid, copula go as settings)
    m[:marginal_cdf_m_star] = CDF_m
    m[:marginal_cdf_k_star] = CDF_k
    m[:marginal_cdf_y_star] = CDF_y

    m <= Setting(:copula, Copula)
    # ------------------------------------------------------------------------------

    original_aggregate_steadystate!(m)
    setup_indices!(m)

    m
end


# in case we ever just want the vector
@inline function construct_steadystate_vector(m::BayerBornLuetticke; only_aggregate::Bool = false)

    if only_aggregate
        keys = [unprime(k) for k in vcat(get_aggregate_state_variables(m), get_aggregate_jump_variables(m))]

        return [get_untransformed_values(m[_bbl_parse_endogenous_states(k)]) for k in keys]
    else
        keys = vcat(:marginal_pdf_m_t, :marginal_pdf_k_t, :marginal_pdf_y_t, :distr_t, # TODO: are marginal_pdfs going to be state variables?
                    [_bbl_parse_endogenous_states(unprime(k)) for
                     k in vcat(get_aggregate_state_variables(m), m.jump_variables)])   # m.jump_variables = [Vm_t, Vk_t, aggregate scalars names...]

        return [get_untransformed_values(m[_bbl_parse_endogenous_states(k)]) for k in keys]
    end
end

@inline function construct_steadystate_namedtuple(m::BayerBornLuetticke; only_aggregate::Bool = false)

    # Create keys for steady-state variables
    keys = if only_aggregate
        ([unprime(k) for k in get_aggregate_state_variables(m)]...,
         [unprime(k) for k in get_aggregate_jump_variables(m)]...)
    else
        (:marginal_pdf_m_t, # not just using m.state_variables b/c we will need
         :marginal_pdf_k_t, # the entire distribution, not just the DCT coefficients
         :marginal_pdf_y_t,
         :distr_t,
         [unprime(k) for
          k in get_aggregate_state_variables(m)]...,
         [unprime(k) for k in m.jump_variables]...) # m.jump_variables = [Vm_t, Vk_t, aggregate scalars names...]
    end

    # Create NamedTuple by parsing key to obtain the implied steady-state value in m
    nt = NamedTuple{keys}(get_untransformed_values(m[_bbl_parse_endogenous_states(k)]) for k in keys)

    return nt
end

@inline function construct_prime_and_noprime_indices(m::BayerBornLuetticke; only_aggregate::Bool = false)

    if only_aggregate
        id = OrderedDict{Symbol, Int64}(k => i for (i, k) in enumerate(get_aggregate_state_variables(m)))
        n_aggr_states = length(id)
        for (i, k) in enumerate(get_aggregate_jump_variables(m))
            id[k] = i + n_aggr_states
        end

        for (k, v) in id
            id[unprime(k)] = v
        end
    else
        id = deepcopy(m.endogenous_states)

        for (k, v) in id
            id[unprime(k)] = v
        end
    end

    return id
end


function copula_marg_equi_y(distr_i, grid_i, nx)
    grid_i       = grid_i.points

    CDF_i        = cumsum(distr_i[:])          # Marginal distribution (cdf) of liquid assets
    aux_marginal = collect(range(CDF_i[1], stop = CDF_i[end], length = nx))

    x2 = 1.0
    for i = 2:nx-1
        equi(x1)            = equishares(x1, x2, grid_i[1:end-1], distr_i[1:end-1], nx-1)
        x2                  = find_zero(equi, (1e-9, x2))
        aux_marginal[end-i] = x2
    end

    aux_marginal[end]   = CDF_i[end]
    aux_marginal[1]     = CDF_i[1]
    aux_marginal[end-1] = CDF_i[end-1]
    copula_marginal     = copy(aux_marginal)
    jlast               = nx
    for i = nx-1:-1:1
        j = locate(aux_marginal[i], CDF_i) + 1
        if jlast == j
            j -=1
        end
        jlast = j
        copula_marginal[i] = CDF_i[j]
    end
    return copula_marginal
end

function copula_marg_equi(distr_i, grid_i, nx)
    grid_i       = grid_i.points
    CDF_i        = cumsum(distr_i[:])          # Marginal distribution (cdf) of liquid assets
    aux_marginal = collect(range(CDF_i[1], stop = CDF_i[end], length = nx))

    x2 = 1.0
    for i = 1:nx-1
        equi(x1)            = equishares(x1, x2, grid_i, distr_i, nx)
        x2                  = find_zero(equi ,(1e-9, x2))
        aux_marginal[end-i] = x2
    end

    aux_marginal[end] = CDF_i[end]
    aux_marginal[1]   = CDF_i[1]
    copula_marginal   = copy(aux_marginal)
    jlast             = nx
    for i = nx-1:-1:1
        j = locate(aux_marginal[i], CDF_i) + 1
        if jlast == j
            j -=1
        end
        jlast = j
        copula_marginal[i] = CDF_i[j]
    end
    return copula_marginal
end

function equishares(x1, x2, grid_i, distr_i, nx)
    FN_Wshares = cumsum(grid_i .* distr_i) ./ sum(grid_i .* distr_i)
    Wshares    = diff(mylinearinterpolate(cumsum(distr_i), FN_Wshares, [x1; x2]))
    dev_equi   = Wshares .- 1.0 ./ nx

    return dev_equi
end
