function steadystate!(m::BayerBornLuetticke; verbose::Symbol = :none)
    # See helper_functions/HetAgentDSGE

    # TODO: add ability to pass initial guesses for VmSS, VkSS more robustly (currently, find_steadystate makes a guess itself)
    if !get_setting(m,:compute_steadystate)
       return m
    end

    if get_setting(m, :replicate_original_output)

        # Thi block replicates the output from the original implementation by Bayer, Born, and Luetticke
        KSS, VmSS, VkSS, distrSS = original_find_steadystate(m; verbose = verbose,
                                                             skip_coarse_grid = haskey(get_settings(m), :skip_coarse_grid) &&
                                                             get_setting(m, :skip_coarse_grid))

        KSS1 = KSS
        VmSS1 = VmSS
        VkSS1 = VkSS
        distrSS1 = distrSS
        save_steadystate(m,KSS,VmSS,VkSS,distrSS)
        println("steady state saved")
        @save "save1.jld2" KSS1 VmSS1 VkSS1 distrSS1

        original_prepare_linearization(m, KSS, VmSS, VkSS, distrSS; verbose = verbose)
        m <= Setting(:compute_steadystate, false)

    else
        if get_setting(m, :compute_full_steadystate)
            # Compute steady state
            KSS, VmSS, VkSS, distrSS = find_steadystate(m; verbose = verbose,
                                                        skip_coarse_grid = haskey(get_settings(m), :skip_coarse_grid) &&
                                                        get_setting(m, :skip_coarse_grid),
                                                        use_old_steadystate = haskey(get_settings(m), :use_old_steadystate) &&
                                                        get_setting(m, :use_old_steadystate),
                                                        parallel = haskey(get_settings(m), :parallel_steadystate) &&
                                                       get_setting(m, :parallel_steadystate))

            m <= Setting(:save_steadystate, true)
            println("saving steady state")

            if get_setting(m, :save_steadystate)
                save_steadystate(m, KSS, VmSS, VkSS, distrSS)
            end

            # Update steady-state parameters, reduce state space, and update indices
            prepare_linearization(m, KSS, VmSS, VkSS, distrSS; verbose = verbose,
                                  parallel = haskey(get_settings(m), :parallel_steadystate) &&
                                  get_setting(m, :parallel_steadystate))

            m <= Setting(:compute_full_steadystate, false)
        else
            aggregate_steadystate!(m)
        end
    end

    return m
end

"""
```
aggregate_steadystate!(m::BayerBornLuetticke)
```
computes the steady state of aggregate scalar variables.

This function breaks out the "analytic" steady-state solution here instead of in steadystate!
since some of these parameters are required to construct the grids that are used in the
functional/distributional steady state variable calculation
(where init_grids! is called before steadystate! in model construction).
"""
function aggregate_steadystate!(m::BayerBornLuetticke{T}) where {T <: Real}

    m_grid = get_gridpts(m, :m_grid)::Vector{T}

    # Unlog some steady state numbers
    K_star = exp(m[:K_star])
    N_star = exp(m[:N_star])
    Y_star = exp(m[:Y_star])
    G_star = exp(m[:G_star])
    T_star = exp(m[:T_star])
    B_star = exp(m[:B_star])
    I_star = exp(m[:I_star])
    w_star = exp(m[:w_star])

    # Create steady state values information. Note these are the LOG numbers
    m[:A_star] = 0.
    m[:Z_star] = 0.
    m[:Ψ_star] = 0.
    m[:RB_star] = log(m[:RB])
    m[:μ_p_star] = log(m[:μ_p])
    m[:μ_w_star] = log(m[:μ_w])
    m[:τ_prog_star] = log(m[:τ_prog])
    m[:τ_level_star] = log(m[:τ_lev])
    m[:σ_star] = 0.
    m[:τ_prog_obs_star] = 0.
    m[:G_sh_star] = 0.
    m[:R_sh_star] = 0.
    m[:P_sh_star] = 0.
    m[:S_sh_star] = 0.
    m[:rk_star] = log(1. + _bbl_interest(K_star, 1. / m[:μ_p], N_star, m[:α], m[:δ_0])) # need to recalculate b/c there are estimated parameters
    rk_star = exp(m[:rk_star])
    m[:LP_star] = log(1. + rk_star - m[:RB])
    m[:LP_XA_star] = log(1. + rk_star - m[:RB])
    m[:π_star] = log(m[:π])
    m[:π_w_star] = 0.
    m[:BD_star] = log(-dot(m[:marginal_pdf_m_star], (m_grid .< 0.) .* m_grid))
    m[:C_star] = log(Y_star - m[:δ_0] * K_star - G_star - m[:Rbar] * exp(m[:BD_star]))
    m[:q_star] = 0.
    m[:mc_star] = log(1. ./ m[:μ_p])
    m[:mc_w_star] = log(1. ./ m[:μ_w])
    m[:mc_w_w_star] = log(w_star * exp(m[:mc_w_star]))
    m[:u_star] = 0.
    m[:profits_star] = log((1. - exp(m[:mc_star])) .* Y_star)
    m[:union_profits_star] = log((1. - exp(m[:mc_w_star])) .* w_star .* N_star)
    m[:BY_star] = log(B_star / Y_star)
    m[:TY_star] = log(T_star / Y_star)
    m[:T_l1_star] = get_untransformed_values(m[:T_star])::T
    m[:Y_l1_star] = get_untransformed_values(m[:Y_star])::T
    m[:B_l1_star] = get_untransformed_values(m[:B_star])::T
    m[:G_l1_star] = get_untransformed_values(m[:G_star])::T
    m[:I_l1_star] = get_untransformed_values(m[:I_star])::T
    m[:w_l1_star] = get_untransformed_values(m[:w_star])::T
    m[:q_l1_star] = get_untransformed_values(m[:q_star])::T
    m[:C_l1_star] = get_untransformed_values(m[:C_star])::T
    m[:avg_tax_rate_l1_star] = get_untransformed_values(m[:avg_tax_rate_star])::T
    m[:τ_prog_l1_star] = log(m[:τ_prog])
    m[:Ygrowth_star] = 0.
    m[:Bgrowth_star] = 0.
    m[:Igrowth_star] = 0.
    m[:wgrowth_star] = 0.
    m[:Cgrowth_star] = 0.
    m[:Tgrowth_star] = 0.
    m[:Ht_star] = 0.
    #m[:retained_star] = 0.
    #m[:firm_profits_star] = get_untransformed_values(m[:profits_star])::T
    #m[:union_retained_star] = 0.
    #m[:union_firm_profits_star] = get_untransformed_values(m[:union_profits_star])::T
    #m[:tot_retained_Y_star] = 0.

    return m
end
