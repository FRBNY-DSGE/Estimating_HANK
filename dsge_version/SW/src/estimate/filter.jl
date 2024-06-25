"""
```
filter(m, data, system, s_0 = [], P_0 = []; cond_type = :none,
       include_presample = true, in_sample = true,
       outputs = [:loglh, :pred, :filt])
```

Computes and returns the filtered values of states for the state-space
system corresponding to the current parameter values of model `m`.

### Inputs

- `m::AbstractDSGEModel`: model object
- `data`: `DataFrame` or `nobs` x `hist_periods` `Matrix{S}` of data for
  observables. This should include the conditional period if `cond_type in
  [:semi, :full]`
- `system::System` or `RegimeSwitchingSystem: `System` object specifying state-space system matrices for
  the model
- `s_0::Vector{S}`: optional `Nz` x 1 initial state vector
- `P_0::Matrix{S}`: optional `Nz` x `Nz` initial state covariance matrix

where `S<:AbstractFloat`.

### Keyword Arguments

- `cond_type::Symbol`: conditional case. See `forecast_all` for documentation of
  all `cond_type` options
- `include_presample::Bool`: indicates whether to include presample periods in
  the returned vector of `Kalman` objects
- `in_sample::Bool`: indicates whether or not to discard out of sample rows in
    `df_to_matrix` call
- `outputs::Vector{Symbol}`: which Kalman filter outputs to compute and return.
  See `?kalman_filter`

### Outputs

- `kal::Kalman`: see `?Kalman`
"""
function filter(m::AbstractDSGEModel, df::DataFrame, system::Union{System{S}, RegimeSwitchingSystem{S}},
                s_0::Vector{S} = Vector{S}(undef, 0),
                P_0::Matrix{S} = Matrix{S}(undef, 0, 0);
                cond_type::Symbol = :none, include_presample::Bool = true,
                in_sample::Bool = true,
                outputs::Vector{Symbol} = [:loglh, :pred, :filt],
                tol::Float64 = 0.0) where {S<:AbstractFloat}

    data = df_to_matrix(m, df; cond_type = cond_type, in_sample = in_sample)
    start_date = max(date_presample_start(m), df[1, :date])
    filter(m, data, system, s_0, P_0; start_date = start_date,
           include_presample = include_presample, outputs = outputs, tol = tol)
end

function filter(m::AbstractDSGEModel, data::AbstractArray, system::System{S},
                s_0::Vector{S} = Vector{S}(undef, 0),
                P_0::Matrix{S} = Matrix{S}(undef, 0, 0);
                start_date::Date = date_presample_start(m),
                include_presample::Bool = true,
                outputs::Vector{Symbol} = [:loglh, :pred, :filt],
                tol::Float64 = 0.0) where {S<:AbstractFloat}

    T = size(data, 2)

    # Partition sample into pre- and post-ZLB regimes
    # Note that the post-ZLB regime may be empty if we do not impose the ZLB
    regime_inds = zlb_regime_indices(m, data, start_date)

    # Get system matrices for each regime
    TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs = zlb_regime_matrices(m, system, start_date)

    # If s_0 and P_0 provided, check that rows and columns corresponding to
    # anticipated shocks are zero in P_0
    if !isempty(s_0) && !isempty(P_0)
        ant_state_inds = setdiff(1:n_states_augmented(m), inds_states_no_ant(m))
        @assert all(x -> x == 0, P_0[:, ant_state_inds])
        @assert all(x -> x == 0, P_0[ant_state_inds, :])
    end

    # Specify number of presample periods if we don't want to include them in
    # the final results
    Nt0 = include_presample ? 0 : n_presample_periods(m)

    # Run Kalman filter, construct Kalman object, and return
    out = kalman_filter(regime_inds, data, TTTs, RRRs, CCCs, QQs,
                        ZZs, DDs, EEs, s_0, P_0; outputs = outputs,
                        Nt0 = Nt0, tol = tol)
    return Kalman(out...)
end

function filter(m::AbstractDSGEModel, data::AbstractArray, system::RegimeSwitchingSystem{S},
                s_0::Vector{S} = Vector{S}(undef, 0),
                P_0::Matrix{S} = Matrix{S}(undef, 0, 0);
                start_date::Date = date_presample_start(m),
                include_presample::Bool = true,
                outputs::Vector{Symbol} = [:loglh, :pred, :filt],
                tol::Float64 = 0.0) where {S<:AbstractFloat}

    T = size(data, 2)

    # Partition sample into regimes, including the pre- and post-ZLB regimes.
    regime_inds, i_zlb_start, splice_zlb_regime = zlb_plus_regime_indices(m, data, start_date)

    # Get system matrices for each regime
    TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs = zlb_plus_regime_matrices(m, system, length(regime_inds),
                                                                    start_date;
                                                                    ind_zlb_start = i_zlb_start,
                                                                    splice_zlb_regime = splice_zlb_regime)

    # If s_0 and P_0 provided, check that rows and columns corresponding to
    # anticipated shocks are zero in P_0
    if !isempty(s_0) && !isempty(P_0)
        ant_state_inds = setdiff(1:n_states_augmented(m), inds_states_no_ant(m))
        @assert all(x -> x == 0, P_0[:, ant_state_inds])
        @assert all(x -> x == 0, P_0[ant_state_inds, :])
    end

    # Specify number of presample periods if we don't want to include them in
    # the final results
    Nt0 = include_presample ? 0 : n_presample_periods(m)

    # Run Kalman filter, construct Kalman object, and return
    out = kalman_filter(regime_inds, data, TTTs, RRRs, CCCs, QQs,
                        ZZs, DDs, EEs, s_0, P_0; outputs = outputs,
                        Nt0 = Nt0, tol = tol)
    return Kalman(out...)
end

"""
```
filter_likelihood(m, data, system, s_0 = [], P_0 = []; cond_type = :none,
       include_presample = true, in_sample = true, tol = 0.)
```

Computes and returns the likelihood after filtering the states for
the state-space system corresponding to the current parameter values of model `m`.

### Inputs

- `m::AbstractDSGEModel`: model object
- `data`: `DataFrame` or `nobs` x `hist_periods` `Matrix{S}` of data for
  observables. This should include the conditional period if `cond_type in
  [:semi, :full]`
- `system::System` or `system::RegimeSwitchingSystem`:
  `System` or `RegimeSwitchingSystem` object specifying
  state-space system matrices for the model
- `s_0::Vector{S}`: optional `Nz` x 1 initial state vector
- `P_0::Matrix{S}`: optional `Nz` x `Nz` initial state covariance matrix

where `S<:AbstractFloat`.

### Keyword Arguments

- `cond_type::Symbol`: conditional case. See `forecast_all` for documentation of
  all `cond_type` options
- `include_presample::Bool`: indicates whether to include presample periods in
  the returned vector of `Kalman` objects
- `in_sample::Bool`: indicates whether or not to discard out of sample rows in
    `df_to_matrix` call
- `outputs::Vector{Symbol}`: which Kalman filter outputs to compute and return.
  See `?kalman_filter`

### Outputs

- `kal::Kalman`: see `?Kalman`
"""
function filter_likelihood(m::AbstractDSGEModel, df::DataFrame, system::Union{System{S}, RegimeSwitchingSystem{S}},
                           s_0::Vector{S} = Vector{S}(undef, 0),
                           P_0::Matrix{S} = Matrix{S}(undef, 0, 0);
                           cond_type::Symbol = :none, include_presample::Bool = true,
                           in_sample::Bool = true,
                           add_zlb_duration::Tuple{Bool, Int} = (false, 1),
                           tol::Float64 = 0.0) where {S<:AbstractFloat}

    data = df_to_matrix(m, df; cond_type = cond_type, in_sample = in_sample)
    start_date = max(date_presample_start(m), df[1, :date])

    filter_likelihood(m, data, system, s_0, P_0; start_date = start_date,
                      add_zlb_duration = add_zlb_duration,
                      include_presample = include_presample, tol = tol)
end

function filter_likelihood(m::AbstractDSGEModel, data::AbstractArray, system::System{S},
                           s_0::Vector{S} = Vector{S}(undef, 0),
                           P_0::Matrix{S} = Matrix{S}(undef, 0, 0);
                           start_date::Date = date_presample_start(m),
                           include_presample::Bool = true,
                           add_zlb_duration::Tuple{Bool, Int} = (false, 1),
                           tol::Float64 = 0.0) where {S<:AbstractFloat}

    # Partition sample into pre- and post-ZLB regimes
    # Note that the post-ZLB regime may be empty if we do not impose the ZLB
    regime_inds = zlb_regime_indices(m, data, start_date)

    # Get system matrices for each regime
    TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs = zlb_regime_matrices(m, system, start_date)

    # If s_0 and P_0 provided, check that rows and columns corresponding to
    # anticipated shocks are zero in P_0
    if !isempty(s_0) && !isempty(P_0)
        ant_state_inds = setdiff(1:n_states_augmented(m), inds_states_no_ant(m))
        @assert all(x -> x == 0, P_0[:, ant_state_inds])
        @assert all(x -> x == 0, P_0[ant_state_inds, :])
    end

    # Specify number of presample periods if we don't want to include them in
    # the final results
    Nt0 = include_presample ? 0 : n_presample_periods(m)

    # Run Kalman filter, construct Kalman object, and return
    kalman_likelihood(regime_inds, data, TTTs, RRRs, CCCs, QQs,
                      ZZs, DDs, EEs, s_0, P_0;
                      Nt0 = Nt0, tol = tol)
end

function filter_likelihood(m::AbstractDSGEModel, data::AbstractArray,
                           system::RegimeSwitchingSystem{S},
                           s_0::Vector{S} = Vector{S}(undef, 0),
                           P_0::Matrix{S} = Matrix{S}(undef, 0, 0);
                           start_date::Date = date_presample_start(m),
                           include_presample::Bool = true,
                           add_zlb_duration::Tuple{Bool, Int} = (false, 1),
                           tol::Float64 = 0.0) where {S<:AbstractFloat}

    # Partition sample into regimes (including pre- and post-ZLB regimes).
    # Note that the post-ZLB regime may be empty if we do not impose the ZLB
    regime_inds, i_zlb_start, splice_zlb_regime = zlb_plus_regime_indices(m, data, start_date)

    # Get system matrices for each regime
    TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs = zlb_plus_regime_matrices(m, system, length(regime_inds),
                                                                    start_date;
                                                                    ind_zlb_start = i_zlb_start,
                                                                    splice_zlb_regime = splice_zlb_regime)

    # If s_0 and P_0 provided, check that rows and columns corresponding to
    # anticipated shocks are zero in P_0
    if !isempty(s_0) && !isempty(P_0)
        ant_state_inds = setdiff(1:n_states_augmented(m), inds_states_no_ant(m))
        @assert all(x -> x == 0, P_0[:, ant_state_inds])
        @assert all(x -> x == 0, P_0[ant_state_inds, :])
    end

    # Specify number of presample periods if we don't want to include them in
    # the final results
    Nt0 = include_presample ? 0 : n_presample_periods(m)

    # Run Kalman filter, construct Kalman object, and return
    if !add_zlb_duration[1]
        kalman_likelihood(regime_inds, data, TTTs, RRRs, CCCs, QQs,
                          ZZs, DDs, EEs, s_0, P_0; add_zlb_duration = add_zlb_duration,
                          Nt0 = Nt0, tol = tol)
    else
        filter_lik, zlb_st = kalman_likelihood(regime_inds, data, TTTs, RRRs, CCCs, QQs,
                          ZZs, DDs, EEs, s_0, P_0; add_zlb_duration = add_zlb_duration,
                          Nt0 = Nt0, tol = tol)

        # Compute implied ZLB duration
        zlb_ind = findfirst(x -> add_zlb_duration[2] in x, regime_inds)#regime_indices(m, start_date))
        ##TODO: Handle case when add_zlb_duration[2] != regime_inds[zlb_ind][end]

        # @show "save model pre"
        # mod_pre = deepcopy(m.settings)

        ### Save settings that need to change to forecast from add_zlb_duration[2]
        horizons = get_setting(m, :forecast_horizons)
        orig_regime_eqcond_info = deepcopy(get_setting(m, :regime_eqcond_info))
        orig_temp_altpol_len = get_setting(m, :temporary_altpolicy_length)
        orig_reg_forecast_start = get_setting(m, :reg_forecast_start)
        orig_reg_post_conditional_end = get_setting(m, :reg_post_conditional_end)
        orig_n_fcast_regimes = get_setting(m, :n_fcast_regimes)
        orig_n_hist_regimes = get_setting(m, :n_hist_regimes)
        orig_min_temp_altpol_len = haskey(m.settings, :min_temporary_altpolicy_length) ? get_setting(m, :min_temporary_altpolicy_length) : nothing
        orig_max_temp_altpol_len = haskey(m.settings, :max_temporary_altpolicy_length) ? get_setting(m, :max_temporary_altpolicy_length) : nothing
        orig_hist_temp_altpol_len = haskey(m.settings, :historical_temporary_altpolicy_length) ? get_setting(m, :historical_temporary_altpolicy_length) : nothing
        orig_cred_vary_until = haskey(m.settings, :cred_vary_until) ? get_setting(m, :cred_vary_until) : nothing
        orig_perf_cred = haskey(m.settings, :perfect_credibility_identical_transitions) ? get_setting(m, :perfect_credibility_identical_transitions) : nothing
        orig_iden_eqcond = haskey(m.settings, :identical_eqcond_regimes) ? get_setting(m, :identical_eqcond_regimes) : nothing
        orig_date_forecast_start = haskey(m.settings, :date_forecast_start) ? get_setting(m, :date_forecast_start) : nothing
        orig_date_conditional_end = haskey(m.settings, :date_conditional_end) ? get_setting(m, :date_conditional_end) : nothing
        orig_n_cond_regimes = haskey(m.settings, :n_cond_regimes) ? get_setting(m, :n_cond_regimes) : nothing
        orig_preprocessed_transitions = haskey(m.settings, :preprocessed_transitions) ? deepcopy(get_setting(m, :preprocessed_transitions)) : nothing

        ### Reset settings for add_zlb_duration[2]
        for a in collect(keys(get_setting(m, :regime_eqcond_info)))#zlb_ind+1:length(get_setting(m, :regime_eqcond_info))
            #=if get_setting(m, :regime_eqcond_info)[a].alternative_policy in [zlb_rule(), zero_rate()]
                get_setting(m, :temporary_altpolicy_length) += 1
            end=#
            if a >= zlb_ind ## = included b/c Great Recession ZLB adds 1 to regime_inds
                # get_setting(m, :regime_eqcond_info)[a].weights = get_setting(m, :regime_eqcond_info)[zlb_ind].weights
                get_setting(m, :regime_eqcond_info)[a].alternative_policy = flexible_ait()
                # get_setting(m, :regime_eqcond_info)[a].temporary_altpolicy_length = zlb_ind - 1
            end
        end ## TODO: Change expectations? No, leave alternative_policies as is.
        first_eqcond_key = sort(collect(keys(get_setting(m, :regime_eqcond_info))))[1]
        m <= Setting(:temporary_altpolicy_length, max(0,zlb_ind - first_eqcond_key + 1)) ## subtract 1 implicitly for ZLB regime
        m <= Setting(:reg_forecast_start, zlb_ind)
        m <= Setting(:reg_post_conditional_end, zlb_ind)
        m <= Setting(:n_fcast_regimes, get_setting(m, :n_regimes) - zlb_ind + 1)
        m <= Setting(:n_hist_regimes, zlb_ind - 1)
        m <= Setting(:date_forecast_start, get_setting(m, :regime_dates)[get_setting(m, :reg_forecast_start)])
        m <= Setting(:date_conditional_end, get_setting(m, :regime_dates)[max(1,zlb_ind-1)])

        if !isnothing(orig_min_temp_altpol_len)
            m <= Setting(:min_temporary_altpolicy_length, 0)
        end
        if !isnothing(orig_max_temp_altpol_len)
            delete!(m.settings, :max_temporary_altpolicy_length)
            # m <= Setting(:max_temporary_altpolicy_length, 30)
        end
        m <= Setting(:historical_temporary_altpolicy_length, max(0,zlb_ind - first_eqcond_key)) ## subtract 1 for ZLB regime
        # m <= Setting(:cred_vary_until, get_setting(m, :n_regimes))
        if !isnothing(orig_perf_cred)
            delete!(m.settings, :perfect_credibility_identical_transitions)
        end
        if !isnothing(orig_iden_eqcond)
            delete!(m.settings, :identical_eqcond_regimes)
        end

        #h5write("model_pre.h5", "mods", collect(keys(m.settings)))
        #=JLD2.jldopen("model_pre3.jld2", "w") do file
            file["m"] = m.settings
            # file["lik"] = filter_lik
        end=#

        ## Actually get the implied ZLB duration
        _, fcast_obs2, _ = forecast(m, zlb_st, zeros(length(zlb_st), horizons), zeros(length(m.observables), horizons),
                 zeros(length(m.pseudo_observables), horizons), zeros(length(m.exogenous_shocks), horizons);
                 cond_type = :none)

        implied_zlb_duration = findfirst(x -> x > get_setting(m, :zlb_rule_value) / 4.0 + 1e-5, fcast_obs2[m.observables[:obs_nominalrate],:]) - 1

        ### Reset to original model settings
        m <= Setting(:regime_eqcond_info, orig_regime_eqcond_info)
        m <= Setting(:reg_forecast_start, orig_reg_forecast_start)
        m <= Setting(:reg_post_conditional_end, orig_reg_post_conditional_end)
        m <= Setting(:n_fcast_regimes, orig_n_fcast_regimes)
        m <= Setting(:n_hist_regimes, orig_n_hist_regimes)
        m <= Setting(:temporary_altpolicy_length, orig_temp_altpol_len)
        m <= Setting(:date_forecast_start, orig_date_forecast_start)
        m <= Setting(:date_conditional_end, orig_date_conditional_end)

        if !isnothing(orig_min_temp_altpol_len)
            m <= Setting(:min_temporary_altpolicy_length, orig_min_temp_altpol_len)
        end
        if !isnothing(orig_max_temp_altpol_len)
            m <= Setting(:max_temporary_altpolicy_length, orig_max_temp_altpol_len)
        end
        if isnothing(orig_hist_temp_altpol_len)
            delete!(m.settings, :historical_temporary_altpolicy_length)
        else
            m <= Setting(:historical_temporary_altpolicy_length, orig_hist_temp_altpol_len)
        end
        if isnothing(orig_cred_vary_until)
            delete!(m.settings, :cred_vary_until)
        else
            m <= Setting(:cred_vary_until, orig_cred_vary_until)
        end
        if !isnothing(orig_perf_cred)
            m <= Setting(:perfect_credibility_identical_transitions, orig_perf_cred)
        else
            delete!(m.settings, :perfect_credibility_identical_transitions)
        end
        if !isnothing(orig_iden_eqcond)
            m <= Setting(:identical_eqcond_regimes, orig_iden_eqcond)
        else
            delete!(m.settings, :identical_eqcond_regimes)
        end
        if !isnothing(orig_n_cond_regimes)
            m <= Setting(:n_cond_regimes, orig_n_cond_regimes)
        else
            delete!(m.settings, :n_cond_regimes)
        end
        if !isnothing(orig_preprocessed_transitions)
            m <= Setting(:preprocessed_transitions, orig_preprocessed_transitions)
        else
            delete!(m.settings, :preprocessed_transitions)
        end

        #=@show "save model post"
        for i in collect(union(keys(m.settings), keys(mod_pre)))
           if !haskey(m.settings, i) || !(i in keys(mod_pre)) || get_setting(m, i) != mod_pre[i].value
               @show i
               @show !haskey(m.settings, i), !(i in keys(mod_pre))
               if get_setting(m, i) != mod_pre[i].value
                   @show "last one", i
                   @show get_setting(m, i)
                   @show mod_pre[i]
               end
           end
       end
       @show filter_lik=#
        #h5write("model_post.h5", "mods", collect(keys(m.settings)))
        #=JLD2.jldopen("model_post3.jld2", "w") do file
            file["m"] = m.settings
        end=#

        # Compute loss for ZLB duration
        prior_prob = log(pdf(Normal(0.0, 0.25), abs(log(get_setting(m, :zlb_duration)) - log(implied_zlb_duration))))

        # zlb_dist = Normal(log(get_setting(m, :zlb_duration)) - log(), log(1.5))
        # prior_prob = log(pdf(zlb_dist, log(implied_zlb_duration))) ##:zlb_duration is the actual median ZLB duration. -1 b/c we are actually including liftoff qtr (since implied_zlb_duration can be 0)

        return filter_lik .+ prior_prob#, prior_prob
    end
end

function filter_shocks(m::AbstractDSGEModel, df::DataFrame, system::System{S},
                       s_0::Vector{S} = Vector{S}(undef, 0), P_0::Matrix{S} = Matrix{S}(undef, 0, 0); cond_type::Symbol = :none,
                       start_date::Date = date_presample_start(m),
                       include_presample::Bool = false) where S<:AbstractFloat

    data = df_to_matrix(m, df, cond_type = cond_type)

    # Partition sample into pre- and post-ZLB regimes
    # Note that the post-ZLB regime may be empty if we do not impose the ZLB
    regime_inds = zlb_regime_indices(m, data, start_date)

    # Get system matrices for each regime
    Ts, Rs, Cs, Qs, Zs, Ds, Es = zlb_regime_matrices(m, system, start_date)

    # Initialize s_0 and P_0
    if isempty(s_0) || isempty(P_0)
        s_0, P_0 = init_stationary_states(Ts[1], Rs[1], Cs[1], Qs[1])

        # If s_0 and P_0 provided, check that rows and columns corresponding to
        # anticipated shocks are zero in P_0
        ant_state_inds = setdiff(1:n_states_augmented(m), inds_states_no_ant(m))
        # MDχ: Had to change the tolerance from exactly == 0, if below tol
        # explicitly set to 0
        @assert all(x -> abs(x) < 1e-14, P_0[:, ant_state_inds])
        @assert all(x -> abs(x) < 1e-14, P_0[ant_state_inds, :])
        P_0[:, ant_state_inds] = 0.
        P_0[ant_state_inds, :] = 0.
    end

    # Specify number of presample periods if we don't want to include them in
    # the final results
    Nt0 = include_presample ? 0 : n_presample_periods(m)

    # Dimensions
    Nt = size(data,  2) # number of periods of data
    Ns = size(Ts[1], 1) # number of states

    # Augment state space with shocks
    Ts, Rs, Cs, Zs, s_0, P_0 =
        augment_states_with_shocks(regime_inds, Ts, Rs, Cs, Qs, Zs, s_0, P_0)

    # Kalman filter stacked states and shocks stil_t
    _, _, _, stil_filt, Ptil_filt, _, _, _, _ =
        kalman_filter(regime_inds, data, Ts, Rs, Cs, Qs, Zs, Ds, Es, s_0, P_0,
                      outputs = [:filt])

    # Index out shocks
    ϵ_filt = stil_filt[Ns+1:end, :] # ϵ_{t|T}

    # Trim the presample if needed
    if Nt0 > 0
        insample = Nt0+1:Nt
        ϵ_filt = ϵ_filt[:, insample]
    end

    return ϵ_filt
end

function filter_shocks(m::AbstractDSGEModel, df::DataFrame, system::RegimeSwitchingSystem{S},
                       s_0::Vector{S} = Vector{S}(undef, 0), P_0::Matrix{S} = Matrix{S}(undef, 0, 0); cond_type::Symbol = :none,
                       start_date::Date = date_presample_start(m),
                       include_presample::Bool = false) where S<:AbstractFloat

    data = df_to_matrix(m, df, cond_type = cond_type)

    # Partition sample into regimes (including pre- and post-ZLB regimes).
    # Note that the post-ZLB regime may be empty if we do not impose the ZLB
    regime_inds, i_zlb_start, splice_zlb_regime = zlb_plus_regime_indices(m, data, start_date)

    # Get system matrices for each regime
    Ts, Rs, Cs, Qs, Zs, Ds, Es = zlb_plus_regime_matrices(m, system, length(regime_inds),
                                                          start_date;
                                                          ind_zlb_start = i_zlb_start,
                                                          splice_zlb_regime = splice_zlb_regime)

    # Initialize s_0 and P_0
    if isempty(s_0) || isempty(P_0)
        s_0, P_0 = init_stationary_states(Ts[1], Rs[1], Cs[1], Qs[1])

        # If s_0 and P_0 provided, check that rows and columns corresponding to
        # anticipated shocks are zero in P_0
        ant_state_inds = setdiff(1:n_states_augmented(m), inds_states_no_ant(m))
        # MDχ: Had to change the tolerance from exactly == 0, if below tol
        # explicitly set to 0
        @assert all(x -> abs(x) < 1e-14, P_0[:, ant_state_inds])
        @assert all(x -> abs(x) < 1e-14, P_0[ant_state_inds, :])
        P_0[:, ant_state_inds] .= 0.
        P_0[ant_state_inds, :] .= 0.
    end

    # Specify number of presample periods if we don't want to include them in
    # the final results
    Nt0 = include_presample ? 0 : n_presample_periods(m)

    # Dimensions
    Nt = size(data,  2) # number of periods of data
    Ns = size(Ts[1], 1) # number of states

    # Augment state space with shocks
    Ts, Rs, Cs, Zs, s_0, P_0 =
        augment_states_with_shocks(regime_inds, Ts, Rs, Cs, Qs, Zs, s_0, P_0)

    # Kalman filter stacked states and shocks stil_t
    _, _, _, stil_filt, Ptil_filt, _, _, _, _ =
        kalman_filter(regime_inds, data, Ts, Rs, Cs, Qs, Zs, Ds, Es, s_0, P_0,
                      outputs = [:filt])

    # Index out shocks
    ϵ_filt = stil_filt[Ns+1:end, :] # ϵ_{t|T}

    # Trim the presample if needed
    if Nt0 > 0
        insample = Nt0+1:Nt
        ϵ_filt = ϵ_filt[:, insample]
    end

    return ϵ_filt
end

"""
```
This section defines filter and filter_likelihood for the PoolModel type
```
"""
function filter(m::PoolModel, data::AbstractArray,
                s_0::AbstractArray{S} = Matrix{Float64}(undef,0,0);
                start_date::Date = date_presample_start(m),
                cond_type::Symbol = :none, include_presample::Bool = true,
                in_sample::Bool = true,
                tol::Float64 = 0., parallel::Bool = false,
                tuning::Dict{Symbol,Any} = Dict{Symbol,Any}()) where {S<:AbstractFloat}

    # Handle data
    Nt0 = include_presample ? 0 : n_presample_periods(m)
    if haskey(tuning, :n_presample_periods)
        tuning[:n_presample_periods] = Nt0
    end

    # Check tuning
    if isempty(tuning)
        try
            tuning = get_setting(m, :tuning)
        catch
            if get_setting(m, :weight_type) == :dynamic
                @warn "no tuning parameters provided; using default tempered particle filter values"
            end
        end
    end
    if haskey(tuning, :parallel) # contains parallel? change keyword to that if so
        parallel = tuning[:parallel]
    end

    # Compute transition and measurement equations
    ## TODO: Save time by just doing compute_system on 1 worker and then sending to
    ### each worker. But sendto and passobj are not working on functions.
    if parallel
        Φ, Ψ, F_ϵ, F_u, F_λ = compute_system(m)
        let m = m
            @sync @distributed for p in workers()
                Φ, Ψ, F_ϵ, F_u, F_λ = compute_system(m)
            end
        end
    else
        Φ, Ψ, F_ϵ, F_u, F_λ = compute_system(m)
    end

    # Check initial states
    n_particles = haskey(tuning, :n_particles) ? tuning[:n_particles] : 1000
    if isempty(s_0)
        s_0 = quantile.(Normal(), rand(F_λ, n_particles))
        # s_0 = minimum(F_λ) == 0.0 && maximum(F_λ) == 1.0 ? rand(F_λ, n_particles) : quantile.(Normal(), rand(F_λ, n_particles))
    elseif get_setting(m, :weight_type) == :dynamic
        if size(s_0,1) != n_particles && size(s_0,2) != n_particles
            error("s0 does not contain enough particles")
        end
    end

    # Check if PoolModel has fixed_sched. If not, assume no tempering
    fixed_sched = [1.]
    try
        fixed_sched = get_setting(m, :fixed_sched)
    catch KeyError
    end

    weight_type = get_setting(m, :weight_type)
    if weight_type == :dynamic
        return tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_0; parallel = parallel,
                                        poolmodel = true,
                                        fixed_sched = fixed_sched,
                                        tuning..., verbose = :none)
    elseif weight_type == :equal
        loglhconditional = log.(mapslices(x -> Ψ([0.], x), data, dims = 1))
        return sum(loglhconditional), loglhconditional
    elseif weight_type == :static
        loglhconditional = log.(mapslices(x -> Ψ([m[:λ].value; 1 - m[:λ].value], x), data, dims = 1))
        return sum(loglhconditional), loglhconditional
    elseif weight_type == :bma
        error("Estimation for Bayesian Model Averaging is computed directly by the estimate_bma function, so the filter function does not return anything.")
    end
end


function filter_likelihood(m::PoolModel, data::AbstractArray,
                           s_0::AbstractArray{S} = Matrix{Float64}(undef,0,0);
                           start_date::Date = date_presample_start(m),
                           cond_type::Symbol = :none, include_presample::Bool = true,
                           in_sample::Bool = true, parallel::Bool = false, tol::Float64 = 0.,
                           tuning::Dict{Symbol,Any} = Dict{Symbol,Any}()) where {S<:AbstractFloat}

    # Guarantee output settings in tuning give desired output
    tuning[:allout] = true
    tuning[:get_t_particle_dist] = false

    if get_setting(m, :weight_type) == :dynamic_weight
        ~, loglhconditional, ~ = filter(m, data, s_0; start_date = start_date,
                                        include_presample = include_presample,
                                        cond_type = cond_type, in_sample = in_sample, tol = tol,
                                        parallel = parallel, tuning = tuning)
    else
        ~, loglhconditional = filter(m, data, s_0; start_date = start_date,
                                     include_presample = include_presample,
                                     cond_type = cond_type, in_sample = in_sample, tol = tol,
                                     parallel = parallel, tuning = tuning)
    end

    return loglhconditional
end
