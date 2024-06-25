"""
```
decompose_forecast(m_new, m_old, df_new, df_old, input_type, cond_new, cond_old,
    classes; verbose = :low, kwargs...)

decompose_forecast(m_new, m_old, df_new, df_old, params_new, params_old,
    cond_new, cond_old, classes; check = false)
```
explains the differences between an old forecast and a new forecast
by decomposing the differences into three sources:

(1) Data revisions,
(2) News (e.g. new data that has become available since the old forecast),
(3) Re-estimation (i.e. changes in model parameters).

This function **does not** compute which shocks explain a forecast.
For example, if you want to know whether TFP or financial shocks
drive a given forecast, then you want to compute the shock decomposition
output variable (see `?shock_decompositions`, `forecast_one`, and `compute_meansbands`).

### Inputs

- `m_new::M` and `m_old::M` where `M<:AbstractDSGEModel`
- `df_new::DataFrame` and `df_old::DataFrame`
- `cond_new::Symbol` and `cond_old::Symbol`
- `classes::Vector{Symbol}`: some subset of `[:states, :obs, :pseudo]`

**Method 1 only:**

- `input_type::Symbol`: estimation type to use. Parameters will be loaded using
  `load_draws(m_new, input_type)` and `load_draws(m_old, input_type)` in this
  method

**Method 2 only:**

- `params_new::Vector{Float64}` and `params_old::Vector{Float64}`: single
  parameter draws to use

### Keyword Arguments

- `check::Bool`: whether to check that the individual components add up to the
  correct total difference in forecasts. This roughly doubles the runtime
- `shockdec_data_only::Bool`: whether the outputted shock decompositions should
    compare new model, new data, new params to old model, old data, new params
    or instead new model, new data to new model, old data.

**Method 1 only:**

- `verbose::Symbol`

### Outputs

The first method returns nothing. The second method returns
`decomp::Dict{Symbol, Matrix{Float64}}`, which has keys of the form
`:decomp<component><class>` and values of size `Ny` x `Nh`, where

- `Ny` is the number of variables in the given class
- `Nh` is the number of common forecast periods, i.e. periods between
  `date_forecast_start(m_new)` and `date_forecast_end(m_old)`
"""
function decompose_forecast(m_new::M, m_old::M, df_new::DataFrame, df_old::DataFrame,
                            input_type::Symbol, cond_new::Symbol, cond_old::Symbol,
                            classes::Vector{Symbol}; verbose::Symbol = :low,
                            forecast_string_new::String = "",
                            forecast_string_old::String = "",
                            params_new::AbstractArray = Vector{Float64}(undef, 0),
                            params_old::AbstractArray = Vector{Float64}(undef, 0),
                            apply_altpolicy::Bool = false, catch_smoother_lapack::Bool = false,
                            model_decomp::Bool = false,
                            kwargs...) where M<:AbstractDSGEModel

    # Get output file names
    decomp_output_files = get_decomp_output_files(m_new, m_old, input_type, cond_new, cond_old, classes, forecast_string_old = forecast_string_old, forecast_string_new = forecast_string_new, model_decomp = model_decomp)

    info_print(verbose, :low, "Decomposing forecast...")
    println(verbose, :low, "Start time: $(now())")

    # Set up call to lower-level method
    f(params_new::Vector{Float64}, params_old::Vector{Float64}) =
      decompose_forecast(m_new, m_old, df_new, df_old, params_new, params_old,
                         cond_new, cond_old, classes; apply_altpolicy = apply_altpolicy,
                         catch_smoother_lapack = catch_smoother_lapack, model_decomp = model_decomp,
                         kwargs...)

    # Single-draw forecasts
    if input_type in [:mode, :mean, :init]

        if isempty(params_new)
            params_new = load_draws(m_new, input_type, verbose = verbose, use_highest_posterior_value = input_type == :mode)
        end
        if isempty(params_old)
            params_old = load_draws(m_old, input_type, verbose = verbose, use_highest_posterior_value = input_type == :mode)
        end
        decomps = f(params_new, params_old)
        write_forecast_decomposition(m_new, m_old, input_type, classes, decomp_output_files, decomps,
                                     forecast_string_new = forecast_string_new, forecast_string_old = forecast_string_old,
                                     verbose = verbose, model_decomp = model_decomp)

    # Multiple-draw forecasts
    elseif input_type == :full

        block_inds, block_inds_thin = forecast_block_inds(m_new, input_type)
        nblocks = length(block_inds)
        total_forecast_time = 0.0

        for block = 1:nblocks
            println(verbose, :low)
            info_print(verbose, :low, "Decomposing block $block of $nblocks...")
            begin_time = time_ns()

            # Get to work!
            params_new_block = isempty(params_new) ? load_draws(m_new, input_type, block_inds[block], verbose = verbose) :
                [params_new[block_inds[block][i],:] for i in 1:length(block_inds[block])]
            params_old_block = isempty(params_old) ? load_draws(m_old, input_type, block_inds[block], verbose = verbose) :
                [params_old[block_inds[block][i],:] for i in 1:length(block_inds[block])]
            mapfcn = use_parallel_workers(m_new) ? pmap : map
            decomps = mapfcn(f, params_new_block, params_old_block)

            # Assemble outputs from this block and write to file
            decomps = convert(Vector{Dict{Symbol, Array{Float64}}}, decomps)
            decomps = assemble_block_outputs(decomps)
            write_forecast_decomposition(m_new, m_old, input_type, classes, decomp_output_files, decomps,
                                         block_number = Nullable(block), block_inds = block_inds_thin[block],
                                         forecast_string_new = forecast_string_new, forecast_string_old = forecast_string_old,
                                         verbose = verbose, model_decomp = model_decomp)
            GC.gc()

            # Calculate time to complete this block, average block time, and
            # expected time to completion
            block_time = (time_ns() - begin_time)/1e9
            total_forecast_time += block_time
            total_forecast_time_min     = total_forecast_time/60
            expected_time_remaining     = (total_forecast_time/block)*(nblocks - block)
            expected_time_remaining_min = expected_time_remaining/60

            println(verbose, :low, "\nCompleted $block of $nblocks blocks.")
            println(verbose, :low, "Total time elapsed: $total_forecast_time_min minutes")
            println(verbose, :low, "Expected time remaining: $expected_time_remaining_min minutes")
        end # of loop through blocks

    else
        error("Invalid input_type: $input_type. Must be in [:mode, :mean, :init, :full]")
    end

    combine_raw_forecast_output_and_metadata(m_new, decomp_output_files, verbose = verbose)

    println(verbose, :low, "\nForecast decomposition complete: $(now())")
end

function decompose_forecast(m_new::M, m_old::M, df_new::DataFrame, df_old::DataFrame,
                            params_new::Vector{Float64}, params_old::Vector{Float64},
                            cond_new::Symbol, cond_old::Symbol, classes::Vector{Symbol};
                            check::Bool = false, apply_altpolicy::Bool = false,
                            catch_smoother_lapack::Bool = false,
                            forecast_string_old::String = "",
                            forecast_string_new::String = "",
                            endogenous_zlb_new::Bool = false, endogenous_zlb_old::Bool = false,
                            enforce_zlb_new::Bool = false, enforce_zlb_old::Bool = false,
                            set_zlb_regime_vals_new::Function = identity, set_zlb_regime_vals_old::Function = identity,
                            shockdec_data_only::Bool = true,
                            model_decomp::Bool = false) where M<:AbstractDSGEModel

    # Check numbers of periods
    T, k, H = decomposition_periods(m_new, m_old, df_new, df_old, cond_new, cond_old)

    # Forecast
    f(m::AbstractDSGEModel, df::DataFrame, params::Vector{Float64}, cond_type::Symbol; kwargs...) =
        decomposition_forecast(m, df, params, cond_type, T, k, H; apply_altpolicy = apply_altpolicy,
                               catch_smoother_lapack = catch_smoother_lapack,
                               kwargs..., check = check)

    # Change old parameters to forecast old model with new parameters
    m_old.parameters = copy(m_new.parameters)
    if haskey(m_new.settings, :model2para_regime)
        m_old <= Setting(:model2para_regime, get_setting(m_new, :model2para_regime))
    end

    # New forecast
    out1 = f(m_new, df_new, params_new, cond_new, outputs = [:forecast, :shockdec],
             enforce_zlb = enforce_zlb_new, endogenous_zlb = endogenous_zlb_new,
             set_zlb_regime_vals = set_zlb_regime_vals_new) # new data, new params
    # New Model with Old Model AIT, New Data, New Data
    m_new <= Setting(:flexible_ait_φ_π, get_setting(m_old,:flexible_ait_φ_π))
    m_new <= Setting(:flexible_ait_φ_y, get_setting(m_old,:flexible_ait_φ_y))
    m_new <= Setting(:ait_Thalf, get_setting(m_old,:ait_Thalf))
    m_new <= Setting(:gdp_Thalf, get_setting(m_old,:gdp_Thalf))
    m_new <= Setting(:pgap_value, get_setting(m_old,:pgap_value))
    m_new <= Setting(:ygap_value, get_setting(m_old,:ygap_value))
    m_new <= Setting(:flexible_ait_ρ_smooth, get_setting(m_old,:flexible_ait_ρ_smooth))
    out2 = f(m_new, df_new, params_new, cond_new, outputs = [:forecast, :shockdec],
             enforce_zlb = enforce_zlb_new, endogenous_zlb = endogenous_zlb_new,
             set_zlb_regime_vals = set_zlb_regime_vals_new)
    # Eqcond Changes
    m_new <= Setting(:regime_eqcond_info, deepcopy(get_setting(m_old, :regime_eqcond_info)))
    m_new <= Setting(:alternative_policies, deepcopy(get_setting(m_old, :alternative_policies)))
    m_new <= Setting(:temporary_altpolicy_length, get_setting(m_old, :temporary_altpolicy_length))
    m_new <= Setting(:tvis_information_set, deepcopy(get_setting(m_old, :tvis_information_set)))
    setup_regime_switching_inds!(m_new, cond_type = cond_new)
    out3 = f(m_new, df_new, params_new, cond_new, outputs = [:forecast, :shockdec],
             enforce_zlb = enforce_zlb_new, endogenous_zlb = endogenous_zlb_new,
             set_zlb_regime_vals = set_zlb_regime_vals_new)
    # DATA
    # Remove just latest quarter of data
    df_new_lesscond = df_new[df_new[!,:date] .<= get_setting(m_old, :date_conditional_end), :]
    m_new <= Setting(:date_conditional_end, get_setting(m_old, :date_conditional_end))
    out4 = f(m_new, df_new_lesscond, params_new, cond_new, outputs = [:forecast, :shockdec],
             enforce_zlb = enforce_zlb_new, endogenous_zlb = endogenous_zlb_new,
             set_zlb_regime_vals = set_zlb_regime_vals_new)
    # Single out forecast quarter data revisions
    df_new_lesscond[.&(df_new_lesscond[!, :date] .<= get_setting(m_old, :date_conditional_end),
                       df_new_lesscond[!, :date] .>= get_setting(m_old, :date_forecast_start)), :] = df_old[.&(df_old[!, :date] .<= get_setting(m_old, :date_conditional_end),
                df_old[!, :date] .>= get_setting(m_old, :date_forecast_start)), :]
    out5 = f(m_new, df_new_lesscond, params_new, cond_new, outputs = [:forecast, :shockdec],
             enforce_zlb = enforce_zlb_new, endogenous_zlb = endogenous_zlb_new,
             set_zlb_regime_vals = set_zlb_regime_vals_new)
    # All other data revisions
    # Change m_new to allow forecasting with old data
    m_new_olddf = deepcopy(m_new)
    m_new_olddf <= Setting(:data_vintage, get_setting(m_old, :data_vintage))
    m_new_olddf <= Setting(:cond_vintage, get_setting(m_old, :cond_vintage))
    m_new_olddf <= Setting(:date_forecast_start, get_setting(m_old, :date_forecast_start))
    m_new_olddf <= Setting(:date_conditional_end, get_setting(m_old, :date_conditional_end))

    haskey(m_new_olddf.settings, :reg_forecast_start) && m_new_olddf <= Setting(:reg_forecast_start,
                                                                                collect(keys(get_setting(m_new_olddf, :regime_dates)))[findfirst(values(get_setting(m_new_olddf, :regime_dates)) .== get_setting(m_old, :date_forecast_start))])
    haskey(m_new_olddf.settings, :n_cond_regimes) && m_new_olddf <= Setting(:n_cond_regimes, get_setting(m_old, :n_cond_regimes))
    haskey(m_new_olddf.settings, :reg_post_conditional_end) && m_new_olddf <= Setting(:reg_post_conditional_end,
                                                                                      findlast(sort!(collect(values(get_setting(m_new_olddf, :regime_dates)))) .<= date_conditional_end(m_new_olddf))+1)
    haskey(m_new_olddf.settings, :n_hist_regimes) && m_new_olddf <= Setting(:n_hist_regimes, get_setting(m_new_olddf, :reg_forecast_start) - 1 + get_setting(m_new_olddf, :n_cond_regimes))
    haskey(m_new_olddf.settings, :n_fcast_regimes) && m_new_olddf <= Setting(:n_fcast_regimes, get_setting(m_new_olddf, :n_regimes) - get_setting(m_new_olddf, :reg_forecast_start) + 1)



    m_old_params = copy(m_old.parameters)
    m_old_mod2par = haskey(m_old.settings, :model2para_regime) ? get_setting(m_old, :model2para_regime) : nothing
    out6 = f(m_new_olddf, df_old, params_new, cond_new, outputs = [:forecast, :shockdec],
             enforce_zlb = enforce_zlb_new, endogenous_zlb = endogenous_zlb_new,
             set_zlb_regime_vals = set_zlb_regime_vals_new)

    # Other Model Settings
    m_old <= Setting(:model2para_regime, get_setting(m_new, :model2para_regime))

    out7 = f(m_old, df_old, params_new, cond_new, outputs = [:forecast, :shockdec],
             enforce_zlb = enforce_zlb_new, endogenous_zlb = endogenous_zlb_new,
             set_zlb_regime_vals = set_zlb_regime_vals_new)

    # Return to old parameters
    m_old.parameters = m_old_params
    if isnothing(m_old_mod2par)
        delete!(m_old.settings, :model2para_regime)
    else
        m_old <= Setting(:model2para_regime, m_old_mod2par)
    end

    # Old Forecast
    out8 = f(m_old, df_old, params_old, cond_old, outputs = [:forecast, :shockdec],
             enforce_zlb = enforce_zlb_old, endogenous_zlb = endogenous_zlb_old,
             set_zlb_regime_vals = set_zlb_regime_vals_old)

    # Initialize output dictionary
    decomp = Dict{Symbol, Array{Float64}}()
    # Decomposition
    for class in classes
        # All elements of out are of size Ny x Nh, where the second dimension
        # ranges from t = T+1:T+H
        forecastvar = Symbol(:histforecast, class) # Z s_{T+h} + D
        trendvar    = Symbol(:trend,        class) # Z D
        dettrendvar = Symbol(:dettrend,     class) # Z T^{T+h} s_0 + D
        shockdecvar = Symbol(:shockdec,     class) # Z \sum_{t=1}^{T+h} T^{T+h-t} R ϵ_t + D
        datavar     = Symbol(:data,         class) # Z \sum_{t=1}^{T-k} T^{T+h-t} R ϵ_t + D
        newsvar     = Symbol(:news,         class) # Z \sum_{t=T-k+1}^{T+h} T^{T+h-t} R ϵ_t + D

        # Minimum forecastvar indices
        min_ind = min(size(out2[forecastvar],1), size(out3[forecastvar],1))

        ## For shockdec, insert zeroes if shock in new model not in the old one
        if length(m_new.exogenous_shocks) > length(m_old.exogenous_shocks)
            old_shocks = zeros(size(out1[shockdecvar]))
            new_exog_keys = string.(keys(m_new.exogenous_shocks))
            old_exog_keys = string.(keys(m_old.exogenous_shocks))
            for i in 1:size(old_shocks, 3)
                indi = findfirst(x -> x == new_exog_keys[i], old_exog_keys)
                if !isnothing(indi)
                    old_shocks[:,:,i] = out4[shockdecvar][:,:,indi]
                end
            end
        else
            old_shocks = out4[shockdecvar]
        end


        # 1. AIT Changes
        policy_comp = out1[forecastvar] - out2[forecastvar]
        decomp[Symbol(:decomppolicyait, class)] = policy_comp

        # 2. Eqcond Changes
        eqcond_comp = out2[forecastvar] - out3[forecastvar]
        decomp[Symbol(:decomppolicyeqcond, class)] = eqcond_comp

        # 3. Latest quarter
        release_comp = out3[forecastvar] - out4[forecastvar]
        decomp[Symbol(:decomprelease, class)] = release_comp

        # 4. Conditional data revision
        cond_comp = out4[forecastvar] - out5[forecastvar]
        decomp[Symbol(:decompcond, class)] = cond_comp

        # 5. Historical data revision
        revise_comp = out5[forecastvar] - out6[forecastvar]
        decomp[Symbol(:decomprevise, class)] = revise_comp

        # 6. Other settings changes
        model_comp = out6[forecastvar] - out7[forecastvar]
        decomp[Symbol(:decompmodel, class)] = model_comp

        # 7. Parameter changes
        param_comp = out7[forecastvar] - out8[forecastvar]
        decomp[Symbol(:decompparam, class)] = param_comp

        shockdec_comp = out1[shockdecvar] - out8[shockdecvar] # Ny x Nh x Ne
        if shockdec_data_only
            decomp[Symbol(:decompshockdec, class)] = shockdec_comp
        else
            decomp[Symbol(:decompshockdec, class)] = out1[shockdecvar] - old_shocks ## Want full difference
        end

        dettrend_comp = out1[dettrendvar] - out8[dettrendvar]
        if shockdec_data_only
            decomp[Symbol(:decompdettrend, class)] = dettrend_comp
        else
            decomp[Symbol(:decompdettrend, class)] = out1[dettrendvar] - out4[dettrendvar]
        end
        # Get difference in trends
        if haskey(m_new.settings, :regime_dates) && haskey(m_new.settings, :n_regimes)
            # TODO adjust to handle forecasting the same regime (or more than 1 regime apart)
            trend_new = out1[trendvar][:, 1:end-1]
            trend_old = out8[trendvar]
            trend_comp = trend_new - trend_old
        else
            trend_new = get_trend_dates(Dict(1 => date_mainsample_start(m_new)), out1[trendvar],
                                        date_mainsample_start(m_new), size(out1[datavar],2),
                                        n_regs = 1)
            trend_old = get_trend_dates(Dict(1 => date_mainsample_start(m_new_olddf)), out4[trendvar],
                                        date_mainsample_start(m_new_olddf), size(out4[datavar],2),
                                        n_regs = 1)
            trend_comp = trend_new - trend_old
        end

        if shockdec_data_only
            decomp[Symbol(:decomptrend, class)] = trend_comp
        elseif haskey(m_old.settings, :regime_dates) && haskey(m_old.settings, :n_regimes)
            trend4 = get_trend_dates(get_setting(m_old, :regime_dates), out4[trendvar],
                                     date_mainsample_start(m_old), size(out4[datavar],2),
                                     n_regs = get_setting(m_old, :n_regimes))
            decomp[Symbol(:decomptrend, class)] = trend_new - trend4
        else
            trend4 = get_trend_dates(Dict(1 => date_mainsample_start(m_old)), out4[trendvar],
                                     date_mainsample_start(m_old), size(out4[datavar],2),
                                     n_regs = 1)
            decomp[Symbol(:decomptrend, class)] = trend_new - trend4
        end

        total_decomp = out1[forecastvar] - out8[forecastvar]
        decomp[Symbol(:decomptotal, class)] = total_decomp
        #check && @assert total_diff ≈ out1[forecastvar][1:min_ind,:] - out4[forecastvar][1:min_ind,:]
    end

    return decomp
end

"""
```
decomposition_periods(m_new, m_old, df_new, df_old, cond_new, cond_old)
```

Returns `T`, `k`, and `H`, where:

- New model has `T` periods of data
- Old model has `T-k` periods of data
- Old and new models both forecast up to `T+H`
"""
function decomposition_periods(m_new::M, m_old::M, df_new::DataFrame, df_old::DataFrame,
                               cond_new::Symbol, cond_old::Symbol) where M<:AbstractDSGEModel
    # Number of presample periods T0 must be the same
    T0 = n_presample_periods(m_new)
    @assert n_presample_periods(m_old) == T0

    # New model has T main-sample periods
    # Old model has T-k main-sample periods
    T = n_mainsample_periods(m_new)
    k = subtract_quarters(date_forecast_start(m_new), date_forecast_start(m_old))
    @assert k >= 0

    # Number of conditional periods T1 may differ
    T1_new = cond_new == :none ? 0 : n_conditional_periods(m_new)
    T1_old = cond_old == :none ? 0 : n_conditional_periods(m_old)

    # Check DataFrame sizes
    @assert size(df_new, 1) == T0 + T + T1_new
    @assert size(df_old, 1) == T0 + T - k + T1_old

    # Old model forecasts up to T+H
    H = subtract_quarters(date_forecast_end(m_old), date_mainsample_end(m_new))

    return T, k, H
end

"""
```
decomposition_forecast(m, df, params, cond_type, keep_startdate, keep_enddate, shockdec_splitdate;
    outputs = [:forecast, :shockdec], check = false)
```

Equivalent of `forecast_one_draw` for forecast decomposition. `keep_startdate =
date_forecast_start(m_new)` corresponds to time T+1, `keep_enddate =
date_forecast_end(m_old)` to time T+H, and `shockdec_splitdate =
date_mainsample_end(m_old)` to time T-k.

Returns `out::Dict{Symbol, Array{Float64}}`, which has keys determined as follows:

- If `:forecast in outputs` or `check = true`:
  - `:forecast<class>`

- If `:shockdec in outputs`:
  - `:trend<class>`
  - `:dettrend<class>`
  - `:data<class>`: like a shockdec, but only applying smoothed shocks up to `shockdec_splitdate`
  - `:news<class>`: like a shockdec, but only applying smoothed shocks after `shockdec_splitdate`
  """
  function decomposition_forecast(m::AbstractDSGEModel, df::DataFrame, params::Vector{Float64}, cond_type::Symbol,
                                  T::Int, k::Int, H::Int; apply_altpolicy::Bool = false,
                                  outputs::Vector{Symbol} = [:forecast, :shockdec], check::Bool = false,
                                  catch_smoother_lapack::Bool = false, enforce_zlb::Bool = false,
                                  endogenous_zlb::Bool = false, set_zlb_regime_vals::Function = identity)

      regime_switching = haskey(m.settings, :regime_switching) ? get_setting(m, :regime_switching) : false

      # Compute state space
      update!(m, params)
      system = compute_system(m; tvis = haskey(get_settings(m), :tvis_information_set))

      # Initialize output dictionary
      out = Dict{Symbol, Array{Float64}}()

      # Smooth and forecast
      histstates, histshocks, histpseudo, s_0 = smooth(m, df, system, cond_type = cond_type, draw_states = false,
                                                       catch_smoother_lapack = catch_smoother_lapack)

      if :forecast in outputs || check
          s_T = histstates[:, end]

          if endogenous_zlb
              forecaststates, forecastobs, forecastpseudo, forecastshocks =
              forecast(m, system, s_T, cond_type = cond_type, enforce_zlb = false, draw_shocks = false)

              _, forecastobs, forecastpseudo, histstates, histshocks, histpseudo, s_0 =
              forecast(m, s_T, forecaststates, forecastobs, forecastpseudo, forecastshocks;
                       cond_type = cond_type, rerun_smoother = true, draw_states = false, df = df,
                       histstates = histstates, histshocks = histshocks, histpseudo = histpseudo,
                       initial_states = s_0, set_zlb_regime_vals = set_zlb_regime_vals)

              system = compute_system(m, tvis = haskey(get_settings(m), :tvis_information_set))
          else
              _, forecastobs, forecastpseudo, _ =
              forecast(m, system, s_T, cond_type = cond_type, enforce_zlb = enforce_zlb, draw_shocks = false)
          end
      end

      if regime_switching
          # Get regime indices. Just want histobs, so no need to handle ZLB regime switch
          start_date = max(date_mainsample_start(m), df[1, :date])
          end_date   = cond_type == :none ? prev_quarter(date_forecast_start(m)) : date_conditional_end(m)
          regime_inds = regime_indices(m, start_date, end_date)
          if regime_inds[1][1] < 1
              regime_inds[1] = 1:regime_inds[1][end]
          end
          cutoff = findfirst([inds[end] > T + H for inds in regime_inds])
          if !isnothing(cutoff)
              regime_inds = regime_inds[1:cutoff]
              regime_inds[end] = regime_inds[end][1]:(T + H)
          end

          # Calculate history
          histobs = zeros(n_observables(m), size(histstates,2))
          for (reg_num, reg_ind) in enumerate(regime_inds)
              histobs[:, reg_ind] = system[reg_num, :ZZ] * histstates[:, reg_ind] .+ system[reg_num, :DD]
          end
      else
          histobs = system[:ZZ] * histstates .+ system[:DD]
      end

      if :forecast in outputs || check
          out[:histforecastobs]    = hcat(histobs,    forecastobs)[:, 1:T+H]
          out[:histforecastpseudo] = hcat(histpseudo, forecastpseudo)[:, 1:T+H]
      end

      # Compute trend, dettrend, and shockdecs
      if :shockdec in outputs
          nstates = n_states_augmented(m)
          nshocks = n_shocks_exogenous(m)

          data_shocks           = zeros(nshocks, T+H)
          data_shocks[:, 1:T-k] = histshocks[:, 1:T-k]
          Tstar                 = size(histshocks, 2) # either T or T+1
          news_shocks           = zeros(nshocks, T+H)
          system0               = zero_system_constants(system)

          if regime_switching

              # Calculate trends
              if haskey(get_settings(m), :time_varying_trends) ? get_setting(m, :time_varying_trends) : false
                  _, out[:trendobs], out[:trendpseudo] = trends(m, system, start_date, end_date, cond_type)
              else
                  _, out[:trendobs], out[:trendpseudo] = trends(system)
              end

              # Calculate deterministic trends
              _, out[:dettrendobs], out[:dettrendpseudo] = deterministic_trends(m, system, s_0, T+H, 1, T+H,
                                                                                regime_inds, cond_type)

              # Applying all shocks
              _, out[:shockdecobs], out[:shockdecpseudo] =
              shock_decompositions(m, system, forecast_horizons(m; cond_type = cond_type),
                                   histshocks, 1, T + H, regime_inds, cond_type)

              # Applying ϵ_{1:T-k} and ϵ_{T-k+1:end}
              m2 = deepcopy(m)
              m2 <= Setting(:reg_forecast_start, 1)
              m2 <= Setting(:date_forecast_start, get_setting(m2, :date_mainsample_start))
              m2 <= Setting(:n_hist_regimes, 0)
              m2 <= Setting(:n_fcast_regimes, get_setting(m2, :n_regimes))
              m2 <= Setting(:reg_post_conditional_end, 1)

              _, out[:dataobs], out[:datapseudo], _ = forecast(m2, system0, zeros(nstates), data_shocks;
                                                               cond_type = :none)

              news_shocks[:, T-k+1:Tstar] = histshocks[:, T-k+1:Tstar]
              _, out[:newsobs], out[:newspseudo], _ = forecast(m2, system0, zeros(nstates), news_shocks;
                                                               cond_type = :none)
          else
              # Calculate trends
              _, out[:trendobs], out[:trendpseudo] = trends(system)

              # Calculate deterministic trends
              _, out[:dettrendobs], out[:dettrendpseudo] = deterministic_trends(system, s_0, T+H, 1, T+H)

              # Applying all shocks
              _, out[:shockdecobs], out[:shockdecpseudo] =
              shock_decompositions(system, forecast_horizons(m), histshocks, 1, T+H)

              # Applying ϵ_{1:T-k} and ϵ_{T-k+1:end}
              _, out[:dataobs], out[:datapseudo], _ = forecast(system0, zeros(nstates), data_shocks)

              news_shocks[:, T-k+1:Tstar] = histshocks[:, T-k+1:Tstar]
              _, out[:newsobs], out[:newspseudo], _ = forecast(system0, zeros(nstates), news_shocks)
          end
      end

      # Return
      return out
  end
