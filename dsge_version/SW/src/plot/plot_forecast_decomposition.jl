"""
```
make_decomp_mbs(m_new, m_old, input_type, cond_new, cond_old, class;
    individual_shocks = false)
```

Construct and return the `MeansBands` (for shockdec, trend, dettrend, hist, and
forecast) necessary to call the plotting function `shockdec` in
`plot_forecast_decomposition`.
"""
function make_decomp_mbs(m_new::M, m_old::M, input_type::Symbol,
                         cond_new::Symbol, cond_old::Symbol,
                         class::Symbol; individual_shocks::Bool = false, forecast_string_old = "", forecast_string_new = "",
                         model_decomp::Bool = false, shockdec_data_only::Bool = false) where M<:AbstractDSGEModel
    # Read in means
    input_file = get_decomp_mean_file(m_new, m_old, input_type, cond_new, cond_old, class, forecast_string_new = forecast_string_new, forecast_string_old = forecast_string_old)
    decomps = JLD2.jldopen(input_file, "r") do file
        read(file, "decomps")
    end

    # Common metadata
    comps = [:policyait, :policyeqcond, :release, :cond, :revise, :param]
    comps = model_decomp ? vcat(comps, :model) : comps
    dates = decomps[collect(keys(decomps))[1]][!,:date]
    vars  = collect(keys(get_dict(m_new, class)))

    metadata = Dict{Symbol, Any}()
    metadata[:para]            = input_type
    metadata[:indices]         = get_dict(m_new, class)
    metadata[:class]           = class
    metadata[:date_inds]       = OrderedDict(date => i for (i, date) in enumerate(dates))
    metadata[:forecast_string] = ""
    metadata[:cond_type]       = cond_new

    # Shock decomposition
    shockdec_mb = MeansBands(Dict(metadata), DataFrame(date = dates), Dict{Symbol, DataFrame}())
    shockdec_mb.metadata[:product] = :shockdec
    if individual_shocks
        for var in vars
            shocks = setdiff(propertynames(decomps[var]), vcat([:date, :dettrend, :total], comps))
            if var == vars[1]
                shockdec_mb.metadata[:shock_indices] = OrderedDict(shock => i for (i, shock) in enumerate(shocks))
            end
            for shock in shocks
                varshock = Symbol(var, "__", shock)
                shockdec_mb.means[!,varshock] = decomps[var][!,shock]
                shockdec_mb.bands[varshock] = DataFrame(date = dates)
            end
        end
    else
        shockdec_mb.metadata[:shock_indices] = OrderedDict(comp => i for (i, comp) in enumerate(comps))
        for var in vars
            for comp in comps
                varcomp = Symbol(var, "__", comp)
                shockdec_mb.means[!,varcomp] = decomps[var][!,comp]
                shockdec_mb.bands[varcomp] = DataFrame(date = dates)
            end
        end
    end

    # Trend
    trend_mb = MeansBands(Dict(metadata), DataFrame(date = dates), Dict{Symbol, DataFrame}())
    trend_mb.metadata[:product] = :trend
    for var in vars
        trend_mb.means[!,var] = zeros(length(dates))
        trend_mb.bands[var] = DataFrame(date = dates)
    end

    # Deterministic trend
    dettrend_mb = MeansBands(Dict(metadata), DataFrame(date = dates), Dict{Symbol, DataFrame}())
    dettrend_mb.metadata[:product] = :dettrend
    for var in vars
        dettrend_mb.means[!,var] = individual_shocks ? decomps[var][!,:dettrend] : zeros(length(dates))
        dettrend_mb.bands[var] = DataFrame(date = dates)
    end

    # History
    hist_inds  = dates .<= date_mainsample_end(m_new)
    hist_dates = dates[hist_inds]
    if isempty(hist_dates)
        hist_mb = MeansBands()
    else
        hist_mb = MeansBands(Dict(metadata), DataFrame(date = hist_dates), Dict{Symbol, DataFrame}())
        hist_mb.metadata[:product]   = :hist
        hist_mb.metadata[:date_inds] = OrderedDict(date => i for (i, date) in enumerate(hist_dates))
        for var in vars
            hist_mb.means[!,var] = individual_shocks && shockdec_data_only ?
                decomps[var][hist_inds, :data] + decomps[var][hist_inds, :news] : decomps[var][hist_inds, :total]
            hist_mb.bands[var] = DataFrame(date = hist_dates)
        end
    end

    # Forecast
    fcast_inds  = dates .> date_mainsample_end(m_new)
    fcast_dates = dates[fcast_inds]
    if isempty(fcast_dates)
        fcast_mb = MeansBands()
    else
        fcast_mb = MeansBands(Dict(metadata), DataFrame(date = fcast_dates), Dict{Symbol, DataFrame}())
        fcast_mb.metadata[:product]   = :forecast
        fcast_mb.metadata[:date_inds] = OrderedDict(date => i for (i, date) in enumerate(fcast_dates))
        for var in vars
            fcast_mb.means[!,var] = individual_shocks && shockdec_data_only ?
                decomps[var][fcast_inds, :data] + decomps[var][fcast_inds, :news] : decomps[var][fcast_inds, :total]
            fcast_mb.bands[var] = DataFrame(date = fcast_dates)
        end
    end

    return shockdec_mb, trend_mb, dettrend_mb, hist_mb, fcast_mb
end

"""
```
plot_forecast_decomposition(m_new, m_old, var, class, input_type,
    cond_new, cond_old; titles = [], individual_shocks = false,
    groups = shock_groupings(m_new), verbose = :low, kwargs...)

plot_forecast_decomposition(m_new, m_old, vars, class, input_type,
    cond_new, cond_old; titles = [], individual_shocks = false,
    groups = shock_groupings(m_new), verbose = :low, kwargs...)
```

Plot forecast decomposition (looks like a shock decomposition). If
`individual_shocks = false`, then the black and red lines give the total
difference and the bars give the data revision, news, and parameter
components. Otherwise, the black and red lines give the total data revision +
news component and the bars give the individual shock contributions.

The `groups` keyword argument is only used if `individual_shocks = true`.
"""
function plot_forecast_decomposition(m_new::M, m_old::M, var::Symbol, class::Symbol,
                                     input_type::Symbol, cond_new::Symbol, cond_old::Symbol;
                                     title::String = "", trend_nostates::DataFrame = DataFrame(),
                                     kwargs...) where M<:AbstractDSGEModel

    plot_forecast_decomposition(m_new, m_old, [var], class, input_type, cond_new, cond_old;
                                titles = isempty(title) ? String[] : [title], trend_nostates = trend_nostates, kwargs...)
end

function plot_forecast_decomposition(m_new::M, m_old::M, vars::Vector{Symbol}, class::Symbol,
                                     input_type::Symbol, cond_new::Symbol, cond_old::Symbol;
                                     titles::Vector{String} = String[],
                                     individual_shocks::Bool = false,
                                     groups::Vector{ShockGroup} = shock_groupings(m_new),
                                     plotroot::String = figurespath(m_new, "forecast"),
                                     verbose::Symbol = :low, forecast_string_new = "", forecast_string_old = "",
                                     model_decomp::Bool = false, trend_nostates::DataFrame = DataFrame(),
                                     shockdec_data_only::Bool = false,
                                     kwargs...) where M<:AbstractDSGEModel
    # Create MeansBands
    mbs = make_decomp_mbs(m_new, m_old, input_type, cond_new, cond_old, class,
                          individual_shocks = individual_shocks, forecast_string_new = forecast_string_new, forecast_string_old = forecast_string_old, model_decomp = model_decomp, shockdec_data_only = shockdec_data_only)

    # Create shock grouping
    if !individual_shocks
        groups = [ShockGroup("Policy-AIT", [:policyait], colorant"#9DE0AD"), # sea foam green
                  ShockGroup("Policy-eqcond", [:policyeqcond], colorant"#45ADA8"), # turquoise
                  ShockGroup("Params", [:param], colorant"green"), # turquoise
                  ShockGroup("Data-Revisions", [:revise], colorant"orange"), # turquoise
                  ShockGroup("Data-Conditional", [:cond], colorant"blue"), # turquoise
                  ShockGroup("Data-Release", [:release], colorant"#547980")] # blue gray
        if model_decomp
            push!(groups, ShockGroup("model", [:model], colorant"red"))
        end
    end

    # Get titles if not provided
    if isempty(titles)
        detexify_title = typeof(Plots.backend()) == Plots.GRBackend
        titles = map(var -> describe_series(m_new, var, class, detexify = detexify_title), vars)
    end

    # Loop through variables
    plots = OrderedDict{Symbol, Plots.Plot}()
    for (var, title) in zip(vars, titles)
        # Call recipe
        plots[var] = shockdec(var, mbs..., groups;
                              hist_label = "Historical Diff", forecast_label = "Forecast Diff",
                              ylabel = series_ylabel(m_new, var, class),
                              title = title, trend_nostates = trend_nostates, kwargs...)

        if !isempty(plotroot)
            # Save plot
            base = Symbol(:decomp, individual_shocks ? :shocks : :total, "_", var)
            output_file = get_decomp_filename(m_new, m_old, input_type, cond_new, cond_old, base, Symbol(),
                                              pathfcn = figurespath, fileformat = plot_extension(), forecast_string_new = forecast_string_new, forecast_string_old = forecast_string_old)
            output_file = joinpath(plotroot, basename(output_file))

            save_plot(plots[var], output_file, verbose = verbose)
        end
    end
end
