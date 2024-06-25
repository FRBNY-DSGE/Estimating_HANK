"""
```
plot_forecast_comparison(m_old, m_new, var, class, input_type, cond_type;
    save_as_csv = false, title = "", use_bdd_new::Symbol = :unbdd,
    use_bdd_old::Symbol = :unbdd, kwargs...)

plot_forecast_comparison(m_old, m_new, vars, class, input_type, cond_type;
    input_type_old = input_type, cond_type_old = cond_type,
    forecast_string = "", forecast_string_old = forecast_string,
    use_bdd_new = :unbdd, use_bdd_old = :unbdd,
    bands_pcts = [\"90.0%\"],
    old_names = Dict(:hist => "", :forecast => \"Old Forecast\"),
    new_names = Dict(:hist => "", :forecast => \"New Forecast\"),
    old_colors = Dict(:hist => :grey, :forecast => :blue, :bands => :blue),
    new_colors = Dict(:hist => :black, :forecast => :red, :bands => :red),
    old_alphas = Dict{Symbol, Float64}(),
    new_alphas = Dict{Symbol, Float64}(),
    old_styles = Dict(:hist => :dash, :forecast => :dash, :bands => :dash),
    new_styles = Dict(:hist => :solid, :forecast => :solid, :bands => :solid),
    plotroot = "", titles = [], verbose = :low,
    save_as_csv = false)
```

Plot forecasts from `m_old` and `m_new` of `var` or `vars`.

### Inputs

- `m_old::AbstractDSGEModel`
- `m_new::AbstractDSGEModel`
- `var::Symbol` or `vars::Vector{Symbol}`: e.g. `:obs_gdp` or `[:obs_gdp,
  :obs_nominalrate]`
- `class::Symbol`
- `input_type::Symbol`
- `cond_type::Symbol`

### Keyword Arguments

- `input_type_old::Symbol`
- `cond_type_old::Symbol`
- `forecast_string::String`
- `forecast_string_old::String`
- `use_bdd_new::Symbol`: specifies combination of bounded/unbounded means &
    bounded/unbounded bands for the new forecast
- `use_bdd_old::Symbol`: specifies combination of bounded/unbounded means &
    bounded/unbounded bands for the old forecast
- `bands_pcts::Vector{String}`: which bands to plot
- `old_names::Dict{Symbol, String}`: maps keys `[:hist, :forecast, :bands]` to
  labels for old forecast
- `new_names::Dict{Symbol, String}`
- `old_colors::Dict{Symbol, Any}`: maps keys `[:hist, :forecast, :bands]` to
  colors for old forecast
- `new_colors::Dict{Symbol, Any}`
- `old_alphas::Dict{Symbol, Float64}`: maps keys `[:hist, :forecast, :bands]` to
  transparency values for old forecast
- `new_alphas::Dict{Symbol, Float64}`
- `old_styles::Dict{Symbol, Symbol}`: maps keys `[:hist, :forecast, :bands]` to
  linestyles for old forecast
- `new_styles::Dict{Symbol, Symbol}`
- `plotroot::String`: if nonempty, plots will be saved in that directory
- `title::String` or `titles::Vector{String}`
- `verbose::Symbol`
- `save_as_csv::Bool`: if true, save plot data to csvs

### Output

- `p::Plot` or `plots::OrderedDict{Symbol, Plot}`
"""
function plot_forecast_comparison(m_old::AbstractDSGEModel, m_new::AbstractDSGEModel,
                                  var::Symbol, class::Symbol,
                                  input_type::Symbol, cond_type::Symbol;
                                  title::String = "",
	                			  save_as_csv::Bool = false,
                                  use_bdd_new::Symbol = :unbdd,
                                  use_bdd_old::Symbol = :unbdd,
                                  weights::Array{Float64} = [],
                                  kwargs...)

    plots = plot_forecast_comparison(m_old, m_new, [var], class, input_type, cond_type;
                                     titles = isempty(title) ? String[] : [title],
                                     save_as_csv = save_as_csv,
                                     use_bdd_new = use_bdd_new,
                                     use_bdd_old = use_bdd_old,
                                     weights = weights,
				                     kwargs...)
    return plots[var]
end


function plot_forecast_comparison(m_old::AbstractDSGEModel, m_new::AbstractDSGEModel,
                                  var_old::Symbol, class_old::Symbol,
                                  var::Symbol, class::Symbol,
                                  input_type::Symbol, cond_type::Symbol;
                                  title::String = "",
				                  save_as_csv::Bool = false,
                                  use_bdd_new::Symbol = :unbdd,
                                  use_bdd_old::Symbol = :unbdd,
                                  weights::Array{Float64} = [],
                                  kwargs...)

    plots = plot_forecast_comparison(m_old, m_new, [var_old], class_old,
                                     [var], class, input_type, cond_type;
                                     titles = isempty(title) ? String[] : [title],
                                     save_as_csv = save_as_csv,
                                     use_bdd_new = use_bdd_new,
                                     use_bdd_old = use_bdd_old,
                                     weights = weights,
				                     kwargs...)
    return plots[var]
end

function plot_forecast_comparison(m_old::AbstractDSGEModel, m_new::AbstractDSGEModel,
                                  vars::Vector{Symbol}, class::Symbol,
                                  input_type::Symbol, cond_type::Symbol;
                                  input_type_old::Symbol = input_type,
                                  cond_type_old::Symbol = cond_type,
                                  forecast_string::String = "",
                                  forecast_string_old::String = forecast_string,
                                  use_bdd_new::Symbol = :unbdd,
                                  use_bdd_old::Symbol = :unbdd,
                                  bands_pcts::Vector{String} = ["90.0%"],
                                  old_names = Dict(:hist => "", :forecast => "Old Forecast"),
                                  new_names = Dict(:hist => "", :forecast => "New Forecast"),
                                  old_colors = Dict(:hist => :grey, :forecast => :blue, :bands => :blue),
                                  new_colors = Dict(:hist => :black, :forecast => :red, :bands => :red),
                                  old_alphas = Dict{Symbol, Float64}(),
                                  new_alphas = Dict{Symbol, Float64}(),
                                  old_styles = Dict(:hist => :dash, :forecast => :dash, :bands => :dash),
                                  new_styles = Dict(:hist => :solid, :forecast => :solid, :bands => :solid),
                                  plotroot::String = "",
                                  titles::Vector{String} = String[],
                                  verbose::Symbol = :low,
				                  save_as_csv::Bool = false,
                                  weights::Array{Float64} = [],
                                  kwargs...)
    # Read in MeansBands
    histold = read_mb(m_old, input_type_old, cond_type_old, Symbol(:hist, class),
                      forecast_string = forecast_string_old)
    forecastold = read_mb(m_old, input_type_old, cond_type_old, Symbol(:forecast, class),
                          forecast_string = forecast_string_old, use_bdd = use_bdd_old)
    histnew = read_mb(m_new, input_type, cond_type, Symbol(:hist, class),
                      forecast_string = forecast_string)
    forecastnew = read_mb(m_new, input_type, cond_type, Symbol(:forecast, class),
                          forecast_string = forecast_string, use_bdd = use_bdd_new)

    # Get titles if not provided
    if isempty(titles)
        detexify_title = typeof(Plots.backend()) == Plots.GRBackend
        titles = map(var -> describe_series(m_new, var, class, detexify = detexify_title), vars)
    end

    # Loop through variables
    plots = OrderedDict{Symbol, Plots.Plot}()
    for (var, title) in zip(vars, titles)
    	# Setup for saving to CSV
    	df_plot_data = DataFrame()

        # Call recipe
        plots[var] = histforecast(var, histold, forecastold;
		     		              df_plot_data = df_plot_data, save_as_csv = save_as_csv,
                                  names = old_names, colors = old_colors,
                                  alphas = old_alphas, styles = old_styles,
                                  bands_pcts = bands_pcts, bands_style = :line,
                                  title = title, ylabel = series_ylabel(m_new, var, class),
                                  kwargs...)
	    if save_as_csv
   	        df_plot_data = df_plot_data[setdiff(names(df_plot_data), [:mean_history])]
            sort!(df_plot_data, [:dates])
            rename!(df_plot_data, :mean_forecast => Symbol("mean_forecast_old"))
	    end

        histforecast!(var, histnew, forecastnew;
	                  df_plot_data = df_plot_data, save_as_csv = save_as_csv,
                      names = new_names, colors = new_colors,
                      alphas = new_alphas, styles = new_styles,
                      bands_pcts = bands_pcts, bands_style = :line, kwargs...)

        if save_as_csv
            if !isdir("blog_plot_data")
                mkdir("blog_plot_data")
            end
            rename!(df_plot_data, :mean_forecast => Symbol("mean_forecast_new"))
            sort!(df_plot_data, [:dates])
            CSV.write(string("blog_plot_data/", get_setting(m_new, :data_vintage),
                             "_", replace(replace(title, " " => "_"), "," => ""), "_", var,
                             join(string.(weights), "_"), ".csv"), df_plot_data)
        end



        # Save plot
        if !isempty(plotroot)
            output_file = joinpath(plotroot, "forecastcomp_" * detexify(string(var)) * "." *
                                   string(plot_extension()))
            save_plot(plots[var], output_file, verbose = verbose)
        end
    end
    return plots
end


function plot_forecast_comparison(m_old::AbstractDSGEModel, m_new::AbstractDSGEModel,
                                  vars_old::Vector{Symbol}, class_old::Symbol,
                                  vars::Vector{Symbol}, class::Symbol,
                                  input_type::Symbol, cond_type::Symbol;
                                  input_type_old::Symbol = input_type,
                                  cond_type_old::Symbol = cond_type,
                                  forecast_string::String = "",
                                  forecast_string_old::String = forecast_string,
                                  use_bdd_new::Symbol = :unbdd,
                                  use_bdd_old::Symbol = :unbdd,
                                  bands_pcts::Vector{String} = ["90.0%"],
                                  old_names = Dict(:hist => "", :forecast => "Old Forecast"),
                                  new_names = Dict(:hist => "", :forecast => "New Forecast"),
                                  old_colors = Dict(:hist => :grey, :forecast => :blue, :bands => :blue),
                                  new_colors = Dict(:hist => :black, :forecast => :red, :bands => :red),
                                  old_alphas = Dict{Symbol, Float64}(),
                                  new_alphas = Dict{Symbol, Float64}(),
                                  old_styles = Dict(:hist => :dash, :forecast => :dash, :bands => :dash),
                                  new_styles = Dict(:hist => :solid, :forecast => :solid, :bands => :solid),
                                  plotroot::String = "",
                                  titles::Vector{String} = String[],
                                  verbose::Symbol = :low,
				                  save_as_csv::Bool = false,
                                  weights::Array{Float64} = [],
                                  kwargs...)
    # Read in MeansBands
    histold = read_mb(m_old, input_type_old, cond_type_old, Symbol(:hist, class_old),
                      forecast_string = forecast_string_old)
    forecastold = read_mb(m_old, input_type_old, cond_type_old, Symbol(:forecast, class_old),
                          forecast_string = forecast_string_old, use_bdd = use_bdd_old)
    histnew = read_mb(m_new, input_type, cond_type, Symbol(:hist, class),
                      forecast_string = forecast_string)
    forecastnew = read_mb(m_new, input_type, cond_type, Symbol(:forecast, class),
                          forecast_string = forecast_string, use_bdd = use_bdd_new)

    # Get titles if not provided
    if isempty(titles)
        detexify_title = typeof(Plots.backend()) == Plots.GRBackend
        titles = map(var -> describe_series(m_new, var, class, detexify = detexify_title), vars)
    end

    # Loop through variables
    plots = OrderedDict{Symbol, Plots.Plot}()
    for (var, var_old, title) in zip(vars, vars_old, titles)
    	# Setup for saving to CSV
    	df_plot_data = DataFrame()

        # Call recipe
        plots[var] = histforecast(var_old, histold, forecastold;
		     		              df_plot_data = df_plot_data, save_as_csv = save_as_csv,
                                  names = old_names, colors = old_colors,
                                  alphas = old_alphas, styles = old_styles,
                                  bands_pcts = bands_pcts, bands_style = :line,
                                  title = title, ylabel = series_ylabel(m_new, var, class),
                                  kwargs...)
	    if save_as_csv
            @show names(df_plot_data)
   	        df_plot_data = df_plot_data[!, setdiff(names(df_plot_data), [:mean_history])]
            @show names(df_plot_data)
            rename!(df_plot_data, :mean_forecast => Symbol("mean_forecast_old"))
	    end

        histforecast!(var, histnew, forecastnew;
	                  df_plot_data = df_plot_data, save_as_csv = save_as_csv,
                      names = new_names, colors = new_colors,
                      alphas = new_alphas, styles = new_styles,
                      bands_pcts = bands_pcts, bands_style = :line, kwargs...)

        if save_as_csv
            if !isdir("blog_plot_data")
                mkdir("blog_plot_data")
            end
            rename!(df_plot_data, :mean_forecast => Symbol("mean_forecast_new"))
            select!(df_plot_data, :dates, :mean_forecast_old, :mean_history, :mean_forecast_new)
            CSV.write(string("blog_plot_data/", get_setting(m_new, :data_vintage),
                             "_", replace(replace(title, " " => "_"), "," => ""), "_", var,
                             "_", join(string.(weights), "_"), ".csv"), df_plot_data)
        end

        # Save plot
        if !isempty(plotroot)
            output_file = joinpath(plotroot, "forecastcomp_" * detexify(string(var)) * "." *
                                   string(plot_extension()))
            save_plot(plots[var], output_file, verbose = verbose)
        end
    end
    return plots
end
