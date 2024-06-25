"""
```
get_scenario_input_file(m, scen::Scenario)
```

Get file name of raw scenario targets from `inpath(m, \"scenarios\")`.
"""
function get_scenario_input_file(m::AbstractDSGEModel, scen::Scenario)
    basename = string(scen.key) * "_" * scen.vintage * ".jld2"
    return inpath(m, "scenarios", basename)
end

"""
```
count_scenario_draws!(m, scen::Scenario)
```

Return the number of draws for `scen`, determined using
`get_scenario_input_file(m, scen)`, and update the
`n_draws` field of `scen` with this count.
"""
function count_scenario_draws!(m::AbstractDSGEModel, scen::Scenario)
    input_file = get_scenario_input_file(m, scen)
    draws = JLD2.jldopen(input_file, "r") do file
        dataset = read(file, "arr")
        size(dataset, 1)
    end
    scen.n_draws = draws
    return draws
end

"""
```
load_scenario_targets!(m, scen::Scenario, draw_index)
```

Add the targets from the `draw_index`th draw of the raw scenario targets to
`scen.targets`.
"""
function load_scenario_targets!(m::AbstractDSGEModel, scen::Scenario, draw_index::Int)
    if isempty(scen.target_names)
        n_periods = forecast_horizons(m)
        for target in keys(m.observables)
            scen.targets[target] = fill(NaN, n_periods)
        end
    else
        path = get_scenario_input_file(m, scen)
        raw_targets, target_inds = JLD2.jldopen(path, "r") do file
            arr = read(file, "arr")
            inds = read(file, "target_indices")
            arr[draw_index, :, :], inds
        end

        @assert collect(keys(target_inds)) == scen.target_names "Target indices in $path do not match target names in $(scen.key)"

        for (target_name, target_index) in target_inds
            scen.targets[!, target_name] = raw_targets[target_index, :]
        end
    end

    @assert collect(keys(target_inds)) == scen.target_names "Target indices in $path do not match target names in $(scen.key)"

    for (target_name, target_index) in target_inds
        scen.targets[!, target_name] = raw_targets[target_index, :]
    end

    return scen.targets
end

"""
```
get_scenario_filename(m, scen::AbstractScenario, output_var;
    pathfcn = rawpath, fileformat = :jld2, directory = "")
```

Get scenario file name of the form
`pathfcn(m, \"scenarios\", output_var * filestring * string(fileformat))`. If
`directory` is provided (nonempty), then the same file name in that directory
will be returned instead.
"""
function get_scenario_filename(m::AbstractDSGEModel, scen::AbstractScenario, output_var::Symbol;
                               pathfcn::Function = rawpath,
                               fileformat::Symbol = :jld2,
                               directory::String = "")
    filestring_addl = Vector{String}()
    if isa(scen, SingleScenario)
        push!(filestring_addl, "scen=" * string(scen.key))
    elseif isa(scen, ScenarioAggregate)
        push!(filestring_addl, "sagg=" * string(scen.key))
    end
    push!(filestring_addl, "svin=" * scen.vintage)

    base = string(output_var) * "." * string(fileformat)
    path = pathfcn(m, "scenarios", base, filestring_addl)
    if !isempty(directory)
        path = joinpath(directory, basename(path))
    end
    return path
end

"""
```
get_scenario_output_files(m, scen::SingleScenario, output_vars)
```

Return a `Dict{Symbol, String}` mapping `output_vars` to the raw simulated
scenario outputs for `scen`.
"""
function get_scenario_output_files(m::AbstractDSGEModel, scen::SingleScenario,
                                   output_vars::Vector{Symbol})
    output_files = Dict{Symbol, String}()
    for var in output_vars
        output_files[var] = get_scenario_filename(m, scen, var)
    end
    return output_files
end

"""
```
get_scenario_mb_input_file(m, scen::AbstractScenario, output_var)
```

Call `get_scenario_filename` while replacing `forecastut` and `forecast4q` in
`output_var` with `forecast`.
"""
function get_scenario_mb_input_file(m::AbstractDSGEModel, scen::AbstractScenario, output_var::Symbol)
    input_file = get_scenario_filename(m, scen, output_var)
    input_file = replace(input_file, "forecastut" => "forecast")
    input_file = replace(input_file, "forecast4q" => "forecast")
    return input_file
end

"""
```
get_scenario_mb_output_file(m, scen::AbstractScenario, output_var;
    directory = "")
```

Call `get_scenario_filename` while tacking on `\"mb\"` to the front of the base
file name.
"""
function get_scenario_mb_output_file(m::AbstractDSGEModel, scen::AbstractScenario, output_var::Symbol;
                                     directory::String = "")
    fullfile = get_scenario_filename(m, scen, output_var, pathfcn = workpath, directory = directory)
    joinpath(dirname(fullfile), "mb" * basename(fullfile))
end

"""
```
get_scenario_mb_metadata(m, scen::SingleScenario, output_var)

get_scenario_mb_metadata(m, agg::ScenarioAggregate, output_var)
```

Return the `MeansBands` metadata dictionary for `scen`.
"""
function get_scenario_mb_metadata(m::AbstractDSGEModel, scen::SingleScenario, output_var::Symbol)
    forecast_output_file = get_scenario_mb_input_file(m, scen, output_var)
    metadata = get_mb_metadata(m, :mode, :none, output_var, forecast_output_file)
    metadata[:scenario_key] = scen.key
    metadata[:scenario_vint] = scen.vintage

    return metadata
end

function get_scenario_mb_metadata(m::AbstractDSGEModel, agg::ScenarioAggregate, output_var::Symbol)
    forecast_output_file = get_scenario_mb_input_file(m, agg.scenarios[1], output_var)
    metadata = get_mb_metadata(m, :mode, :none, output_var, forecast_output_file)

    # Initialize start and end date
    start_date = date_forecast_start(m)
    end_date   = maximum(keys(metadata[:date_inds]))

    for scen in agg.scenarios
        forecast_output_file = get_scenario_mb_input_file(m, scen, output_var)
        scen_dates = JLD2.jldopen(forecast_output_file, "r") do file
            read(file, "date_indices")
        end

        # Throw error if start date for this scenario doesn't match
        if Dict([reverse(scen_date) for scen_date = pairs(scen_dates)])[1] != start_date
            error("All scenarios in agg must start from the same date")
        end

        # Update end date if necessary
        end_date = max(end_date, maximum(keys(metadata[:date_inds])))
    end

    dates = quarter_range(start_date, end_date)
    metadata[:date_inds] = OrderedDict{Date, Int}(d => i for (i, d) in enumerate(dates))
    metadata[:scenario_key] = agg.key
    metadata[:scenario_vint] = agg.vintage

    return metadata
end

"""
```
read_scenario_output(m, scen::SingleScenario, class, product, var_name)

read_scenario_output(m, agg::ScenarioAggregate, class, product, var_name)
```

Given either `scen` or `agg`, read in and return all draws of and the
appropriate reverse transform for `var_name`.

The third function that takes in two models is used for when we have scenarios from two different models.
"""
function read_scenario_output(m::AbstractDSGEModel, scen::SingleScenario, class::Symbol, product::Symbol,
                              var_name::Symbol)
    # Get filename
    filename = get_scenario_mb_input_file(m, scen, Symbol(product, class))

    fcast_dates_dict = load(filename, "date_indices")
    fcast_dates = map(x -> x[1], sort(collect(fcast_dates_dict), by = x->x[2]))

    jldopen(filename, "r") do file
        # Get index corresponding to var_name
        class_long = get_class_longname(class)
        indices = FileIO.load(filename, "$(class_long)_indices")
        var_ind = indices[var_name]

        # Read forecast outputs
        fcast_series = read_forecast_series(filename, product, var_ind)

        # Parse transform
        class_long = get_class_longname(class)
        transforms = load(filename, string(class_long) * "_revtransforms")
        transform = parse_transform(transforms[var_name])

        fcast_series, transform, fcast_dates
    end
end

function read_scenario_output(m::AbstractDSGEModel, m904::AbstractDSGEModel, m_cov::AbstractDSGEModel,
                              agg::ScenarioAggregate, class::Symbol,
                              product::Symbol, var_name::Symbol)
    # Aggregate scenarios
    nscens = length(agg.scenarios)
    agg_draws = Vector{Matrix{Float64}}(undef, nscens)
    scen_dates = Vector{Date}(undef, nscens)

    # If not sampling, initialize vector to record number of draws in each
    # scenario in order to update `agg.proportions` and `agg.total_draws` at the
    # end
    if !agg.sample
        n_scen_draws = zeros(Int, nscens)
    end

    # Initialize transform so it can be assigned from within the following for
    # loop. Each transform read in from read_scenario_output will be the
    # same. We just want to delegate the transform parsing to the recursive
    # read_scenario_output call.
    transform = identity

    for (i, scen) in enumerate(agg.scenarios)
        if in(:scenarios, fieldnames(typeof(scen))) #length(scen.scenarios)>1
            scen_draws, transform, scen_dates = read_scenario_output(m, m904, m_cov, scen, class, product, var_name)
        else
            if scen.key==:bor8 || scen.key==:bor9 || scen.key==:bor8_02 || scen.key==:bor9_02
                if var_name==:obs_corepce
                    var_name = :obs_gdpdeflator
                end
                # Recursively read in scenario draws
                @show m904
                scen_draws, transform, scen_dates = read_scenario_output(m904, scen, class, product, var_name)
            elseif scen.key==:bor10 || scen.key == :bor11
                @show m_cov
                scen_draws, transform, scen_dates = read_scenario_output(m_cov, scen, class, product, var_name)
            else
                @show m
                # Recursively read in scenario draws
                scen_draws, transform, scen_dates = read_scenario_output(m, scen, class, product, var_name)
            end
        end
        # Sample if desired
        agg_draws[i]  = if agg.sample
            pct = agg.proportions[i]
            actual_ndraws = size(scen_draws, 1)
            desired_ndraws = convert(Int, round(pct * agg.total_draws))

            sampled_inds = if agg.replace
                sample(1:actual_ndraws, desired_ndraws, replace = true)
            else
                if desired_ndraws == 0
                    Int[]
                else
                    quotient  = convert(Int, floor(actual_ndraws / desired_ndraws))
                    remainder = actual_ndraws % desired_ndraws
                    vcat(repeat(1:actual_ndraws, quotient),
                         sample(1:actual_ndraws, remainder, replace = false))
                end
            end
            sort!(sampled_inds)
            scen_draws[sampled_inds, :]
        else
            # Record number of draws in this scenario
            n_scen_draws[i] = size(scen_draws, 1)
            scen_draws
        end
    end

    # Stack draws from all component scenarios
    fcast_series = cat(agg_draws..., dims = 1)

    # If not sampling, update `agg.proportions` and `agg.total_draws`
    if !agg.sample
        agg.total_draws = sum(n_scen_draws)
        agg.proportions = n_scen_draws ./ agg.total_draws
    end


    return fcast_series, transform, scen_dates
end


#=function read_scenario_output(m::AbstractDSGEModel, m904::AbstractDSGEModel, agg::ScenarioAggregate, class::Symbol,
                              product::Symbol, var_name::Symbol)
    # Aggregate scenarios
    nscens = length(agg.scenarios)
    agg_draws = Vector{Matrix{Float64}}(nscens)

    # If not sampling, initialize vector to record number of draws in each
    # scenario in order to update `agg.proportions` and `agg.total_draws` at the
    # end
    if !agg.sample
        n_scen_draws = zeros(Int, nscens)
    end

    # Initialize transform so it can be assigned from within the following for
    # loop. Each transform read in from read_scenario_output will be the
    # same. We just want to delegate the transform parsing to the recursive
    # read_scenario_output call.
    transform = identity

    for (i, scen) in enumerate(agg.scenarios)
        @show scen
        @show scen.key
        if in(:scenarios, fieldnames(scen)) #length(scen.scenarios)>1
            scen_draws, transform = read_scenario_output(m, m904, scen, class, product, var_name)
        else
            # If BOR8 or BOR9 (switching or non-switching), read GDP Deflator instead of core PCE
            if scen.key==:bor8 || scen.key==:bor9 || scen.key==:bor8_02 || scen.key==:bor9_02
                if var_name==:obs_corepce
                    var_name = :obs_gdpdeflator
                end
                # Recursively read in scenario draws
                scen_draws, transform = read_scenario_output(m904, scen, class, product, var_name)
            else
                # Recursively read in scenario draws
                scen_draws, transform = read_scenario_output(m, scen, class, product, var_name)
            end
        end
        # Sample if desired
        agg_draws[i] = if agg.sample
            pct = agg.proportions[i]
            actual_ndraws = size(scen_draws, 1)
            desired_ndraws = convert(Int, round(pct * agg.total_draws))

            sampled_inds = if agg.replace
                sample(1:actual_ndraws, desired_ndraws, replace = true)
            else
                if desired_ndraws == 0
                    Int[]
                else
                    quotient  = convert(Int, floor(actual_ndraws / desired_ndraws))
                    remainder = actual_ndraws % desired_ndraws
                    vcat(repmat(1:actual_ndraws, quotient),
                         sample(1:actual_ndraws, remainder, replace = false))
                end
            end
            sort!(sampled_inds)
            scen_draws[sampled_inds, :]
        else
            # Record number of draws in this scenario
            n_scen_draws[i] = size(scen_draws, 1)
            scen_draws
        end
    end

    # Stack draws from all component scenarios
    fcast_series = cat(1, agg_draws...)

    # If not sampling, update `agg.proportions` and `agg.total_draws`
    if !agg.sample
        agg.total_draws = sum(n_scen_draws)
        agg.proportions = n_scen_draws ./ agg.total_draws
    end

    return fcast_series, transform
end=#

function read_scenario_output(m::AbstractDSGEModel, agg::ScenarioAggregate, class::Symbol,
                              product::Symbol, var_name::Symbol)
    # Aggregate scenarios
    nscens = length(agg.scenarios)
    agg_draws = Vector{Matrix{Float64}}(undef, nscens)
    agg_dates = Vector{Vector{Date}}(undef, nscens)

    # If not sampling, initialize vector to record number of draws in each
    # scenario in order to update `agg.proportions` and `agg.total_draws` at the
    # end
    if !agg.sample
        n_scen_draws = zeros(Int, nscens)
    end

    # Initialize transform so it can be assigned from within the following for
    # loop. Each transform read in from read_scenario_output will be the
    # same. We just want to delegate the transform parsing to the recursive
    # read_scenario_output call.
    transform = identity

    for (i, scen) in enumerate(agg.scenarios)
        # Recursively read in scenario draws
        scen_draws, transform, scen_dates = read_scenario_output(m, scen, class, product, var_name)

        # Sample if desired
        agg_draws[i], agg_dates[i] = if agg.sample
            pct = agg.proportions[i]
            actual_ndraws = size(scen_draws, 1)
            desired_ndraws = convert(Int, round(pct * agg.total_draws))

            sampled_inds = if agg.replace
                sample(1:actual_ndraws, desired_ndraws, replace = true)
            else
                if desired_ndraws == 0
                    Int[]
                else
                    quotient  = convert(Int, floor(actual_ndraws / desired_ndraws))
                    remainder = actual_ndraws % desired_ndraws
                    vcat(repeat(1:actual_ndraws, outer = quotient),
                         sample(1:actual_ndraws, remainder, replace = false))
                end
            end
            sort!(sampled_inds)
            scen_draws[sampled_inds, :], scen_dates
        else
            # Record number of draws in this scenario
            n_scen_draws[i] = size(scen_draws, 1)
            scen_draws, scen_dates
        end
    end

    # Stack draws from all component scenarios
    fcast_series = cat(agg_draws..., dims = 1)

   #= fcast_dates_dict = load(get_scenario_mb_input_file(m, scen, Symbol(:forecast, class)), "date_indices")
    fcast_dates = map(x -> x[1], sort(collect(fcast_draws_dates_dict), by = x->x[2]))=#

    # If not sampling, update `agg.proportions` and `agg.total_draws`
    if !agg.sample
        agg.total_draws = sum(n_scen_draws)
        agg.proportions = n_scen_draws ./ agg.total_draws
    end

    return fcast_series, transform, agg_dates
end


"""
```
read_scenario_mb(m, scen::AbstractScenario, output_var; directory = "")
```

Read in an alternative scenario `MeansBands` object.
"""
function read_scenario_mb(m::AbstractDSGEModel, scen::AbstractScenario, output_var::Symbol;
                          directory::String = "")
    filepath = get_scenario_mb_output_file(m, scen, output_var, directory = directory)
    read_mb(filepath)
end
