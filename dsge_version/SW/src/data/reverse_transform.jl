"""
Methods that read in all draws of forecast output:
```
reverse_transform(m, input_type, cond_type, class, product, vars;
                           forecast_string = "", fourquarter = false, verbose = :low)

reverse_transform(path, class, product, var, rev_transform;
                           pop_growth = [], fourquarter = false, verbose = :low)
```

Methods that take a single draw of a forecast output that is already in memory:
```
reverse_transform(m::AbstractDSGEModel, untransformed::Matrix, start_date,
                           row_labels, class; fourquarter = false,
                           verbose = :low)

reverse_transform(m, untransformed::DataFrame, class; fourquarter = false, verbose = :low)
```

Lowest level:
```
reverse_transform(y, rev_transform; fourquarter = false, y0 = NaN, y0s = [],
    pop_growth = [])
```

The purpose of the `reverse_transform` function is to transform raw
forecast outputs from model units into final output units. We
provide methods at various levels of abstraction.

At the highest level, the first two methods listed above read in raw
forecast output from a file and transform **all draws** corresponding to
the given `vars` or `var`. The model-agnostic function takes the name of the
file containing the forecast output and a reverse transform
function. The model-dependent function determines the filepath and
appropriate reverse transform from the information contained in the
model.

The second two methods take in raw forecast output that is already in
memory corresponding to a **single draw**. In the first method, the
user must specify `row_labels`, which label the rows of the `untransformed`
matrix. Each element of `row_labels` must be a valid observable or
pseudoobservable key that corresponds to the row it labels. These
methods return a `DataFrame` of all transformed series.

The lowest-level method requires the caller to provide additional
information. It is applied to **one or many draws** of a single
series. This method is called by the higher-level functions as well as
`compute_means_bands`.

## Inputs
- `m::AbstractDSGEModel`
- `input_type::Symbol`: see `?forecast_one`
- `cond_type::Symbol`: see `?forecast_one`
- `class::Symbol`: Indicates whether the reverse transform should be
                   applied to observables or pseudo observables
- `product::Symbol`: Indicates the product the reverse transform should be applied to (e.g. `:hist`,
                    `:forecast`)
- `vars::Vector{Symbol}`: Names of specific series for which to apply the reverse transform
- `untransformed`: `nperiods` x `nseries` `Matrix` or `DataFrame` of
                   series in model units.
- `y`:`ndraws` x `nperiods` array of a series in model units.
- `rev_transform::Function`: a function used to to transform a variable

## Keyword arguments
- `fourquarter::Bool`: produce four-quarter results instead of annualized quarter-to-quarter results
- `verbose::Symbol`: level of verbosity: `:high`, `:low`, `:none`
- `y0::AbstractFloat`: value of `y` in the period prior to the
  first period. This is used to compute growth rates for the first
  period.
- `y0s::Vector{AbstractFloat}`: contains the value of `y` in the `n`
  periods prior to the first period that `y` contains.
- `pop_growth::Vector{AbstractFloat}`: a vector of population growth
  data. This is required when the reverse transform associated with
  `untransformed` or `y` adjusts for population.
"""
function reverse_transform(m::AbstractDSGEModel, untransformed::AbstractArray, start_date::Dates.Date,
                           vars::Vector{Symbol}, class::Symbol;
                           fourquarter::Bool = false, accumulate::Bool = false,
                           verbose::Symbol = :low)
    df = DataFrame()
    nperiods = size(untransformed, 2)
    end_date = iterate_quarters(start_date, nperiods - 1)
    df[!, :date] = quarter_range(start_date, end_date)

    for (ind, var) in enumerate(vars)
        series = untransformed[ind, :]
        df[!, var] = series
    end

    reverse_transform(m, df, class; fourquarter = fourquarter,
                      accumulate = accumulate, verbose = verbose)
end

function reverse_transform(m::AbstractDSGEModel, untransformed::DataFrame, class::Symbol;
                           fourquarter::Bool = false, accumulate::Bool = false, verbose::Symbol = :low)
    # Dates
    @assert (:date in propertynames(untransformed)) "untransformed must have a date column"
    date_list  = convert(Vector{Dates.Date}, untransformed[!,:date])
    start_date = date_list[1]
    end_date   = date_list[end]

    # Get mapping from variable name to Observable or PseudoObservable instance
    if class == :obs
        dict = m.observable_mappings
    elseif class == :pseudo
        dict = m.pseudo_observable_mappings
    else
        error("Class $class does not have reverse transformations")
    end

    # Prepare population series
    population_data, population_forecast = load_population_growth(m, verbose = verbose)
    population_series = get_population_series(:population_growth, population_data,
                                              population_forecast, start_date, end_date)
    population_series = convert(Vector{Float64}, population_series)
    if fourquarter
        population_series = vcat(fill(NaN,3), population_series)
    end

    # Apply reverse transform
    transformed = DataFrame()
    transformed[!,:date] = untransformed[!,:date]

    vars = setdiff(propertynames(untransformed), [:date])
    for var in vars
        y = convert(Vector{Float64}, untransformed[!,var])
        rev_transform = dict[var].rev_transform
        if fourquarter
            rev_transform = get_transform4q(rev_transform)
        elseif accumulate
            rev_transform = get_transformlvl(rev_transform)
        end
        transformed[!,var] = reverse_transform(y, rev_transform; fourquarter = fourquarter,
                                               accumulate = accumulate,
                                               pop_growth = population_series)
    end
    return transformed
end

function reverse_transform(y::AbstractArray, rev_transform::Function;
                           fourquarter::Bool = false, accumulate::Bool = false,
                           y0 = missing, y0s::Vector = Vector(undef, 0),
                           pop_growth::Vector = Vector(undef, 0)) where T<:AbstractFloat
    if fourquarter
        if rev_transform in [loggrowthtopct_4q_percapita, loggrowthtopct_4q,
                             loggrowthtopct_4q_approx]
            # Sum growth rates y_{t-3}, y_{t-2}, y_{t-1}, and y_t
            y0s = isempty(y0s) ? fill(missing, 3) : y0s
            if rev_transform == loggrowthtopct_4q_percapita
                rev_transform(y, pop_growth, y0s)
            else
                rev_transform(y, y0s)
            end
        elseif rev_transform in [logleveltopct_4q_percapita, logleveltopct_4q,
                                 logleveltopct_4q_approx]
            # Divide log levels y_t by y_{t-4}
            y0s = isempty(y0s) ? fill(missing, 4) : y0s
            if rev_transform == logleveltopct_4q_percapita
                rev_transform(y, pop_growth, y0s)
            else
                rev_transform(y, y0s)
            end
        elseif rev_transform in [quartertoannual, identity]
            rev_transform(y)
        else
            error("Invalid 4-quarter reverse transform: $rev_transform")
        end
    # elseif accumulate # Rewrite for various cases of transformations
    #     if rev_transform in [logleveltopct_annualized_percapita]
    #         rev_transform(y, pop_growth, y0)
    #     elseif rev_transform in [loggrowthtopct_annualized_percapita, loggrowthtopct_percapita]
    #         rev_transform(y, pop_growth)
    #     elseif rev_transform in [logleveltopct_annualized, logleveltopct_annualized_approx]
    #         rev_transform(y, y0)
    #     elseif rev_transform in [loggrowthtopct_annualized, loggrowthtopct, quartertoannual, identity]
    #         rev_transform(y)
    #     else
    #         error("Invalid reverse transform: $rev_transform")
    #     end
    else
        if rev_transform in [logleveltopct_annualized_percapita]
            rev_transform(y, pop_growth, y0)
        elseif rev_transform in [loggrowthtopct_annualized_percapita, loggrowthtopct_percapita]
            rev_transform(y, pop_growth)
        elseif rev_transform in [logleveltopct_annualized, logleveltopct_annualized_approx]
            rev_transform(y, y0)
        elseif rev_transform in [loggrowthtopct_annualized, loggrowthtopct, quartertoannual, identity]
            rev_transform(y)
        else
            error("Invalid reverse transform: $rev_transform")
        end
    end
end

# function reverse_transform(m::AbstractDSGEModel, input_type::Symbol, cond_type::Symbol,
#                            class::Symbol, product::Symbol, vars::Vector{Symbol};
#                            forecast_string = "", fourquarter::Bool = false, verbose::Symbol = :low)
#     # Figure out dates
#     start_date, end_date  = if product in [:hist]
#         date_mainsample_start(m), date_mainsample_end(m)
#     elseif product in [:forecast, :bddforecast, :forecast4q, :bddforecast4q]
#         date_forecast_start(m), date_forecast_end(m)
#     elseif product in [:shockdec, :trend, :dettrend]
#         date_shockdec_start(m), date_shockdec_end(m)
#     else
#         error("reverse_transform not supported for product $product")
#     end

#     # Load population series
#     population_data, population_forecast = load_population_growth(m,
#                                                                   verbose = verbose)
#     population_series = get_population_series(:population_growth, population_data,
#                                               population_forecast,
#                                               start_date, end_date)
#     if sum(ismissing.(population_series)) > 0
#         error("Population series being loaded has a missing value.")
#     else
#         population_series = Vector{Float64}(population_series)
#     end

#     if class == :obs
#         dict = m.observable_mappings
#     elseif class == :pseudo
#         dict = m.pseudo_observable_mappings
#     else
#         error("Class $class does not have reverse transformations")
#     end

#     # Determine which file to read in
#     path       = get_forecast_filename(m, input_type, cond_type, Symbol(product, class);
#                                  forecast_string = forecast_string)

#     # Apply reverse transform to each desired variable
#     results    = Dict{Symbol, Array{Float64,2}}()
#     for var in vars
#         series = dict[var]
#         transformed = reverse_transform(path, class, product,
#                                         var, series.rev_transform,
#                                         pop_growth = population_series,
#                                         fourquarter = fourquarter)
#         results[var] = transformed
#     end

#     return results
# end

# function reverse_transform(path::String, class::Symbol, product::Symbol,
#                            var::Symbol, rev_transform::Function;
#                            pop_growth::Vector{Float64} = Vector{Float64}(),
#                            fourquarter::Bool = false)

#     # Read in the draws for this variable
#     draws = JLD2.jldopen(path, "r") do jld
#         read_forecast_output(jld, class, product, var)
#     end

#     # Transform and return
#     transformed = reverse_transform(draws, rev_transform, pop_growth = pop_growth,
#                                     fourquarter = fourquarter)

#     return transformed
# end
