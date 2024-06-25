"""
```
usual_settings!(m, vint; cdvt = vint, dsid = data_id(m), cdid = cond_id(m),
    fcast_date = Dates.lastdayofquarter(Dates.today()),
    altpolicy = AltPolicy(:historical, eqcond, solve))
```

Apply usual defaults for the following settings:

- `dataroot`: \"../../data/input_data/\"
- `saveroot` \"../../data/\"
- `data_vintage` and `cond_vintage`: given by input argument `vint`
- `date_forecast_start` and `date_conditional_end`: given by kwarg `fcast_date`
- `use_population_forecast`: `true`
- `alternative_policy`: given by input argument `altpolicy`. If this argument is
  specified, then `altpolicy_settings!` and `altpolicy.setup` are also called.
"""
function usual_settings!(m::AbstractModel, vint::String;
                         cdvt::String = vint,
                         cdvt_filestring::Bool = false,
                         dsid::Int = data_id(m),
                         cdid::Int = cond_id(m),
                         fcast_date::Dates.Date = Dates.lastdayofquarter(Dates.today()),
                         altpolicy::AltPolicy = AltPolicy(:historical, eqcond, solve),
                         cond_full_names::Array{Symbol} = [:obs_gdp, :obs_corepce, :obs_spread, :obs_longrate],
                         cond_semi_names::Array{Symbol} = [:obs_spread, :obs_longrate])
    m <= Setting(:dataroot, "../../data/input_data/")
    m <= Setting(:saveroot, "../../data/")
    m <= Setting(:data_vintage, vint)
    if cdvt_filestring
        m <= Setting(:cond_vintage, cdvt, true, "cdvt", "")
    else
        m <= Setting(:cond_vintage, cdvt)
    end
    m <= Setting(:data_id, dsid)
    m <= Setting(:cond_id, cdid)
    m <= Setting(:date_forecast_start,  fcast_date)
    m <= Setting(:date_conditional_end, fcast_date)
    m <= Setting(:use_population_forecast, true)

    m <= Setting(:cond_full_names, cond_full_names)
    m <= Setting(:cond_semi_names, cond_semi_names)
    m <= Setting(:date_conditional_end, fcast_date)

    if altpolicy.key != :historical
        altpolicy_settings!(m, altpolicy)
    end

    # Set parameters for conditional forecast
    for k in [:σ_biidc_prop, :σ_φ_prop, :σ_ziid_prop]
        if k in [m.parameters[i].key for i in 1:length(m.parameters)]
            set_regime_fixed!(m[k], 1, true)
            set_regime_fixed!(m[k], 2, true)
        end
    end

    m <= Setting(:sampling_method, :SMC)
    m <= Setting(:forecast_jstep, 1)
end


function altpolicy_settings!(m::AbstractModel, altpolicy::AltPolicy)
    # I/O and data settings
    m <= Setting(:dataroot, "../../data/input_data/")
    m <= Setting(:saveroot, "../../data/")
    m <= Setting(:dataroot, dataroot, "Input data directory path")
    m <= Setting(:saveroot, saveroot, "Output data directory path")
    m <= Setting(:use_population_forecast, true)
    m <= Setting(:alternative_policy, altpolicy, true, "apol", "Alternative policy")
end

function use_estimation_vintage!(m::AbstractModel, input_type::Symbol)
    # Determine estimation vintage
    est_file = get_forecast_input_file(m, input_type)
    est_vint = match(r"\d{6}", basename(est_file)).match

    # Cache current vintage and update to estimation vintage
    fcast_vint = data_vintage(m)
    m <= Setting(:data_vintage, est_vint)
    return fcast_vint
end

"""
```
rd_draft_prior_posterior_tables(m,
    outdir = \"../../tables/\",
    prior_file = "", posterior_file = "", params = [])
```

Write prior and posterior tables to `outdir`.
Options to pass in prior/posterior filenames (with paths) and posterior distribution.
"""
function rd_draft_prior_posterior_tables(m::AbstractModel;
                                         outdir::String = "../../tables/",
                                         prior_file::String = "",
                                         posterior_file::String = "", params::Array{Float64} = Vector{Float64}())

    # Switch to estimation vintage
    fcast_vint = use_estimation_vintage!(m, :full)

    # Process prior file and posterior file names
    if isempty(prior_file)
        prior_file = tablespath(m, "estimate", "priors.tex")
        prior_file = replace(prior_file, dirname(prior_file) => outdir)
    end
    if isempty(posterior_file)
        posterior_file = tablespath(m, "estimate", "posterior.tex")
        posterior_file = replace(posterior_file, dirname(posterior_file) => outdir)
    end

    # Print tables
    moment_tables(m, groupings = parameter_groupings(m), tables = [:prior, :posterior],
                  caption = false, outdir = outdir, verbose = :none, params = params)

    # Rename tables
    mv(prior_file,    joinpath(outdir, "prior_table.tex"),
       force = true)
    mv(posterior_file, joinpath(outdir, "posterior_table.tex"),
       force = true)

    filenames = joinpath(outdir, "{prior,posterior}_table.tex")
    println("Wrote $filenames")

    # Reset to original (forecast) vintage
    m <= Setting(:data_vintage, fcast_vint)
end
