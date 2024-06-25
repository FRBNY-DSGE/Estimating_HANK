using ModelConstructors, Random, Plots, Distributed
using Plots.PlotMeasures, Dates, Distributions, ClusterManagers
using HDF5, Nullables, CSV, FileIO


include("../../dsge_version/SW/src/SW.jl")
include("../../dsge_version/SW/src/helpers.jl")


# What do you want to do?
estimate_dsgevar  = false   # Estimate a DSGEVAR using SMC
use_estim_output  = true   # Use posterior from SMC estimation for following code. Otherwise, use default parameters.
run_forecasts     = false   # run forecasts
calc_rmses        = true
get_VAR_system    = false   # Compute VAR coefficients and innovations-covariance matrix
do_modal_irf      = false   # Compute IRFs using modal parameters
compare_modal_irf = false   # Plot the modal DSGEVAR λ = ∞ rotation IRF and the actual DSGE IRF for observables
do_full_band_irf  = false   # Compute IRFs using parameters drawn from a distribution
create_meansbands = false  # Save full_band_irfs to MeansBands
do_parallel       = true  # Use parallel workers
auto_add_procs    = true
n_workers         = 96
dsgevar_λ         = 1    # What λ do you want to use?



# Set up DSGE and load data
m10_fcast_date = Date(2019,12,31) #double check which date to set here
# Data vintages
data_vint  = "210504"
cond_vint  = "210504"
fcast_date = quartertodate("2020-Q1")
date_fcast_end = iterate_quarters(fcast_date, 40)
mainsample_start_date = quartertodate("1954-Q1")

use_fixed_schedule            = false
n_particles                   = 1000 #set this variable to the number of particles you would like to use
n_smc_blocks                  = 3
n_mh_steps_smc                = 3
step_size_smc                 = 0.001 # try to guarantee high initial acceptance and adapt to a lower accept rate
mixture_proportion            = 0.9
adaptive_tempering_target_smc = 0.95
resampling_threshold          = 0.5


# Initialize model objects
m = SmetsWouters()
usual_settings!(m, data_vint, cdvt = cond_vint, fcast_date = fcast_date)
m <= Setting(:date_conditional_end, fcast_date)
m <= Setting(:fcast_date, "$(year(fcast_date))Q$(datetoquarter(fcast_date))", true, "fcastdate", "")
m <= Setting(:date_mainsample_start, mainsample_start_date)


modal_paras = map(x -> x.value, m.parameters)

forecast_string = "smets_dsgevar" # Change this to an empty string if you don't want an identifier for saved output

m <= Setting(:sampling_method, :SMC)
m <= Setting(:use_parallel_workers, auto_add_procs)
m <= Setting(:use_fixed_schedule, use_fixed_schedule)
m <= Setting(:n_particles, n_particles, true, "demeaned_nparts", "")
m <= Setting(:n_smc_blocks, n_smc_blocks)
m <= Setting(:n_mh_steps_smc, n_mh_steps_smc, true, "mhsteps", "")
m <= Setting(:step_size_smc, step_size_smc)
m <= Setting(:mixture_proportion, mixture_proportion)
m <= Setting(:adaptive_tempering_target_smc, adaptive_tempering_target_smc, true, "alpha", "")
m <= Setting(:resampling_threshold, resampling_threshold)
m <= Setting(:impulse_response_horizons, 20)


# load data based on DSGEVAR specification
df = load_data(m; check_empty_columns = false)
data = df_to_matrix(m, df)

#data = Matrix(CSV.read("bbl_data.csv", DataFrame))




if calc_rmses

    # forecast_date_strings = ["2000-Q1", "2001-Q1", "2002-Q1", "2003-Q1", "2004-Q1",
    #                          "2005-Q1", "2006-Q1", "2007-Q1", "2008-Q1", "2009-Q1",
    #                          "2010-Q1","2011-Q1", "2012-Q1", "2013-Q1", "2014-Q1",
    #                          "2015-Q1", "2016-Q1", "2017-Q1", "2018-Q1"]


    forecast_date_strings = ["2010-Q1","2011-Q1", "2012-Q1", "2013-Q1", "2014-Q1",
                             "2015-Q1", "2016-Q1", "2017-Q1", "2018-Q1"]


    #df = load(* insert filepath to the output of the corresponding forecast here *)["df"]

    df= select(df, :date, :obs_gdp, :obs_hours, :obs_wages, :obs_gdpdeflator, :obs_nominalrate,
        :obs_consumption, :obs_investment)

    df = df[44:end,:]


    gdp_rmse = zeros(7,1)
    con_rmse = zeros(7,1)
    def_rmse = zeros(7,1)
    inv_rmse = zeros(7,1)
    for (j,var) in enumerate([:obs_gdp, :obs_consumption, :obs_gdpdeflator, :obs_investment])
        for (counter, i) in enumerate(forecast_date_strings)
            year = i[1:4]
            quarter = i[6:end]

            fn = workpath(m,"forecast")*
            "/mbforecastobs_alpha=0.95_cond=none_demeaned_nparts=1000_fcastdate=$(year)$(quarter)_fcid=estim$(year)$(quarter)_mhsteps=3_para=full_vint=210504.jld2"

            mbs = read_mb(fn)
            fcast_date = quartertodate(i)

            start_ind = (findall(x -> x == fcast_date, df[:, "date"]))[1]

            square_errors = (df[start_ind:start_ind+6, var] .- mbs.means[1:7,var]).^2
            println(mbs.means[1:7,var])
            if var == :obs_gdp
                gdp_rmse[:] .+= square_errors
            elseif var == :obs_consumption
                con_rmse[:] .+= square_errors
            elseif var == :obs_gdpdeflator
                def_rmse[:] .+= square_errors
            else
                inv_rmse[:] .+= square_errors
            end

        end

    end
    gdp_rmse = sqrt.(gdp_rmse ./ 8)
    con_rmse = sqrt.(con_rmse ./ 8)
    inv_rmse = sqrt.(inv_rmse ./ 8)
    def_rmse = sqrt.(def_rmse ./ 8)
    save("sw_rmse_back.jld2", "gdp", gdp_rmse, "con", con_rmse, "def", def_rmse, "inv", inv_rmse)
    @assert false
end
