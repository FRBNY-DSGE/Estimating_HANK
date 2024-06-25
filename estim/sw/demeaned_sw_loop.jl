using ModelConstructors, Random, Plots, Distributed
using Plots.PlotMeasures, Dates, Distributions, ClusterManagers
using HDF5, Nullables, CSV, FileIO
using DataFrames

include("../../dsge_version/SW/src/SW.jl")
include("../../dsge_version/SW/src/helpers.jl")

filestring_addl = ["sw"] #Add filestring identifier here
# What do you want to do?
estimate_firststep = true
estimate_loop      = true
do_parallel        = true  # Use parallel workers
auto_add_procs     = false
n_workers          = 96
dsgevar_λ          = 1    # What λ do you want to use?



# Set up DSGE and load data
m10_fcast_date = Date(2019,12,31)
# Data vintages
data_vint  = "210504"
cond_vint  = "210504"
fcast_date = quartertodate("2020-Q1")
date_fcast_end = iterate_quarters(fcast_date, 40)
mainsample_start_date = quartertodate("1954-Q4")

use_fixed_schedule            = false
n_particles                   = 1000
n_smc_blocks                  = 3
n_mh_steps_smc                = 3
step_size_smc                 = 0.001
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

df = load("../../data/output_data/dataf.jld2")["df"]

select!(df, :obs_gdp, :obs_hours, :obs_wages, :obs_gdpdeflator, :obs_nominalrate,
        :obs_consumption, :obs_investment)

data = Matrix(df)
data = data[4:end,:]'
data = Matrix(data)

data = Float64.(collect(Missings.replace(data,NaN)))

if estimate_firststep


    #Add workers here:
    # my_procs =
    @everywhere include("../../dsge_version/SW/src/SW.jl")

    estimate(m, data;
                  run_csminwel = false, verbose = :low,
                  save_intermediate = true , filestring_addl=filestring_addl)

end

if estimate_loop
    if do_parallel

        #Add workers here:
        # my_procs =
        @everywhere include("../../dsge_version/SW/src/SW.jl")
    end

    tempered_update_prior_weight = 0.0
    m <= Setting(:tempered_update_prior_weight, 0.0)

    function run_loop()
        year = 2019 #2019
        year_prev = 2020 #2020
        for i=1:80 #1:80
            quarter = mod(80-i,4) + 1 #80
            quarter_prev = mod(80-i+1, 4) + 1 #80
            #bridge_cloud = load(LOAD FILEPATH TO "smc_cloud_alpha=0.95_demeaned_nparts=$(n_particles)_fcastdate=$(year_prev)Q$(quarter_prev)_mhsteps=3_vint=210504.jld2")["cloud"]
            bridge_cloud = load("../../data/output_data/smets_wouters/ss0/estimate/raw/smc_cloud_alpha=0.95_demeaned_nparts=$(n_particles)_fcastdate=$(year_prev)Q$(quarter_prev)_mhsteps=3_$(filestring_addl[begin])_vint=210504.jld2")["cloud"]


            fcast_date = quartertodate("$(year)-Q$(quarter)")
            m <= Setting(:fcast_date, "$(year)Q$(quarter)",
                         true, "fcastdate", "")

            estimate(m, data[:, 1:end-1-i];
                          run_csminwel = false, verbose = :low,
                          save_intermediate = true,
                          old_cloud = bridge_cloud,
                          old_data = data[:,1:end-i] , filestring_addl=filestring_addl )


            year_prev = year

            if mod(i,4)==0
                year = year - 1
            end
            sleep(60)

        end
    end
    run_loop()
    if do_parallel
        rmprocs(my_procs)
    end
end
