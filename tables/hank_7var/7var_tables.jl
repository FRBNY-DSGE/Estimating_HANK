using ModelConstructors, Dates, Distributions
using ClusterManagers, HDF5, Plots, ModelConstructors, Nullables, Random, HDF5, FileIO
GR.inline("pdf")

include("../../dsge_version/BBL_7var/src/BBL_7var.jl")
include("../../dsge_version/BBL_7var/src/helpers.jl")


filestring_addl = "7var"


run_forecast           = true
create_estim_output    = true
sampling_method        = :SMC
bridge_sampling_method = :SMC
bridge_cloud_ss        = "ss10"


#bridge cloud uses past estimation
create_bridge_cloud    = false
use_bridge_cloud       = true
use_intermed_start     = false
intermed_stage         = 252


auto_add_procs         = true
n_workers              = 96
intermediate_stage_increment = 10

# Run iteration
iter = 1
total_iters = 1

# SMC Settings
if sampling_method == :SMC

    use_fixed_schedule = false
    n_parts = 1000
    n_smc_blocks = 1
    n_mh_steps_smc = 3
    step_size_smc = 0.5
    mixture_proportion = 0.95
    adaptive_tempering_target_smc = 0.95
    resampling_threshold = 0.5
    tempered_update_prior_weight = 0.0

end

# MH Settings
if sampling_method == :MH
    mh_cc                  = 1.
    n_mh_simulations       = 1000
    n_mh_blocks            = 22
end


# Data vintages
data_vint  = "210504"
cond_vint  = "210504"
fcast_date = quartertodate("2020-Q1") #change this to 2020 Q1
date_fcast_end = iterate_quarters(fcast_date, 40)
mainsample_start_date = quartertodate("1954-Q4") #2015 Q1, 1954-Q4
fcast_plot_start = quartertodate("2015-Q1")


# Initialize model objects
m = BayerBornLuetticke()
usual_settings!(m, data_vint, cdvt = cond_vint, fcast_date = fcast_date)
m <= Setting(:date_conditional_end, fcast_date)
m <= Setting(:fcast_date, "$(year(fcast_date))Q$(datetoquarter(fcast_date))", true, "fcastdate", "")
m <= Setting(:iteration, iter, true, "iter", "")
m <= Setting(:date_mainsample_start, mainsample_start_date)
m <= Setting(:sampling_method, :SMC)
m <= Setting(:forecast_jstep, 1)
m <= Setting(:save_jacobian, false)

presample_start_date = quartertodate("1954-Q1")
m <= Setting(:date_presample_start, presample_start_date)
# Get data
df = load_data(m, try_disk = false, check_empty_columns = false, cond_type = :none)
df = df[date_presample_start(m) .<= df[!, :date], :]# .<= date_mainsample_end(m), :] # subset for the correct time frame

data = df_to_matrix(m, df)

ModelConstructors.toggle_regime!(m.parameters, 1)

# Estimate!
m <= Setting(:sampling_method, sampling_method)

if sampling_method == :MH


    update!(m, load_draws(m, :mode))
    m <= Setting(:reoptimize, false)

    for k in regswitch_para
        println("Parameter $(k)")
        println("Regime 1 = $(regime_val(m[k], 1))")
        println("Regime 2 = $(regime_val(m[k], 2))")
    end




    m <= Setting(:calculate_hessian, false)
    m <= Setting(:hessian_path, rawpath(m, "estimate", "hessian.h5"))

    # Non-default MH estimation settings

    m <= Setting(:mh_cc, mh_cc)
    m <= Setting(:n_mh_simulations, n_mh_simulations)
elseif sampling_method == :SMC
    # Add settings for SMC algorithm
    m <= Setting(:use_parallel_workers, auto_add_procs)
    m <= Setting(:use_fixed_schedule, use_fixed_schedule)
    m <= Setting(:n_particles, n_parts, true, "nparts", "")
    m <= Setting(:n_smc_blocks, n_smc_blocks)
    m <= Setting(:n_mh_steps_smc, n_mh_steps_smc, true, "mhsteps", "")
    m <= Setting(:step_size_smc, step_size_smc)
    m <= Setting(:mixture_proportion, mixture_proportion)
    m <= Setting(:adaptive_tempering_target_smc, adaptive_tempering_target_smc, true, "phi", "")
    m <= Setting(:resampling_threshold, resampling_threshold)
    m <= Setting(:klein_inversion_method,:direct)
    m <= Setting(:save_steadystate, false)
    m <= Setting(:tempered_update_prior_weight,tempered_update_prior_weight)
end

if create_estim_output

    if run_forecast
        input_type = :full
        cond_type = :none
        output_vars = [:forecastobs, :histobs]
        density_bands = [0.5,0.6,0.68,0.7,0.8,0.9]
        forecast_vars = collect(keys(m.observables))
        vars = forecast_vars
        class = :obs
        shocks = collect(keys(m.exogenous_shocks))
        if auto_add_procs
            #Add workers and assign additional memory here
            # myprocs =
            @everywhere include("../../dsge_version/BBL_7var/src/BBL_7var.jl")
        end
        function forecast_loop()



            for year=2019:2019

                #input_file_name = filepath of the estimation output file




                forecast_string = "estim=smc$(year)Q4_$(filestring_addl)"

                fcast_date = quartertodate("$(year)-Q4")

                m <= Setting(:fcast_date, "$(year)Q4",
                             true, "fcastdate","")
                m <= Setting(:date_forecast_start,
                             quartertodate("$(year)-Q4"))
                m <= Setting(:forecast_block_size, 20)

                para_dsge = load_draws(m,input_type; input_file_name = input_file_name)

                rd_draft_prior_posterior_tables(m;
                                                outdir = "output/",
                                                params = para_dsge)


            end

            if auto_add_procs
                rmprocs(myprocs)
            end
        end
        forecast_loop()

    end

end
