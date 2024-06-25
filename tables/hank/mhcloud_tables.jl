start = time()
using ModelConstructors, Dates, Distributions
using ClusterManagers, HDF5, Plots, ModelConstructors, Nullables, Random, HDF5, FileIO
GR.inline("pdf")

include("../../dsge_version/BBL_MH_SMC/src/BBL_MH_SMC.jl")
include("../../dsge_version/BBL_MH_SMC/src/helpers.jl")

filestring_addl = "mhcloud"

# What do you want to do? (additional estimation-specific settings are specified further below)
run_estimate           = false
run_forecast           = true
create_estim_output    = true

sampling_method        = :SMC
bridge_sampling_method = :SMC
bridge_cloud_ss        = "ss10"


#bridge cloud uses past estimation
create_bridge_cloud    = true
use_bridge_cloud       = true
use_intermed_start     = false
intermed_stage         = 252

# MH
csminwel_covid         = false
run_csminwel           = false
run_hessian            = false
test_mh_run            = false
check_hessian_neg_diag = false

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


if sampling_method == :MH
    mh_cc                  = 1.
    n_mh_simulations       = 1000
    n_mh_blocks            = 22
end



# Data vintages
data_vint  = "210504"
cond_vint  = "210504"
fcast_date = quartertodate("2020-Q1")
date_fcast_end = iterate_quarters(fcast_date, 40)
mainsample_start_date = quartertodate("1954-Q4")
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
df = df[date_presample_start(m) .<= df[!, :date], :]

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

    if run_hessian
        hessian, _ = hessian!(m, get_values(m.parameters), verbose = :low,
                              check_neg_diag = check_hessian_neg_diag)

        # Check we can take the inverse
        F = svd(hessian)
        big_eig_vals = findall(x -> x > 1e-6, F.S)
        hessian_rank = length(big_eig_vals)

        S_inv = zeros(size(hessian))
        for i = 1:hessian_rank
            S_inv[i, i] = 1/F.S[i]
        end

        hessian_inv = F.V * S_inv * F.U'
        test_d = DegenerateMvNormal(params, hessian_inv, hessian, diag(S_inv))

        if isa(rand(test_d, 1), Vector)
            h5open(rawpath(m, "estimate","hessian.h5"), "w") do file
                file["hessian"] = hessian
            end
        end
    end

    # Assumed that the Hessian has been saved
    m <= Setting(:calculate_hessian, false) # Don't re-calculate Hessian
    m <= Setting(:hessian_path, rawpath(m, "estimate", "hessian.h5"))

    # Non-default MH estimation settings
    if test_mh_run

       m <= Setting(:n_mh_simulations, 50)
        m <= Setting(:n_mh_burn, 0)
        m <= Setting(:n_mh_blocks, 1)
    end
    m <= Setting(:mh_cc, mh_cc)
    m <= Setting(:n_mh_simulations, n_mh_simulations)

    # Run!
    if run_estimate
        estimate(m, data, verbose = :low)
    end
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


    steadystate!(m)


    m <= Setting(:reoptimize, false)
    m <= Setting(:load_bbl_posterior_mode, true)
    m <= Setting(:smc_fix_at_mode, true)

    m <= Setting(:calculate_hessian, false)
    m <= Setting(:hessian_path, rawpath(m, "estimate", "hessian_fixed_at_mode.h5"))


    if run_estimate
    if auto_add_procs
        #Set parallel workers and memory here
        #myprocs =
        @everywhere include ("../dsge_version/BBL_MH_SMC/src/BBL_MH_SMC.jl")
    end


    # Run SMC without cloud
    if use_intermed_start
       estimate(m, data; verbose = :low, save_intermediate = true, run_csminwel = false,
             continue_intermediate = true, intermediate_stage_start = intermed_stage)
    elseif use_bridge_cloud

        bridge_cloud = load("../../data/output_data/bayer_born_luetticke/ss1/estimate/raw/smc_cloud_$(filestring_addl)_fcastdate=$(year_prev)Q$(quarter_prev)_iter=1_mhsteps=3_nparts=$(n_parts)_phi=0.95_vint=210504.jld2")["cloud"]


       estimate(m,data[:,1:end-1], verbose=:high, save_intermediate=true, run_csminwel=false, old_cloud=bridge_cloud, old_data = data, log_prob_old_data = 6560.0, filestring_addl=[filestring_addl])
    else
       estimate(m, data; verbose = :high, save_intermediate = true, run_csminwel = false)
    end




    if auto_add_procs
        rmprocs(myprocs)
    end
 end
end

if create_estim_output
    if run_forecast
        input_type = :full
        cond_type = :none
        output_vars = [:forecastobs, :irfobs, :histobs]
        density_bands = [0.5,0.6,0.68,0.7,0.8,0.9]
        forecast_vars = collect(keys(m.observables))
        vars = forecast_vars
        class = :obs
        shocks = collect(keys(m.exogenous_shocks))

        if auto_add_procs
            #Add parallel workers and more memory here
            #myprocs =
            @everywhere include("../../BBL_MH_SMC/src/BBL_MH_SMC.jl")
        end
        function forecast_loop()


            for year=2000:2019

                #input_file_name = filepath to corresponding estimation output file




                #csv_file_path = rawpath(m,"estimate")*"/bbl_draws_full_params.csv"
                #df_mh_draws = CSV.read(csv_file_path, DataFrame, header=false)


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



    else
        # Create model for old parameters
        m10 = Model1002("ss10")
        usual_settings!(m10, m10_vint, fcast_date = m10_fcast_date)

        m97_Q2 = Model1002("ss97")
        m97_Q2 <= Setting(:standard_shocks_mode_adjust, 1,
                          true, "modeadj", "")
        m97_Q2 <= Setting(:standard_shocks_spread_adjust, 2,
                          true, "spreadadj", "")
        usual_settings!(m97_Q2, "210506", fcast_date = Date(2021, 6, 30))
        m97_Q2 <= Setting(:fcast_date, "2021Q2", true, "fcastdate", "")

        m97_Q2 <= Setting(:sampling_method, :SMC)
        m97_Q2 <= Setting(:iteration, 1, true, "iter", "")
        m97_Q2 <= Setting(:tempered_update_prior_weight, 0.0, true, "priorwt", "")
        m97_Q2 <= Setting(:bridge, true, true, "bridge", "")
        m97_Q2 <= Setting(:n_mh_steps_smc, 3, true, "mhsteps", "")
        m97_Q2 <= Setting(:n_particles, 15_000, true, "nparts", "")
        m97_Q2 <= Setting(:adaptive_tempering_target_smc, 0.95, true, "phi", "")

        m <= Setting(:tempered_update_prior_weight, tempered_update_prior_weight, true, "priorwt", "")
        m97_Q2 <= Setting(:tempered_update_prior_weight, tempered_update_prior_weight, true, "priorwt", "")

        if set_spd_err_sig == 1
            set_regime_valuebounds!(m97_Q2[:ρ_corepce], 1, (0.0, 5.0))
            set_regime_valuebounds!(m97_Q2[:ρ_corepce], 2, (0.0, 5.0))
            set_regime_val!(m97_Q2[:ρ_corepce], 2, 0.0)
            set_regime_val!(m97_Q2[:ρ_corepce], 1, 0.232)
            set_regime_fixed!(m97_Q2[:ρ_corepce], 2, true)
            set_regime_fixed!(m97_Q2[:ρ_corepce], 1, false)
            set_regime_prior!(m97_Q2[:ρ_corepce], 1, m[:ρ_corepce].prior)
            set_regime_prior!(m97_Q2[:ρ_corepce], 2, m[:ρ_corepce].prior)

            set_regime_valuebounds!(m97_Q2[:ρ_meas_π], 1, (0.0, 5.0))
            set_regime_valuebounds!(m97_Q2[:ρ_meas_π], 2, (0.0, 5.0))
            set_regime_val!(m97_Q2[:ρ_meas_π], 1, 0.0)
            set_regime_val!(m97_Q2[:ρ_meas_π], 2, 0.232)
            set_regime_fixed!(m97_Q2[:ρ_meas_π], 1, true)
            set_regime_fixed!(m97_Q2[:ρ_meas_π], 2, false)
            set_regime_prior!(m97_Q2[:ρ_meas_π], 1, m[:ρ_meas_π].prior)
            set_regime_prior!(m97_Q2[:ρ_meas_π], 2, m[:ρ_meas_π].prior)
        end

        plot_posterior_comparison(m, m97_Q2, m10, ("2021-Q4", "2021-Q2", "Old"), third_regime_switching = false,
                                  use_reg2 = set_spd_err_sig == 1 ? [:ρ_meas_π] : [:whatever_dont_care],
                                  add_sig_meas_reg3 = false)
    end
end
