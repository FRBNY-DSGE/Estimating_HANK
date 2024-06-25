using ModelConstructors, Dates, Distributions
using ClusterManagers, HDF5, Plots, ModelConstructors, Nullables, Random, HDF5, FileIO
GR.inline("pdf")

include("../../dsge_version/BBL_MH_SMC/src/BBL_MH_SMC.jl")
include("../../dsge_version/BBL_MH_SMC/src/helpers.jl")

filestring_addl = ["mhcloud"] #Add filestring identifier here

# What do you want to do? (additional estimation-specific settings are specified further below)
run_estimate           = true
run_forecast           = true
sampling_method        = :SMC
bridge_sampling_method = :SMC
bridge_cloud_ss        = "ss10"


#bridge cloud uses past estimation
first_run              = true
use_bridge_cloud       = true
use_intermed_start     = false
intermed_stage         = 0

# MH
run_csminwel           = false
run_hessian            = false
test_mh_run            = false
check_hessian_neg_diag = false

auto_add_procs         = true
intermediate_stage_increment = 10


iter = 1
total_iters = 1

# SMC Settings
if sampling_method == :SMC

    use_fixed_schedule = false

    #Number of particles:
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
    if run_csminwel
        estimate(m, data; sampling = false)
    end

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


    m <= Setting(:calculate_hessian, false)
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
    m <= Setting(:smc_fix_at_mode, false)

    m <= Setting(:calculate_hessian, false)
    m <= Setting(:hessian_path, rawpath(m, "estimate", "hessian_fixed_at_mode.h5"))

    steadystate!(m)



    if run_estimate
        if auto_add_procs
            #Add extra workers here:
            #n_workers =
            #myprocs =

            @everywhere include("../../dsge_version/BBL_MH_SMC/src/BBL_MH_SMC.jl")
            @everywhere using SMC, OrderedCollections
        end


        # Run SMC without cloud
        if use_intermed_start
            estimate(m, data; verbose = :low, save_intermediate = true, run_csminwel = false,
                     continue_intermediate = true, intermediate_stage_start = intermed_stage)
        elseif use_bridge_cloud && first_run
            f = jldopen(rawpath(m,"estimate","mh_cloud_smc.jld2"),"r")
            bridge_cloud = f["mh_cloud"]


            m <= Setting(:fcast_date, "2019-Q4", true, "fcast_date", "")
            estimate(m,data[:,1:end-1], verbose=:high, save_intermediate=true, run_csminwel=false, old_cloud=bridge_cloud, old_data = data, log_prob_old_data = 6560.0 , filestring_addl= filestring_addl )

        elseif use_bridge_cloud && !first_run
            function run_loop()
                year = 2019
                year_prev = 2019
                for i=1:80
                    quarter = mod(80-i,4) + 1
                    quarter_prev = mod(80-i+1, 4) + 1
                    #If you recreate the _____ estimation, replace that estimation with the filepath below
                    bridge_cloud = load("../../data/output_data/bayer_born_luetticke/ss1/estimate/raw/")["cloud"]
                    #load("*load file path to past run of estimation to use bridge cloud*")


                    fcast_date = quartertodate("$(year)-Q$(quarter)")
                    m <= Setting(:fcast_date, "$(year)Q$(quarter)",
                                 true, "fcastdate","")
                    estimate(m,data[:,1:end-1-i], verbose=:low, save_intermediate=true, run_csminwel=false, old_cloud=bridge_cloud, old_data = data[:, 1:end-i], log_prob_old_data = 6560.0, filestring_addl=filestring_addl) #was ["MH"]

                    year_prev = year

                    if mod(i,4)==0
                        year = year - 1
                    end
                    sleep(120)
                end
            end
            run_loop()
        else
            estimate(m,data;verbose = :high, save_intermediate = true, run_csminwel = false)
        end
    end


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
            #Add workers here

            @everywhere include("../../dsge_version/BBL_MH_SMC/src/BBL_MH_SMC.jl")
            @everywhere using SMC, OrderedCollections
        end
        fcast_date = quartertodate("2020-Q1")
        # m <= Setting(:fcast_date, "$(year(fcast_date))Q$(datetoquarter(fcast_date))", false, "fcastdate", "")
        m <= Setting(:forecast_block_size, 20)
        #        input_file_name = rawpath(m,"estimate","smc_cloud.h5")
        input_file_name = rawpath(m,"estimate")*
        "/smc_cloud_fullsmc_fcastdate=2020Q1_iter=1_mhsteps=3_nparts=1000_phi=0.95_vint=210504.jld2"
        forecast_string = "fixed_at_mode"
        para_dsge = load_draws(m,input_type; input_file_name = input_file_name)


        forecast_one(m,input_type, cond_type, output_vars;forecast_string=forecast_string, params = para_dsge, rerun_smoother = true, filestring_addl = filestring_addl)
        compute_meansbands(m,input_type, cond_type, output_vars;forecast_string = forecast_string, density_bands = density_bands, verbose=:high, df=df)
        plot_history_and_forecast(m,forecast_vars,class,input_type,:none; forecast_string = forecast_string, start_date = fcast_plot_start)
        for shock in shocks
            plot_impulse_response(m,shock,vars,class,input_type, :none; forecast_string = forecast_string)
        end

        if auto_add_procs

            rmprocs(myprocs)
        end
    end
end
