using ModelConstructors, Dates, Distributions, Pkg
using ClusterManagers, HDF5, Plots, ModelConstructors, Nullables, Random, HDF5, FileIO, OrderedCollections, Distributed, FileIO, JLD2, StatsBase, DataFrames
GR.inline("pdf")

include("../../dsge_version/BBL_7var/src/BBL_7var.jl")
include("../../dsge_version/BBL_7var/src/helpers.jl")

filestring_addl= ["7var_test"] # Add filestring identifier here

# What do you want to do? (additional estimation-specific settings are specified further below)
run_estimate           = true
sampling_method        = :SMC
bridge_sampling_method = :SMC
bridge_cloud_ss        = "ss10"


#bridge cloud uses past estimation
use_bridge_cloud       = true
use_intermed_start     = false
intermed_stage         = 0

# MH
run_hessian            = false
test_mh_run            = false
check_hessian_neg_diag = false

auto_add_procs         = false
n_workers              = 450
get_mdds               = false
intermediate_stage_increment = 10


iter = 1
total_iters = 1

# SMC Settings
if sampling_method == :SMC

    use_fixed_schedule = false

    # Number of particles used in SMC
    n_parts = 100

    n_smc_blocks = 3
    n_mh_steps_smc = 3
    step_size_smc = 0.5
    mixture_proportion = 0.95
    adaptive_tempering_target_smc = 0.95
    resampling_threshold = 0.5
    tempered_update_prior_weight = 0



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
usual_model_settings!(m, data_vint, cdvt = cond_vint, fcast_date = fcast_date)
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

    m <= Setting(:reoptimize, false)      # don't re-optimize with csminwel
    m <= Setting(:load_bbl_posterior_mode, true)
    m <= Setting(:smc_fix_at_mode, false)

    m <= Setting(:calculate_hessian, false) # Don't re-calculate Hessian
    m <= Setting(:hessian_path, rawpath(m, "estimate", "hessian_fixed_at_mode.h5"))

    steadystate!(m)

    if run_estimate
        if auto_add_procs
            @everywhere include("../../dsge_version/BBL_7var/src/BBL_7var.jl")

            # You can set parallel workers and commands to use more memory here:
            #n_workers =
            #myprocs =


        end


    # Run SMC without cloud
    if use_intermed_start
       estimate(m, data; verbose = :low, save_intermediate = true, run_csminwel = false,
             continue_intermediate = true, intermediate_stage_start = intermed_stage)
    elseif use_bridge_cloud
        function run_loop()
            year = 2019
            year_prev = 2020
            for i=1:80
                quarter = mod(80-i,4) + 1
                quarter_prev = mod(80-i+1, 4) + 1
                if i == 1
                    #If you rerun the 2020Q1 estimation, replace this filepath with the your own estimation filepath
                    bridge_cloud = load("../../data/output_data/bayer_born_luetticke/ss1/estimate/raw/smc_cloud_7var_fcastdate=2020Q1_iter=1_mhsteps=3_nparts=10000_phi=0.95_vint=210504.jld2")["cloud"]
                else
                    #If you rerun the estimation for each year/quarter combination that is not 2020 Q1, include filepath here
                    bridge_cloud = load("../../data/output_data/bayer_born_luetticke/ss1/estimate/raw/smc_cloud_7var_fcastdate=$(year_prev)Q$(quarter_prev)_iter=1_mhsteps=3_nparts=10000_phi=0.95_vint=210504.jld2")["cloud"]
                end
                m <= Setting(:fcast_date, "$(year)Q$(quarter)",
                             true, "fcastdate","")
                # iterate each one lower
                estimate(m,data[:,1:end-1-i], verbose=:low, save_intermediate=true, run_csminwel=false, old_cloud=bridge_cloud, old_data = data[:, 1:end-i], filestring_addl = filestring_addl)

                year_prev = year

                if mod(i,4)==0
                    year = year - 1
                end
                sleep(120)
            end
        end
        run_loop()
    else
       estimate(m, data; verbose = :high, save_intermediate = true, run_csminwel = false,
                     filestring_addl = filestring_addl)
    end

    if auto_add_procs
        rmprocs(myprocs)
    end
 end
end

if get_mdds

    m <= Setting(:bridge, true, true, "bridge", "")

end
