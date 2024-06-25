start = time()
using ModelConstructors, Dates, Distributions
using ClusterManagers, HDF5, Plots, ModelConstructors, Nullables, Random, HDF5, FileIO
GR.inline("pdf")

include("../../dsge_version/BBL_MH_SMC/src/BBL_MH_SMC.jl")
include("../../dsge_version/BBL_MH_SMC/src/helpers.jl")

forecast_date_strings = ["2000-Q1", "2001-Q1", "2002-Q1", "2003-Q1", "2004-Q1",
                         "2005-Q1", "2006-Q1", "2007-Q1", "2008-Q1", "2009-Q1",
                         "2010-Q1","2011-Q1", "2012-Q1", "2013-Q1", "2014-Q1",
                         "2015-Q1", "2016-Q1", "2017-Q1", "2018-Q1"]

df = load_data(m; check_empty_columns = false)

gdp_rmse = zeros(7,1)
con_rmse = zeros(7,1)
def_rmse = zeros(7,1)
inv_rmse = zeros(7,1)
for (j,var) in enumerate([:obs_gdp, :obs_consumption, :obs_gdpdeflator, :obs_investment])
    for (counter, i) in enumerate(forecast_date_strings)
        year = i[1:4]
        quarter = i[6:end]

        # fn =  * insert the means bands output forecast filepath, it will look similar to output_data/bayer_born_luetticke/ss1/forecast/work/mbforecastobs_cond=none_fcastdate=$(year)$(quarter)_fcid=estim=smc$(year)$(quarter)_fixed_iter=1_mhsteps=3_nparts=1000_para=full_phi=0.95_vint=210504.jld2 *

        mbs = read_mb(fn)
        fcast_date = quartertodate(i)

        start_ind = (findall(x -> x == fcast_date, df[:, "date"]))[1]

        square_errors = (df[start_ind:start_ind+6, var] .- mbs.means[1:7,var]).^2

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
save("smc_rmse_notfixed.jld2", "gdp", gdp_rmse, "con", con_rmse, "def", def_rmse, "inv", inv_rmse)
