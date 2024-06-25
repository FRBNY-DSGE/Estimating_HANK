using Plots, FileIO, JLD2
using Plots.PlotMeasures

#Add input file paths to lines 6 and 8, then rename output pdf

# Change to load the filepath of the hank_7var rmse output file
bbl_smc   = load("seven_rmse_final.jld2")

# Change to load the filepath of the sw rmse output file
sw  = load("MH_10k.jld2")

gdp = zeros(7,2)
gdp[:,1] = sw["gdp"]
gdp[:,2] = bbl_smc["gdp"]

con = zeros(7,2)
con[:,1] = sw["con"]
con[:,2] = bbl_smc["con"]

inv = zeros(7,2)
inv[:,1] = sw["inv"]
inv[:,2] = bbl_smc["inv"]

def = zeros(7,2)
def[:,1] = sw["def"]
def[:,2] = bbl_smc["def"]

def = def .* 100
gdp = gdp .* 100
con = con .* 100
inv = inv .* 100

gdp_plot = plot(1:7, gdp,
                label = ["SW" "BBL (SMC)"],title="GDP Growth",
                lw = 3, xlabel = "Horizon", ylabel = "percent", margin = 15mm,
                seriescolor = [:red :black],
                linestyle = [:solid :solid])

con_plot = plot(1:7, con,
                label = ["SW" "BBL (SMC)"],title = "Consumption Growth",
                lw = 3, xlabel = "Horizon", ylabel = "percent", margin = 15mm,
                seriescolor = [:red :black],
                linestyle = [:solid :solid])

inv_plot = plot(1:7, inv,
                label = ["SW" "BBL (SMC)"], title = "Investment Growth",
                lw = 3, xlabel = "Horizon", ylabel = "percent", margin = 15mm,
                seriescolor = [:red :black],
                linestyle = [:solid :solid])

def_plot = plot(1:7, def,
                label = ["SW" "BBL (SMC)"],title = "GDP Deflator Inflation",
                lw = 3, xlabel = "Horizon", ylabel = "percent", margin = 15mm,
                seriescolor = [:red :black],
                linestyle = [:solid :solid])



savefig(gdp_plot, "gdp_rmse_bbl_sw.pdf")
savefig(con_plot, "con_rmse_bbl_sw.pdf")
savefig(inv_plot, "inv_rmse_bbl_sw.pdf")
savefig(def_plot, "def_rmse_bbl_sw.pdf")
