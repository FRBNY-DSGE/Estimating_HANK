using Plots, FileIO, JLD2
using Plots.PlotMeasures

#Replace load file paths and output pdf names

sw   = load(#=input filepath to sw_rmse_back.jld2=#)

bbl_mh = load(#=filepath to mh file for 1K particles run of mh=#)

bbl_smc = load(#=filepath to rmse file for 1K particles run of smc=#)

bbl_seven = load(#=filepath to rmse file for 1K particles run of sevenvar=#)

gdp = zeros(7,4)
gdp[:,1] = sw["gdp"]
gdp[:,2] = bbl_mh["gdp"]
gdp[:,3] = bbl_smc["gdp"]
gdp[:,4] = bbl_seven["gdp"]

con = zeros(7,4)
con[:,1] = sw["con"]
con[:,2] = bbl_mh["con"]
con[:,3] = bbl_smc["con"]
con[:,4] = bbl_seven["con"]

inv = zeros(7,4)
inv[:,1] = sw["inv"]
inv[:,2] = bbl_mh["inv"]
inv[:,3] = bbl_smc["inv"]
inv[:,4] = bbl_seven["inv"]

def = zeros(7,4)
def[:,1] = sw["def"]
def[:,2] = bbl_mh["def"]
def[:,3] = bbl_smc["def"]
def[:,4] = bbl_seven["def"]

def = def .* 100
gdp = gdp .* 100
con = con .* 100
inv = inv .* 100

gdp_plot = plot(1:7, gdp,
                label = ["SW" "BBL (MH)" "BBL (SMC)" "BBL 7Var"],title="GDP Growth",
                lw = 3, xlabel = "Horizon", ylabel = "percent", margin = 15mm,
                seriescolor = [:red :black :black :black],
                linestyle = [:solid :dash :solid :dashdot])

con_plot = plot(1:7, con,
                label = ["SW" "BBL (MH)" "BBL (SMC)" "BBL 7Var"],title="Consumption Growth",
                lw = 3, xlabel = "Horizon", ylabel = "percent", margin = 15mm,
                seriescolor = [:red :black :black :black],
                linestyle = [:solid :dash :solid :dashdot])


inv_plot = plot(1:7, inv,
                label = ["SW" "BBL (MH)" "BBL (SMC)" "BBL 7Var"],title="Investment Growth",
                lw = 3, xlabel = "Horizon", ylabel = "percent", margin = 15mm,
                seriescolor = [:red :black :black :black],
                linestyle = [:solid :dash :solid :dashdot])


def_plot = plot(1:7, def,
                label = ["SW" "BBL (MH)" "BBL (SMC)" "BBL 7Var"],title = "GDP Deflator Inflation",
                lw = 3, xlabel = "Horizon", ylabel = "percent", margin = 15mm,
                seriescolor = [:red :black :black :black],
                linestyle = [:solid :dash :solid :dashdot])


savefig(gdp_plot, "gdp_rmse_all__aws_fig2.pdf")
savefig(con_plot, "con_rmse_all__aws_fig2.pdf")
savefig(inv_plot, "inv_rmse_all__aws_fig2.pdf")
savefig(def_plot, "def_rmse_all__aws_fig2.pdf")
