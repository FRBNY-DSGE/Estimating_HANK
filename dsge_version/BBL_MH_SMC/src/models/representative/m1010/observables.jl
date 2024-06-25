function init_observable_mappings!(m::Model1010)

    observables = OrderedDict{Symbol,Observable}()
    population_mnemonic = get(get_setting(m, :population_mnemonic))

    ############################################################################
    ## 1. GDP
    ############################################################################
    gdp_fwd_transform =  function (levels)
        # FROM: Level of nominal GDP (FRED :GDP series)
        # TO:   Quarter-to-quarter percent change of real, per-capita GDP, adjusted for population smoothing

        levels[!,:temp] = percapita(m, :GDP, levels)
        gdp = 1000 * nominal_to_real(:temp, levels)
        oneqtrpctchange(gdp)
    end

    gdp_rev_transform = loggrowthtopct_annualized_percapita

    observables[:obs_gdp] = Observable(:obs_gdp, [:GDP__FRED, population_mnemonic, :GDPDEF__FRED],
                                       gdp_fwd_transform, gdp_rev_transform,
                                       "Real GDP Growth", "Real GDP Growth Per Capita")

    ############################################################################
    ## 2. Hours per-capita
    ############################################################################

    hrs_fwd_transform =  function (levels)
        # FROM: Average weekly hours (AWHNONAG) & civilian employment (CE16OV)
        # TO:   log (3 * per-capita weekly hours / 100)
        # Note: Not sure why the 3 is there.

        levels[!,:temp] = levels[!,:AWHNONAG] .* levels[!,:CE16OV]
        weeklyhours = percapita(m, :temp, levels)
        100*log.(3 * weeklyhours / 100)
    end

    hrs_rev_transform = logleveltopct_annualized_percapita

    observables[:obs_hours] = Observable(:obs_hours, [:AWHNONAG__FRED, :CE16OV__FRED],
                                         hrs_fwd_transform, hrs_rev_transform,
                                         "Hours Per Capita", "Log Hours Per Capita")


    ############################################################################
    ## 3. Wages
    ############################################################################

    wages_fwd_transform = function (levels)
        # FROM: Nominal compensation per hour (:COMPNFB from FRED)
        # TO: quarter to quarter percent change of real compensation (using GDP deflator)

        oneqtrpctchange(nominal_to_real(:COMPNFB, levels))
    end

    wages_rev_transform = loggrowthtopct_annualized

    observables[:obs_wages] = Observable(:obs_wages, [:COMPNFB__FRED, :GDPDEF__FRED],
                                         wages_fwd_transform, wages_rev_transform,
                                         "Percent Change in Wages",
                                         "Q-to-Q Percent Change of Real Compensation (using GDP deflator)")

    ############################################################################
    ## 4. GDP Deflator
    ############################################################################

    gdpdeflator_fwd_transform =  function (levels)
        # FROM: GDP deflator (index)
        # TO:   Approximate quarter-to-quarter percent change of gdp deflator,
        #       i.e.  quarterly gdp deflator inflation

        oneqtrpctchange(levels[!,:GDPDEF])
    end


    gdpdeflator_rev_transform = loggrowthtopct_annualized

    observables[:obs_gdpdeflator] = Observable(:obs_gdpdeflator, [:GDPDEF__FRED],
                                               gdpdeflator_fwd_transform, gdpdeflator_rev_transform,
                                               "GDP Deflator",
                                               "Q-to-Q Percent Change of GDP Deflator")

    ############################################################################
    ## 5. Core PCE Inflation
    ############################################################################

    pce_fwd_transform = function (levels)
        # FROM: Core PCE index
        # INTO: Approximate quarter-to-quarter percent change of Core PCE,
        # i.e. quarterly core pce inflation

        oneqtrpctchange(levels[!,:PCEPILFE])
    end

    pce_rev_transform = loggrowthtopct_annualized

    observables[:obs_corepce] = Observable(:obs_corepce, [:PCEPILFE__FRED],
                                           pce_fwd_transform, pce_rev_transform,
                                           "Core PCE Inflation",
                                           "Core PCE Inflation")


    ############################################################################
    ## 6. Nominal short-term interest rate (3 months)
    ############################################################################

    nominalrate_fwd_transform = function (levels)
        # FROM: Nominal effective federal funds rate (aggregate daily data at a
        #       quarterly frequency at an annual rate)
        # TO:   Nominal effective fed funds rate, at a quarterly rate

        annualtoquarter(levels[!,:DFF])
    end

    nominalrate_rev_transform = quartertoannual

    observables[:obs_nominalrate] = Observable(:obs_nominalrate, [:DFF__FRED],
                                               nominalrate_fwd_transform, nominalrate_rev_transform,
                                               "Nominal FFR",
                                               "Nominal Effective Fed Funds Rate")

    ############################################################################
    ## 7. Consumption
    ############################################################################

    consumption_fwd_transform = function (levels)
        # FROM: Nominal consumption
        # TO:   Real consumption, approximate quarter-to-quarter percent change,
        #       per capita, adjusted for population filtering

        levels[!,:temp] = percapita(m, :PCE, levels)
        cons = 1000 * nominal_to_real(:temp, levels)
        oneqtrpctchange(cons)
    end

    consumption_rev_transform = loggrowthtopct_annualized_percapita

    observables[:obs_consumption] = Observable(:obs_consumption, [:PCE__FRED, population_mnemonic],
                                               consumption_fwd_transform, consumption_rev_transform,
                                               "Consumption growth per capita",
                                               "Consumption growth adjusted for population filtering")


    ############################################################################
    ## 8. Investment growth per capita
    ############################################################################

    investment_fwd_transform = function (levels)
        # FROM: Nominal investment
        # INTO: Real investment, approximate quarter-to-quarter percent change,
        #       per capita, adjusted for population filtering

        levels[!,:temp] = percapita(m, :FPI, levels)
        inv = 10000 * nominal_to_real(:temp, levels)
        oneqtrpctchange(inv)
    end

    investment_rev_transform  = loggrowthtopct_annualized_percapita

    observables[:obs_investment] = Observable(:obs_investment, [:FPI__FRED, population_mnemonic],
                                              investment_fwd_transform, investment_rev_transform,
                                              "Real Investment per capita",
                                              "Real investment per capita, adjusted for population filtering")


    ############################################################################
    ## 9. Spread: BAA-20yr TBill
    ############################################################################

    spread_fwd_transform = function (levels)
        # FROM: Baa corporate bond yield (percent annualized), and 20-year
        #       treasury note yield (percent annualized)
        # TO:   Baa yield - 20T yield spread at a quarterly rate
        # Note: Moody's corporate bond yields on the H15 are based on corporate
        #       bonds with remaining maturities of at least 20 years.
        # Another note: There were no 20-year treasuries between
        #       1987-1993. Therefore, we use the LTGOVTBD series until 2000, then splice in the GS20 series.

        start_splice = Dates.Date("1987-03-31")
        end_splice   = Dates.Date("1993-09-30")

        stopgap_dates   = start_splice .<= levels[!,:date] .<= end_splice
        avg_stopgap     = (levels[stopgap_dates, :GS10] + levels[stopgap_dates, :GS30]) / 2.
        treasury_yields = vcat(levels[levels[!,:date] .< start_splice, :GS20],
                               avg_stopgap,
                               levels[levels[!,:date] .> end_splice, :GS20])

        annualtoquarter(levels[!,:BAA] - treasury_yields)
    end

    spread_rev_transform = quartertoannual

    observables[:obs_BBBspread] = Observable(:obs_BBBspread, [:BAA__FRED, :GS20__FRED, :GS10__FRED,
                                                              :GS30__FRED],
                                          spread_fwd_transform, spread_rev_transform,
                                          "BAA - 20yr Treasury Spread",
                                          "BAA - 20yr Treasury Spread")


    ############################################################################
    # 10. Long term inflation expectations
    ############################################################################

    longinflation_fwd_transform = function (levels)
        # FROM: SPF: 10-Year average yr/yr CPI inflation expectations (annual percent)
        # TO:   FROM, less 0.5
        # Note: We subtract 0.5 because 0.5% inflation corresponds to
        #       the assumed long-term rate of 2 percent inflation, but the
        #       data are measuring expectations of actual inflation.

        annualtoquarter(levels[!,:ASACX10]  .- 0.5)
    end

    longinflation_rev_transform = loggrowthtopct_annualized

    observables[:obs_longinflation] = Observable(:obs_longinflation, [:ASACX10__DLX],
                                                 longinflation_fwd_transform, longinflation_rev_transform,
                                                 "Long term inflation expectations",
                                                 "10-year average yr/yr CPI inflation expectations")


    ############################################################################
    # 11. Long rate (10-year, zero-coupon)
    ############################################################################
    longrate_fwd_transform = function (levels)
        # FROM: pre-computed long rate at an annual rate
        # TO:   10T yield at a quarterly rate

        annualtoquarter(levels[!,:FYCCZA])
    end

    longrate_rev_transform = quartertoannual

    observables[:obs_longrate] = Observable(:obs_longrate, [:FYCCZA__DLX],
                                            longrate_fwd_transform, longrate_rev_transform,
                                            "Long term interest rate expectations",
                                            "10T yield")


    ############################################################################
    # 12. Fernald TFP
    ############################################################################
    tfp_fwd_transform =  function (levels)
        # FROM: Fernald's unadjusted TFP series
        # TO:   De-meaned unadjusted TFP series, adjusted by Fernald's
        #       estimated alpha
        # Note: We only want to calculate the mean of unadjusted TFP over the
        #       periods between date_presample_start(m) - 1 and
        #       date_mainsample_end(m), though we may end up transforming
        #       additional periods of data.

        start_date = Dates.lastdayofquarter(date_presample_start(m) - Dates.Month(3))
        end_date   = date_mainsample_end(m)
        date_range = start_date .<= levels[!, :date] .<= end_date
        tfp_unadj_inrange = levels[date_range, :TFPKQ]

        tfp_unadj      = levels[!,:TFPKQ]
        tfp_unadj_inrange_nonmissing = tfp_unadj_inrange[.!ismissing.(tfp_unadj_inrange)]
        tfp_unadj_inrange_nonmissing = tfp_unadj_inrange_nonmissing[.!isnan.(tfp_unadj_inrange_nonmissing)]
        tfp_unadj_mean = isempty(tfp_unadj_inrange_nonmissing) ? missing : mean(tfp_unadj_inrange_nonmissing)
        (tfp_unadj .- tfp_unadj_mean) ./ (4*(1 .- levels[!,:TFPJQ]))
    end

    tfp_rev_transform = quartertoannual

    observables[:obs_tfp] = Observable(:obs_tfp, [:TFPKQ__DLX, :TFPJQ__DLX],
                                       tfp_fwd_transform, tfp_rev_transform,
                                       "Total Factor Productivity",
                                       "Fernald's TFP, adjusted by Fernald's estimated alpha")

    ############################################################################
    # 13. GDI
    ############################################################################
    gdi_fwd_transform = function (levels)
        # FROM: level of real per-capita income (FRED mnemonic: GDI)
        # TO:   approximate quarter-to-quarter percent change of real, per-capita income, adjusted
        #       for population smoothing

        levels[!,:temp] = percapita(m, :GDI, levels)
        gdi = 1000 * nominal_to_real(:temp, levels)
        oneqtrpctchange(gdi)
    end

    gdi_rev_transform = loggrowthtopct_annualized_percapita

    observables[:obs_gdi] = Observable(:obs_gdi, [:GDI__FRED],
                                       gdi_fwd_transform, gdi_rev_transform,
                                       "Real GDI Growth",
                                       "Real GDI Growth Per Capita")

    ############################################################################
    ## 14. Spread: AAA-20yr TBill
    ############################################################################

    spread_fwd_transform = function (levels)
        # FROM: AAA corporate bond yield (percent annualized), and 20-year
        #       treasury note yield (percent annualized)
        # TO:   AAA yield - 20T yield spread at a quarterly rate
        # Note: Moody's corporate bond yields on the H15 are based on corporate
        #       bonds with remaining maturities of at least 20 years.
        # Another note: There were no 20-year treasuries between
        #       1987-1993. Therefore, we use the LTGOVTBD series until 2000, then splice in the GS20 series.


        start_splice = Date("1987-03-31")
        end_splice   = Date("1993-09-30")

        stopgap_dates   = start_splice .<= levels[!,:date] .<= end_splice
        avg_stopgap     = (levels[stopgap_dates, :GS10] + levels[stopgap_dates, :GS30]) / 2.
        treasury_yields = vcat(levels[levels[!,:date] .< start_splice, :GS20],
                               avg_stopgap,
                               levels[levels[!,:date] .> end_splice, :GS20])

        annualtoquarter(levels[!,:AAA] - treasury_yields)
    end

    spread_rev_transform = quartertoannual

    observables[:obs_AAAspread] = Observable(:obs_AAAspread, [:AAA__FRED, :GS20__FRED, :LTGOVTBD__FRED],
                                             spread_fwd_transform, spread_rev_transform,
                                             "AAA - 20yr Treasury Spread",
                                             "AAA - 20yr Treasury Spread")


    ############################################################################
    # Columns 15 - 15 + n_anticipated_shocks
    ############################################################################

    for i = 1:n_anticipated_shocks(m)
        # FROM: OIS expectations of $i-period-ahead interest rates at a quarterly rate
        # TO:   Same

        ant_fwd_transform = function (levels)
            levels[!, Symbol("ant$i")]
        end

        ant_rev_transform = quartertoannual

        observables[Symbol("obs_nominalrate$i")] = Observable(Symbol("obs_ant$i"), [Symbol("ant$(i)__OIS")],
                                                      ant_fwd_transform, ant_rev_transform,
                                                      "Anticipated Shock $i",
                                                      "$i-period ahead anticipated monetary policy shock")
    end

   m.observable_mappings = observables
end
