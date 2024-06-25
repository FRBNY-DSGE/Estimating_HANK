function init_observable_mappings!(m::BayerBornLuetticke)

    observables = OrderedDict{Symbol,Observable}()
    population_mnemonic = get(get_setting(m, :population_mnemonic))

    iden_fwd_transform = function (levels, name)
        return levels[!, name]
    end

    ############################################################################
    ## 1. GDP growth per capita
    ############################################################################
    gdp_fwd_transform =  function (levels)
        # FROM: Level of nominal GDP (FRED :GDP series)
        # TO:   Quarter-to-quarter percent change of real, per-capita GDP, adjusted for population smoothing

        levels[!,:temp] = percapita(m, :GDP, levels)
        gdp = 1000 * nominal_to_real(:temp, levels) # multiply 1000 b/c population is in thousands
        oneqtrpctchange(gdp)
    end

    gdp_rev_transform = loggrowthtopct_annualized_percapita

    observables[:obs_gdp] = Observable(:obs_gdp, [:GDP__FRED, population_mnemonic, :GDPDEF__FRED],
                                       gdp_fwd_transform, identity,#gdp_rev_transform,
                                       "Real GDP Growth", "Real GDP Growth Per Capita")

    ############################################################################
    ## 2. Consumption growth per-capita
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
                                               consumption_fwd_transform,identity,# consumption_rev_transform,
                                               "Consumption growth per capita",
                                               "Consumption growth adjusted for population filtering")

    ############################################################################
    ## 3. Investment growth per capita
    ############################################################################

    investment_fwd_transform = function (levels)
        # FROM: Nominal investment
        # INTO: Real investment, approximate quarter-to-quarter percent change,
        #       per capita, adjusted for population filtering

        levels[!,:temp] = percapita(m, :FPI, levels)
        inv = 1000 * nominal_to_real(:temp, levels) # multiply 1000 b/c population is in thousands
        oneqtrpctchange(inv)
    end

    investment_rev_transform  = loggrowthtopct_annualized_percapita

    observables[:obs_investment] = Observable(:obs_investment, [:FPI__FRED, population_mnemonic],
                                              investment_fwd_transform, identity,#investment_rev_transform,
                                              "Real Investment per capita",
                                              "Real investment per capita, adjusted for population filtering")

    ############################################################################
    ## 4. Wage growth
    ############################################################################

    wages_fwd_transform = function (levels)
        # FROM: Nominal compensation per hour (:COMPNFB from FRED)
        # TO: quarter to quarter percent change of real compensation (using GDP deflator)

        oneqtrpctchange(nominal_to_real(:COMPNFB, levels))
    end

    wages_rev_transform = loggrowthtopct_annualized

    observables[:obs_wages] = Observable(:obs_wages, [:COMPNFB__FRED, :GDPDEF__FRED],
                                         wages_fwd_transform, identity,#wages_rev_transform,
                                         "Percent Change in Wages",
                                         "Q-to-Q Percent Change of Real Compensation (using GDP deflator)")

    ############################################################################
    ## 5. Hours per capita
    ############################################################################

    hrs_fwd_transform =  function (levels)
        # FROM: Average weekly hours (AWHNONAG) & civilian employment (CE16OV)
        # TO:   log (13 * per-capita weekly hours / 100)

        levels[!,:temp] = levels[!,:AWHNONAG] .* levels[!,:CE16OV]
        weeklyhours = percapita(m, :temp, levels)
        100 * log.(13 * weeklyhours) # multiply weekly by 13 to go from weekly to quarterly
    end

    hrs_rev_transform = logleveltopct_annualized_percapita

    observables[:obs_hours] = Observable(:obs_hours, [:AWHNONAG__FRED, :CE16OV__FRED],
                                         hrs_fwd_transform, identity,#hrs_rev_transform,
                                         "Hours Per Capita", "Log Hours Per Capita")

    ############################################################################
    ## 6. GDP Deflator (Inflation)
    ############################################################################

    gdpdeflator_fwd_transform =  function (levels)
        # FROM: GDP deflator (index)
        # TO:   Approximate quarter-to-quarter percent change of gdp deflator,
        #       i.e.  quarterly gdp deflator inflation

        oneqtrpctchange(levels[!,:GDPDEF])
    end


    gdpdeflator_rev_transform = loggrowthtopct_annualized

    observables[:obs_gdpdeflator] = Observable(:obs_gdpdeflator, [:GDPDEF__FRED],
                                               gdpdeflator_fwd_transform, identity,#gdpdeflator_rev_transform,
                                               "GDP Deflator",
                                               "Q-to-Q Percent Change of GDP Deflator")

    ############################################################################
    ## 7. Nominal interest rate (shadow)
    ############################################################################

    nomrate_rev_transform = quartertoannual

    observables[:obs_nominalrate] = Observable(:obs_nominalrate, [:RB__BBL],
                                               x -> iden_fwd_transform(x, :RB), identity,#nomrate_rev_transform,
                                               "Nominal Interest Rate",
                                               "Nominal Interest Rate, augmented by shadow rate from Wu and Xia (2016) during ZLB")

    ############################################################################
    # 8. Wealth inequality
    ############################################################################

    observables[:obs_W90share] = Observable(:obs_W90share, [:w90share__BBL],
                                            x -> iden_fwd_transform(x, :w90share), identity,
                                            "Top 90% Wealth Share",
                                            "90th Percentile of Net Personal Wealth Distribution")

    ############################################################################
    # 9. Income inequality
    ############################################################################

    observables[:obs_I90share] = Observable(:obs_I90share, [:I90share__BBL],
                                            x -> iden_fwd_transform(x, :I90share), identity,
                                            "Top 90% Income Share",
                                            "Top 90% Percentile of Pre-Tax National Income Distribution")

    ############################################################################
    # 10. Idiosyncratic income risk
    ############################################################################

    observables[:obs_sigmasq] = Observable(:obs_sigmasq, [:sigma2__BBL],
                                           x -> iden_fwd_transform(x, :sigma2), identity,
                                           "Idiosyncratic Income Risk",
                                           "Variance of Idiosyncratic Income")

    ############################################################################
    # 11. Tax progressivity
    ############################################################################

    observables[:obs_taxprogressivity] = Observable(:obs_taxprogressivity, [:tauprog__BBL],
                                                    x -> iden_fwd_transform(x, :tauprog), identity,
                                                    "Tax Progressivity",
                                                    "Tax Progressivity from Ferriere and Navarro (2018)")

    m.observable_mappings = observables
end

function _init_original_observable_mappings!(m::BayerBornLuetticke, observables::OrderedDict{Symbol, Observable})

    population_mnemonic = get(get_setting(m, :population_mnemonic))

    iden_fwd_transform = function (levels, name)
        return levels[!, name]
    end

    ############################################################################
    ## 1. GDP growth per capita
    ############################################################################
    gdp_rev_transform = loggrowthtopct_annualized_percapita

    observables[:obs_gdp] = Observable(:obs_gdp, [:Ygrowth__BBL, population_mnemonic], # add population_mnenomic here just to make pop data available
                                       x -> iden_fwd_transform(x, :Ygrowth), identity,# gdp_rev_transform,
                                       "Real GDP Growth", "Real GDP Growth Per Capita")

    ############################################################################
    ## 2. Consumption growth per-capita
    ############################################################################

    cons_rev_transform = loggrowthtopct_annualized_percapita

    observables[:obs_consumption] = Observable(:obs_consumption, [:Cgrowth__BBL],
                                               x -> iden_fwd_transform(x, :Cgrowth), identity, #cons_rev_transform,
                                               "Real Consumption Growth", "Real Consumption Growth Per Capita")

    ############################################################################
    ## 3. Investment growth per capita
    ############################################################################

    invst_rev_transform = loggrowthtopct_annualized_percapita

    observables[:obs_investment] = Observable(:obs_investment, [:Igrowth__BBL],
                                              x -> iden_fwd_transform(x, :Igrowth), identity, #invst_rev_transform,
                                              "Real Investment Growth",
                                              "Real Investment Growth Per Capita")

    ############################################################################
    ## 4. Wage growth
    ############################################################################

    wages_rev_transform = loggrowthtopct_annualized

    observables[:obs_wages] = Observable(:obs_wages, [:wgrowth__BBL],
                                         x -> iden_fwd_transform(x, :wgrowth), identity,#wages_rev_transform,
                                         "Real Wage Growth",
                                         "Real Wage Growth in Non-Farm Business Sector")

    ############################################################################
    ## 5. Hours per capita
    ############################################################################

    hrs_rev_transform = logleveltopct_annualized_percapita

    observables[:obs_hours] = Observable(:obs_hours, [:N__BBL],
                                         x -> iden_fwd_transform(x, :N), identity,#hrs_rev_transform,
                                         "Hours",
                                         "Hours Per Capita")

    ############################################################################
    ## 6. GDP Deflator (Inflation)
    ############################################################################

    gdpdef_rev_transform = loggrowthtopct_annualized

    observables[:obs_gdpdeflator] = Observable(:obs_gdpdeflator, [:pi__BBL],
                                               x -> iden_fwd_transform(x, :pi), identity,#gdpdef_rev_transform,
                                               "GDP Deflator",
                                               "GDP Deflator Inflation")

    ############################################################################
    ## 7. Nominal interest rate (shadow)
    ############################################################################

    nomrate_rev_transform = quartertoannual

    observables[:obs_nominalrate] = Observable(:obs_nominalrate, [:RB__BBL],
                                               x -> iden_fwd_transform(x, :RB), identity,#nomrate_rev_transform,
                                               "Nominal Interest Rate",
                                               "Nominal Interest Rate, augmented by shadow rate from Wu and Xia (2016) during ZLB")

    ############################################################################
    # 8. Wealth inequality
    ############################################################################

    observables[:obs_W90share] = Observable(:obs_W90share, [:w90share__BBL],
                                            x -> iden_fwd_transform(x, :w90share), identity,
                                            "Top 90% Wealth Share",
                                            "90th Percentile of Net Personal Wealth Distribution")

    ############################################################################
    # 9. Income inequality
    ############################################################################

    observables[:obs_I90share] = Observable(:obs_I90share, [:I90share__BBL],
                                            x -> iden_fwd_transform(x, :I90share), identity,
                                            "Top 90% Income Share",
                                            "Top 90% Percentile of Pre-Tax National Income Distribution")

    ############################################################################
    # 10. Idiosyncratic income risk
    ############################################################################

    observables[:obs_sigmasq] = Observable(:obs_sigmasq, [:sigma2__BBL],
                                           x -> iden_fwd_transform(x, :sigma2), identity,
                                           "Idiosyncratic Income Risk",
                                           "Variance of Idiosyncratic Income")

    ############################################################################
    # 11. Tax progressivity
    ############################################################################

    observables[:obs_taxprogressivity] = Observable(:obs_taxprogressivity, [:tauprog__BBL],
                                                    x -> iden_fwd_transform(x, :tauprog), identity,
                                                    "Tax Progressivity",
                                                    "Tax Progressivity from Ferriere and Navarro (2018)")

    observables
end
