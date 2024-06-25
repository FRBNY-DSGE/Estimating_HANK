"""
```
init_subspec!(m::DSGEVAR)
```
initializes a model subspecification of a DSGEVAR. We utilize
subspecifications to distinguish between possible VARs, such as
specifying the observables and the number of lags.
"""
function init_subspec!(m::DSGEVAR)
    if subspec(m) == "ss0"
        return # initializes an empty DSGEVAR
    elseif subspec(m) == "ss1"
        ss1!(m)
    elseif subspec(m) == "ss2"
        ss2!(m)
    elseif subspec(m) == "ss3"
        ss3!(m)
    elseif subspec(m) == "ss10"
        ss10!(m)
    elseif subspec(m) == "ss11"
        ss11!(m)
    elseif subspec(m) == "ss12"
        ss12!(m)
    elseif subspec(m) == "ss13"
        ss13!(m)
    elseif subspec(m) == "ss20"
        ss20!(m)
    elseif subspec(m) == "ss21"
        ss21!(m)
    else
        error("DSGEVAR subspec $(subspec(m)) is not defined.")
    end

    return m
end

# Basic DSGE-VARs, mainly for testing
function ss1!(m::DSGEVAR)
    observables = [:obs_hours, :obs_gdpdeflator]
    lags        = 4
    λ           = .5
    update!(m; observables = observables, lags = lags, λ = λ)
end

function ss2!(m::DSGEVAR)
    observables = [:obs_gdp, :obs_cpi]
    lags        = 4
    λ           = 0.5
    update!(m; observables = observables, lags = lags, λ = λ)
end

function ss3!(m::DSGEVAR)
    observables = [:obs_gdp, :obs_cpi, :obs_nominalrate]
    lags        = 4
    λ           = 0.5
    update!(m; observables = observables, lags = lags, λ = λ)
end

# Some DSGE-VARs for Model1002
function ss10!(m::DSGEVAR)
    observables = [:obs_hours, :obs_gdpdeflator, :laborshare_t, :NominalWageGrowth]
    lags        = 4
    λ           = .5
    update!(m; observables = observables, lags = lags, λ = λ)
end

function ss11!(m::DSGEVAR)
    observables = [:obs_hours, :π_t, :laborshare_t, :NominalWageGrowth]
    lags        = 4
    λ           = .5
    update!(m; observables = observables, lags = lags, λ = λ)
end

function ss12!(m::DSGEVAR)
    observables = [:obs_hours, :π_t, :laborshare_t, :NominalWageGrowth, :Epi_t]
    lags        = 4
    λ           = .5
    update!(m; observables = observables, lags = lags, λ = λ)
end

function ss13!(m::DSGEVAR)
    observables = [:obs_spread, :obs_hours, :π_t, :laborshare_t, :NominalWageGrowth, :Epi_t]
    lags        = 4
    λ           = .5
    update!(m; observables = observables, lags = lags, λ = λ)
end

# DSGE-VARs for BayerBornLuetticke
function ss20!(m::DSGEVAR)
    observables = [:obs_gdp, :obs_consumption, :obs_investment, :obs_wages, :obs_hours, :obs_gdpdeflator, :obs_nominalrate]
    lags        = 4
    λ           = 1.
    update!(m; observables = observables, lags = lags, λ = λ)
end

function ss21!(m::DSGEVAR)
    observables = [:obs_gdp, :obs_consumption, :obs_investment, :obs_wages, :obs_hours, :obs_gdpdeflator, :obs_nominalrate, :obs_sigmasq]
    lags        = 4
    λ           = 1.
    update!(m; observables = observables, lags = lags, λ = λ)
end
