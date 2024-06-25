"""
Model805{T} <: AbstractDSGEModel{T}

The Model805 type defines the structure of the Smets & Wouters (2007) model augmented
with long run inflation expetations. We can then concisely pass around a Model object
to the remaining steps of the model (solve, estimate, and forecast).

### Fields

#### Parameters and Steady-States
* `parameters::Vector{AbstractParameter}`: Vector of all time-invariant model parameters.

* `steady_state::Vector`: Model steady-state values, computed as a
function of elements of `parameters`.

* `keys::OrderedDict{Symbol,Int}`: Maps human-readable names for all model
parameters and steady-states to their indices in `parameters` and
`steady_state`.

#### Inputs to Measurement and Equilibrium Condition Equations

The following fields are dictionaries that map human-readible names to
row and column indices in the matrix representations of of the
measurement equation and equilibrium conditions.

* `endogenous_states::OrderedDict{Symbol,Int}`: Maps each state to a column
in the measurement and equilibrium condition matrices.

* `exogenous_shocks::OrderedDict{Symbol,Int}`: Maps each shock to a column in
the measurement and equilibrium condition matrices.

* `expected_shocks::OrderedDict{Symbol,Int}`: Maps each expected shock to a
column in the measurement and equilibrium condition matrices.

* `equilibrium_conditions::OrderedDict{Symbol,Int}`: Maps each equlibrium
condition to a row in the model's equilibrium condition matrices.

* `endogenous_states_augmented::OrderedDict{Symbol,Int}`: Maps lagged states
to their columns in the measurement and equilibrium condition
equations. These are added after Gensys solves the model.

* `observables::OrderedDict{Symbol,Int}`: Maps each observable to a row in
the model's measurement equation matrices.

* `pseudo_observables::OrderedDict{Symbol,Int}`: Maps each pseudo-observable to
  a row in the model's pseudo-measurement equation matrices.

#### Model Specifications and Settings

* `spec::String`: The model specification identifier, \"smets_wouters\",
cached here for filepath computation.

* `subspec::String`: The model subspecification number,
indicating that some parameters from the original model spec (\"ss0\")
are initialized differently. Cached here for filepath computation.

* `settings::Dict{Symbol,Setting}`: Settings/flags that affect
computation without changing the economic or mathematical setup of
the model.

* `test_settings::Dict{Symbol,Setting}`: Settings/flags for testing mode

#### Other Fields

* `rng::MersenneTwister`: Random number generator. By default, it is
seeded to ensure replicability in algorithms that involve randomness
(such as Metropolis-Hastings).

* `testing::Bool`: Indicates whether the model is in testing mode. If
`true`, settings from `m.test_settings` are used in place of those in
`m.settings`.

* `observable_mappings::OrderedDict{Symbol,Observable}`: A dictionary that
  stores data sources, series mnemonics, and transformations to/from model units.
  The package will fetch data from the Federal Reserve Bank of
  St. Louis's FRED database; all other data must be downloaded by the
  user. See `load_data` and `Observable` for further details.

* `pseudo_observable_mappings::OrderedDict{Symbol,PseudoObservable}`: A
  dictionary that stores names and transformations to/from model units. See
  `PseudoObservable` for further details.
"""
mutable struct Model805{T} <: AbstractDSGEModel{T}
    parameters::ParameterVector{T}                  # vector of all time-invariant model parameters
    steady_state::ParameterVector{T}                # model steady-state values
    keys::OrderedDict{Symbol,Int}                   # human-readable names for all the model
                                                    # parameters and steady-n_states

    endogenous_states::OrderedDict{Symbol,Int}            # these fields used to create matrices in the
    exogenous_shocks::OrderedDict{Symbol,Int}             # measurement and equilibrium condition equations.
    expected_shocks::OrderedDict{Symbol,Int}              #
    equilibrium_conditions::OrderedDict{Symbol,Int}       #
    endogenous_states_augmented::OrderedDict{Symbol,Int}  #
    observables::OrderedDict{Symbol,Int}                  #
    pseudo_observables::OrderedDict{Symbol,Int}           #

    spec::String                                    # Model specification number (eg "m805")
    subspec::String                                 # Model subspecification (eg "ss0")
    settings::Dict{Symbol,Setting}                  # Settings/flags for computation
    test_settings::Dict{Symbol,Setting}             # Settings/flags for testing mode
    rng::MersenneTwister                            # Random number generator
    testing::Bool                                   # Whether we are in testing mode or not

    observable_mappings::OrderedDict{Symbol, Observable}
    pseudo_observable_mappings::OrderedDict{Symbol, PseudoObservable}
end

description(m::Model805) = "Smets-Wouters model, with long run inflation expectations"

"""
`init_model_indices!(m::Model805)`

Arguments:
`m:: Model805`: a model object

Description:
Initializes indices for all of `m`'s states, shocks, and equilibrium conditions.
"""
function init_model_indices!(m::Model805)
    # Endogenous states
    endogenous_states = [[
        :y_t, :c_t, :i_t, :qk_t, :k_t, :kbar_t, :u_t, :rk_t, :mc_t,
        :π_t, :μ_ω_t, :w_t, :L_t, :R_t, :g_t, :b_t, :μ_t, :z_t,
        :λ_f_t, :λ_f_t1, :λ_w_t, :λ_w_t1, :rm_t, :π_star_t, :Ec_t, :Eqk_t, :Ei_t,
        :Eπ_t, :EL_t, :Erk_t, :Ew_t, :y_f_t, :c_f_t,
        :i_f_t, :qk_f_t, :k_f_t, :kbar_f_t, :u_f_t, :rk_f_t, :w_f_t,
        :L_f_t, :r_f_t, :Ec_f_t, :Eqk_f_t, :Ei_f_t, :EL_f_t, :Erk_f_t, :ztil_t];
        [Symbol("rm_tl$i") for i = 1:n_anticipated_shocks(m)]]

    # Exogenous shocks
    exogenous_shocks = [[
        :g_sh, :b_sh, :μ_sh, :z_sh, :λ_f_sh, :λ_w_sh, :rm_sh, :π_star_sh];
        [Symbol("rm_shl$i") for i = 1:n_anticipated_shocks(m)]]

    # Expectations shocks
    expected_shocks = [
        :Ec_sh, :Eqk_sh, :Ei_sh, :Eπ_sh, :EL_sh, :Erk_sh, :Ew_sh, :Ec_f_sh,
        :Eqk_f_sh, :Ei_f_sh, :EL_f_sh, :Erk_f_sh]

    # Equilibrium conditions
    equilibrium_conditions = [[
        :eq_euler, :eq_inv, :eq_capval, :eq_output, :eq_caputl, :eq_capsrv, :eq_capev,
        :eq_mkupp, :eq_phlps, :eq_caprnt, :eq_msub, :eq_wage, :eq_mp, :eq_res, :eq_g, :eq_b, :eq_μ, :eq_z,
        :eq_λ_f, :eq_λ_w, :eq_rm, :eq_λ_f1, :eq_λ_w1, :eq_Ec,
        :eq_Eqk, :eq_Ei, :eq_Eπ, :eq_EL, :eq_Erk, :eq_Ew, :eq_euler_f, :eq_inv_f,
        :eq_capval_f, :eq_output_f, :eq_caputl_f, :eq_capsrv_f, :eq_capev_f, :eq_mkupp_f, :eq_caprnt_f, :eq_msub_f,
        :eq_res_f, :eq_Ec_f, :eq_Eqk_f, :eq_Ei_f, :eq_EL_f, :eq_Erk_f, :eq_ztil, :eq_π_star];
        [Symbol("eq_rml$i") for i=1:n_anticipated_shocks(m)]]

    # Additional states added after solving model
    # Lagged states and observables measurement error
    endogenous_states_augmented = [
        :y_t1, :c_t1, :i_t1, :w_t1, :π_t1, :L_t1, :Et_π_t]

    # Observables
    observables = keys(m.observable_mappings)

    # Pseudo-observables
    pseudo_observables = keys(m.pseudo_observable_mappings)

    for (i,k) in enumerate(endogenous_states);           m.endogenous_states[k]           = i end
    for (i,k) in enumerate(exogenous_shocks);            m.exogenous_shocks[k]            = i end
    for (i,k) in enumerate(expected_shocks);             m.expected_shocks[k]             = i end
    for (i,k) in enumerate(equilibrium_conditions);      m.equilibrium_conditions[k]      = i end
    for (i,k) in enumerate(endogenous_states);           m.endogenous_states[k]           = i end
    for (i,k) in enumerate(endogenous_states_augmented); m.endogenous_states_augmented[k] = i+length(endogenous_states) end
    for (i,k) in enumerate(observables);                 m.observables[k]                 = i end
    for (i,k) in enumerate(pseudo_observables);          m.pseudo_observables[k]          = i end
end

function Model805(subspec::String="ss0";
                  custom_settings::Array{S} where S<:Setting = Array{Setting{Bool}}(undef, 0),
                      testing = false)

    # Model-specific specifications
    spec               = split(basename(@__FILE__),'.')[1]
    subspec            = subspec
    settings           = Dict{Symbol,Setting}()
    test_settings      = Dict{Symbol,Setting}()
    rng                = MersenneTwister(0)        # Random Number Generator

    # initialize empty model
    m = Model805{Float64}(
            # model parameters and steady state values
            Vector{AbstractParameter{Float64}}(), Vector{Float64}(), OrderedDict{Symbol,Int}(),

            # model indices
            OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(),

            spec,
            subspec,
            settings,
            test_settings,
            rng,
            testing,
            OrderedDict{Symbol,Observable}(),
            OrderedDict{Symbol,PseudoObservable}())

    # Set settings
    model_settings!(m)
    default_test_settings!(m)
    for custom_setting in custom_settings
        m <= custom_setting
    end

    # Set observable and pseudo-observable transformations
    init_observable_mappings!(m)

    # Initialize parameters
    init_parameters!(m)
    init_model_indices!(m)
    init_subspec!(m)
    steadystate!(m)
    return m
end

"""
```
init_parameters!(m::Model805)
```

Initializes the model's parameters, as well as empty values for the steady-state
parameters (in preparation for `steadystate!(m)` being called to initialize
those).
"""
function init_parameters!(m::Model805)
    m <= parameter(:α,      0.1687, (1e-5, 0.999), (1e-5, 0.999),   SquareRoot(),     Normal(0.30, 0.05),         fixed=false,
                   description="α: Capital elasticity in the intermediate goods sector's Cobb-Douglas production function.",
                   tex_label="\\alpha")

    m <= parameter(:ζ_p,   0.7467, (1e-5, 0.999), (1e-5, 0.999),   SquareRoot(),     BetaAlt(0.5, 0.1),          fixed=false,
                   description="ζ_p: The Calvo parameter. In every period, intermediate goods producers optimize prices with probability (1-ζ_p). With probability ζ_p, prices are adjusted according to a weighted average of the previous period's inflation (π_t1) and steady-state inflation (π_star).",
                   tex_label="\\zeta_p")

    m <= parameter(:ι_p,   0.2684, (1e-5, 0.999), (1e-5, 0.999),   SquareRoot(),     BetaAlt(0.5, 0.15),         fixed=false,
                   description="ι_p: The weight attributed to last period's inflation in price indexation. (1-ι_p) is the weight attributed to steady-state inflation.",

                   tex_label="\\iota_p")

    m <= parameter(:δ,      0.025,  fixed=true,
                   description="δ: The capital depreciation rate.", tex_label="\\delta" )


    m <= parameter(:Upsilon,  1.000,  (0., 10.),     (1e-5, 0.),      ModelConstructors.Exponential(),    GammaAlt(1., 0.5),          fixed=true,
                   description="Υ: The trend evolution of the price of investment goods relative to consumption goods. Set equal to 1.",
                   tex_label="\\mathcal{\\Upsilon}")

    m <= parameter(:Φ,   1.4933, (1., 10.),     (1.00, 10.00),   ModelConstructors.Exponential(),    Normal(1.25, 0.12),         fixed=false,
                   description="Φ: Fixed costs.",
                   tex_label="\\Phi")

    m <= parameter(:S′′,       3.3543, (-15., 15.),   (-15., 15.),     Untransformed(),  Normal(4., 1.5),            fixed=false,
                   description="S'': The second derivative of households' cost of adjusting investment.", tex_label="S\\prime\\prime")

    m <= parameter(:h,        0.4656, (1e-5, 0.999), (1e-5, 0.999),   SquareRoot(),     BetaAlt(0.7, 0.1),          fixed=false,
                   description="h: Consumption habit persistence.", tex_label="h")

    m <= parameter(:ppsi,     0.7614, (1e-5, 0.999), (1e-5, 0.999),   SquareRoot(),     BetaAlt(0.5, 0.15),         fixed=false,
                   description="ppsi: Utilization costs.", tex_label="ppsi")

    m <= parameter(:ν_l,     1.0647, (1e-5, 10.),   (1e-5, 10.),     ModelConstructors.Exponential(),    Normal(2, 0.75),            fixed=false,
                   description="ν_l: The coefficient of relative risk aversion on the labor term of households' utility function.", tex_label="\\nu_l")

    m <= parameter(:ζ_w,   0.7922, (1e-5, 0.999), (1e-5, 0.999),   SquareRoot(),     BetaAlt(0.5, 0.1),          fixed=false,
                   description="ζ_w: (1-ζ_w) is the probability with which households can freely choose wages in each period. With probability ζ_w, wages increase at a geometrically weighted average of the steady state rate of wage increases and last period's productivity times last period's inflation.",
                   tex_label="\\zeta_w")

    m <= parameter(:ι_w,   0.5729, (1e-5, 0.999), (1e-5, 0.999),   SquareRoot(),     BetaAlt(0.5, 0.15),         fixed=false,
                   description="ι_w: No description available.",
                   tex_label="\\iota_w")

    m <= parameter(:λ_w,      1.5000,                                                                               fixed=true,
                   description="λ_w: The wage markup, which affects the elasticity of substitution between differentiated labor services.",
                   tex_label="\\lambda_w")

    m <= parameter(:β, 0.7420, (1e-7, 10.),   (1e-7, 10.),     ModelConstructors.Exponential(),    GammaAlt(0.25, 0.1),        fixed=false,  scaling = x -> 1/(1 + x/100),
                   description="β: Discount rate.",
                   tex_label="\\beta ")

    m <= parameter(:ψ1,  1.8678, (1e-5, 10.),   (1e-5, 10.00),   ModelConstructors.Exponential(),    Normal(1.5, 0.25),          fixed=false,
                   description="ψ₁: Weight on inflation gap in monetary policy rule.",
                   tex_label="\\psi_1")

    m <= parameter(:ψ2,  0.0715, (-0.5, 0.5),   (-0.5, 0.5),     Untransformed(),  Normal(0.12, 0.05),         fixed=false,
                   description="ψ₂: Weight on output gap in monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:ψ3, 0.2131, (-0.5, 0.5),   (-0.5, 0.5),     Untransformed(),  Normal(0.12, 0.05),         fixed=false,
                   description="ψ₃: Weight on rate of change of output gap in the monetary policy rule.",
                   tex_label="\\psi_3")

    m <= parameter(:π_star,   0.6231, (1e-5, 10.),   (1e-5, 10.),     ModelConstructors.Exponential(),    GammaAlt(0.75, 0.4),        fixed=false,  scaling = x -> 1 + x/100,
                   description="π_star: The steady-state rate of inflation.",
                   tex_label="\\pi_*")

    m <= parameter(:σ_c, 1.5073, (1e-5, 10.),   (1e-5, 10.),     ModelConstructors.Exponential(),    Normal(1.5, 0.37),          fixed=false,
                   description="σ_c: No description available.",
                   tex_label="\\sigma_{c}")

    m <= parameter(:ρ,      .8519, (1e-5, 0.999), (1e-5, 0.999),   SquareRoot(),     BetaAlt(0.75, 0.10),        fixed=false,
                   description="ρ: The degree of inertia in the monetary policy rule.",
                   tex_label="\\rho")

    m <= parameter(:ϵ_p,     10.000,                                                                               fixed=true,
                   description="ϵ_p: No description available.",
                   tex_label="\\varepsilon_{p}")

    m <= parameter(:ϵ_w,     10.000,                                                                               fixed=true,
                   description="ϵ_w: No description available.",
                   tex_label="\\varepsilon_{w}")


    # exogenous processes - level
    m <= parameter(:γ,      0.3085, (-5.0, 5.0),     (-5., 5.),     Untransformed(), Normal(0.4, 0.1),            fixed=false, scaling = x -> x/100,
                   description="γ: The log of the steady-state growth rate of technology.",
                   tex_label="\\gamma")

    m <= parameter(:Lmean,  -45., (-1000., 1000.), (-1e3, 1e3),   Untransformed(), Normal(-45, 5),   fixed=false,
                   description="Lmean: No description available.",
                   tex_label="Lmean")

    m <= parameter(:g_star,    0.1800,                                                                               fixed=true,
                   description="g_star: No description available.",
                   tex_label="g_*")

    # exogenous processes - autocorrelation
    m <= parameter(:ρ_g,      0.9930, (1e-5, 0.999),   (1e-5, 0.999), SquareRoot(),    BetaAlt(0.5, 0.2),           fixed=false,
                   description="ρ_g: AR(1) coefficient in the government spending process.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_b,      0.8100, (1e-5, 0.999),   (1e-5, 0.999), SquareRoot(),    BetaAlt(0.5, 0.2),           fixed=false,
                   description="ρ_b: AR(1) coefficient in the intertemporal preference shifter process.",
                   tex_label="\\rho_b")

    m <= parameter(:ρ_μ,     0.7790, (1e-5, 0.999),   (1e-5, 0.999), SquareRoot(),    BetaAlt(0.5, 0.2),           fixed=false,
                   description="ρ_μ: AR(1) coefficient in capital adjustment cost process.",
                   tex_label="\\rho_{\\mu}")

    m <= parameter(:ρ_z,      0.9676, (1e-5, 0.999),   (1e-5, 0.999), SquareRoot(),    BetaAlt(0.5, 0.2),           fixed=false,
                   description="ρ_z: AR(1) coefficient in the technology process.",
                   tex_label="\\rho_z")

    m <= parameter(:ρ_λ_f,    0.8372, (1e-5, 0.999),   (1e-5, 0.999), SquareRoot(),    BetaAlt(0.5, 0.2),           fixed=false,
                   description="ρ_λ_f: AR(1) coefficient in the price mark-up shock process.",
                   tex_label="\\rho_{\\lambda_f}")

    m <= parameter(:ρ_λ_w,    0.9853, (1e-5, 0.999),   (1e-5, 0.999), SquareRoot(),    BetaAlt(0.5, 0.2),           fixed=false,
                   description="ρ_λ_w: AR(1) coefficient in the wage mark-up shock process.", # CHECK THIS
                   tex_label="\\rho_{\\lambda_w}")

    m <= parameter(:ρ_rm,     0.9000, (1e-5, 0.999),   (1e-5, 0.999), SquareRoot(),    BetaAlt(0.5, 0.2),           fixed=false,
                   description="ρ_rm: AR(1) coefficient in the monetary policy shock process.", # CHECK THIS
                   tex_label="\\rho_{rm}")

m <= parameter(:ρ_π_star,     0.9900, (1e-5, 0.999),   (1e-5, 0.999), SquareRoot(),    BetaAlt(0.5, 0.2),
               fixed=true,
               description="ρ_π_star: AR(1) coefficient in the π* process.",
               tex_label="\\rho_{\\pi^*}")


    # exogenous processes - standard deviation
    m <= parameter(:σ_g,      0.6090, (1e-8, 5.),      (1e-8, 5.),    ModelConstructors.Exponential(),   RootInverseGamma(2., 0.10),  fixed=false,
                   description="σ_g: The standard deviation of the government spending process.",
                   tex_label="\\sigma_{g}")

    m <= parameter(:σ_b,      0.1818, (1e-8, 5.),      (1e-8, 5.),    ModelConstructors.Exponential(),   RootInverseGamma(2., 0.10),  fixed=false,
                   description="σ_b: No description available.",
                   tex_label="\\sigma_{b}")

    m <= parameter(:σ_μ,     0.4601, (1e-8, 5.),      (1e-8, 5.),    ModelConstructors.Exponential(),   RootInverseGamma(2., 0.10),  fixed=false,
                   description="σ_μ: The standard deviation of the exogenous marginal efficiency of investment shock process.",
                   tex_label="\\sigma_{\\mu}")

    m <= parameter(:σ_z,      0.4618, (1e-8, 5.),      (1e-8, 5.),    ModelConstructors.Exponential(),   RootInverseGamma(2., 0.10),  fixed=false,
                   description="σ_z: No description available.",
                   tex_label="\\sigma_{z}")

    m <= parameter(:σ_λ_f,    0.1455, (1e-8, 5.),      (1e-8, 5.),    ModelConstructors.Exponential(),   RootInverseGamma(2., 0.10),  fixed=false,
                   description="σ_λ_f: The mean of the process that generates the price elasticity of the composite good.  Specifically, the elasticity is (1+λ_{f,t})/(λ_{f_t}).",
                   tex_label="\\sigma_{\\lambda_f}")

    m <= parameter(:σ_λ_w,    0.2089, (1e-8, 5.),      (1e-8, 5.),    ModelConstructors.Exponential(),   RootInverseGamma(2., 0.10),  fixed=false,
                   description="σ_λ_w: No description available.",
                   tex_label="\\sigma_{\\lambda_w}")

    m <= parameter(:σ_rm,     0.2397, (1e-8, 5.),      (1e-8, 5.),    ModelConstructors.Exponential(),   RootInverseGamma(2., 0.10),  fixed=false,
                   description="σ_rm: No description available.",
                   tex_label="\\sigma_{rm}")

    m <= parameter(:σ_π_star,      0.0300, (1e-8, 5.),      (1e-8, 5.),    ModelConstructors.Exponential(),   RootInverseGamma(2., 0.10),  fixed=false,
                   description="σ_π_star: The standard deviation of the π* process.",
                   tex_label="\\sigma_{\\pi^*}")



    m <= parameter(:η_gz,       0.5632, (1e-5, 0.999), (1e-5, 0.999), SquareRoot(),
                   BetaAlt(0.50, 0.20), fixed=false,
                   description="η_gz: Correlate g and z shocks.",
                   tex_label="\\eta_{gz}")

    m <= parameter(:η_λ_f,      0.7652, (1e-5, 0.999), (1e-5, 0.999), SquareRoot(),
                   BetaAlt(0.50, 0.20),         fixed=false,
                   description="η_λ_f: No description available.",
                   tex_label="\\eta_{\\lambda_f}")

    m <= parameter(:η_λ_w,      0.8936, (1e-5, 0.999), (1e-5, 0.999), SquareRoot(),
                   BetaAlt(0.50, 0.20),         fixed=false,
                   description="η_λ_w: AR(2) coefficient on wage markup shock process.",
                   tex_label="\\eta_{\\lambda_w}")


    # steady states
    m <= SteadyStateParameter(:zstar,  NaN, description="steady-state growth rate of productivity", tex_label="\\z_*")
    m <= SteadyStateParameter(:rstar,   NaN, description="steady-state something something", tex_label="\\r_*")
    m <= SteadyStateParameter(:Rstarn,  NaN, description="steady-state something something", tex_label="\\R_*_n")
    m <= SteadyStateParameter(:rkstar,  NaN, description="steady-state something something", tex_label="\\BLAH")
    m <= SteadyStateParameter(:wstar,   NaN, description="steady-state something something", tex_label="\\w_*")
    m <= SteadyStateParameter(:Lstar,   NaN, description="steady-state something something", tex_label="\\L_*")
    m <= SteadyStateParameter(:kstar,   NaN, description="Effective capital that households rent to firms in the steady state.", tex_label="\\k_*")
    m <= SteadyStateParameter(:kbarstar, NaN, description="Total capital owned by households in the steady state.", tex_label="\\bar{k}_*")
    m <= SteadyStateParameter(:istar,  NaN, description="Detrended steady-state investment", tex_label="\\i_*")
    m <= SteadyStateParameter(:ystar,  NaN, description="steady-state something something", tex_label="\\y_*")
    m <= SteadyStateParameter(:cstar,  NaN, description="steady-state something something", tex_label="\\c_*")
    m <= SteadyStateParameter(:wl_c,   NaN, description="steady-state something something", tex_label="\\wl_c")
end


"""
```
steadystate!(m::Model805)
```

Calculates the model's steady-state values. `steadystate!(m)` must be called whenever the parameters of `m` are updated.
"""
function steadystate!(m::Model805)
    m[:zstar]    = log(1+m[:γ]) + m[:α]/(1-m[:α])*log(m[:Upsilon])
    m[:rstar]    = exp(m[:σ_c]*m[:zstar]) / m[:β]
    m[:Rstarn]   = 100*(m[:rstar]*m[:π_star] - 1)
    m[:rkstar]   = m[:rstar]*m[:Upsilon] - (1-m[:δ])
    m[:wstar]    = (m[:α]^m[:α] * (1-m[:α])^(1-m[:α]) * m[:rkstar]^(-m[:α]) / m[:Φ])^(1/(1-m[:α]))
    m[:Lstar]    = 1.
    m[:kstar]    = (m[:α]/(1-m[:α])) * m[:wstar] * m[:Lstar] / m[:rkstar]
    m[:kbarstar] = m[:kstar] * (1+m[:γ]) * m[:Upsilon]^(1 / (1-m[:α]))
    m[:istar]    = m[:kbarstar] * (1-((1-m[:δ])/((1+m[:γ]) * m[:Upsilon]^(1/(1-m[:α])))))
    m[:ystar]    = (m[:kstar]^m[:α]) * (m[:Lstar]^(1-m[:α])) / m[:Φ]
    m[:cstar]    = (1-m[:g_star])*m[:ystar] - m[:istar]
    m[:wl_c]     = (m[:wstar]*m[:Lstar])/(m[:cstar]*m[:λ_w])

    return m
end


function model_settings!(m::Model805)

    default_settings!(m)

    # Anticipated shocks
    m <= Setting(:n_anticipated_shocks, 0)
    m <= Setting(:n_anticipated_shocks_padding, 20)

    # Data vintage
    m <= Setting(:data_vintage, "150827")

    # Conditional data variables
    m <= Setting(:cond_semi_names, [:obs_nominalrate])
    m <= Setting(:cond_full_names, [:obs_gdp, :obs_nominalrate])

    # Estimation
    m <= Setting(:reoptimize, true)
    m <= Setting(:recalculate_hessian, true)

    # Forecast
    m <= Setting(:forecast_pseudoobservables, false)
end
