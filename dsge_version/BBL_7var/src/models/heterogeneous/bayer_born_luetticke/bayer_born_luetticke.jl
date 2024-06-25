#CHANGED:
#changed the original replication settings

"""
```
BayerBornLuetticke{T} <: AbstractHeterogeneousModel{T}
```

### Fields

#### Parameters and Steady-States
* `parameters::Vector{AbstractParameter}`: Vector of all time-invariant model
  parameters.

* `steady_state::Vector{AbstractParameter}`: Model steady-state values, computed
  as a function of elements of `parameters`.

* `keys::OrderedDict{Symbol,Int}`: Maps human-readable names for all model
  parameters and steady-states to their indices in `parameters` and
  `steady_state`.

#### Inputs to Measurement and Equilibrium Condition Equations

The following fields are dictionaries that map human-readable names to row and
column indices in the matrix representations of of the measurement equation and
equilibrium conditions.

* `endogenous_states::OrderedDict{Symbol,Int}`: Maps each state to a column in
  the measurement and equilibrium condition matrices.

* `exogenous_shocks::OrderedDict{Symbol,Int}`: Maps each shock to a column in the measurement and equilibrium condition matrices.

* `expected_shocks::OrderedDict{Symbol,Int}`: Maps each expected shock to a
  column in the measurement and equilibrium condition matrices.

* `equilibrium_conditions::OrderedDict{Symbol,Int}`: Maps each equlibrium
  condition to a row in the model's equilibrium condition matrices.

* `observables::OrderedDict{Symbol,Int}`: Maps each observable to a row in the
  model's measurement equation matrices.

#### Model Specifications and Settings

* `spec::String`: The model specification identifier, \"an_schorfheide\", cached
  here for filepath computation.

* `subspec::String`: The model subspecification number, indicating that some
  parameters from the original model spec (\"ss1\") are initialized
  differently. Cached here for filepath computation.

* `settings::Dict{Symbol,Setting}`: Settings/flags that affect computation
  without changing the economic or mathematical setup of the model.

* `test_settings::Dict{Symbol,Setting}`: Settings/flags for testing mode

#### Other Fields

* `rng::MersenneTwister`: Random number generator. Can be is seeded to ensure
  reproducibility in algorithms that involve randomness (such as
  Metropolis-Hastings).

* `testing::Bool`: Indicates whether the model is in testing mode. If `true`,
  settings from `m.test_settings` are used in place of those in `m.settings`.

* `observable_mappings::OrderedDict{Symbol,Observable}`: A dictionary that
  stores data sources, series mnemonics, and transformations to/from model units.
  The package will fetch data from the Federal Reserve Bank of
  St. Louis's FRED database; all other data must be downloaded by the
  user. See `load_data` and `Observable` for further details.

"""
mutable struct BayerBornLuetticke{T} <: AbstractHetModel{T}
    parameters::ParameterVector{T}               # vector of all time-invariant model parameters
    steady_state::ParameterVector{T}             # model steady-state values

    # Temporary to get it to work. Need to
    # figure out a more flexible way to define
    # "grids" that are not necessarily quadrature
    # grids within the model
# TODO: maybe add field/type to hold reduction information e.g. DCTindices (but not coefficient values), copula info
    grids::OrderedDict{Symbol,Union{Grid, Array, T}}
    keys::OrderedDict{Symbol,Int}                    # Human-readable names for all the model
                                              # parameters and steady-states

    state_variables::Vector{Symbol}                  # Vector of symbols of the state variables
    jump_variables::Vector{Symbol}                   # Vector of symbols of the jump variables
    aggregate_state_variables::Vector{Symbol}                  # Vector of symbols of the state variables
    aggregate_jump_variables::Vector{Symbol}                   # Vector of symbols of the jump variables
#=    normalized_model_states::Vector{Symbol}          # All of the distributional model
                                                     # state variables that need to be normalized=#

    # Vector of ranges corresponding to normalized (post Klein solution) indices
    aggregate_endogenous_states::OrderedDict{Symbol,Int}
    endogenous_states::OrderedDict{Symbol,UnitRange{Int}}
    exogenous_shocks::OrderedDict{Symbol,Int}
    expected_shocks::OrderedDict{Symbol,Int}
    equilibrium_conditions::OrderedDict{Symbol,UnitRange{Int}}
    aggregate_equilibrium_conditions::OrderedDict{Symbol,Int}
    endogenous_states_augmented::OrderedDict{Symbol,Int}
    observables::OrderedDict{Symbol,Int}
    pseudo_observables::OrderedDict{Symbol,Int}

    spec::String                                     # Model specification (eg "bayer_born_luetticke")
    subspec::String                                  # Model subspecification (eg "ss1")
    settings::Dict{Symbol,Setting}                   # Settings/flags for computation
    test_settings::Dict{Symbol,Setting}              # Settings/flags for testing mode
    rng::MersenneTwister                             # Random number generator
    testing::Bool                                    # Whether we are in testing mode or not
    observable_mappings::OrderedDict{Symbol, Observable}
    pseudo_observable_mappings::OrderedDict{Symbol, PseudoObservable}
end

description(m::BayerBornLuetticke) = "BayerBornLuetticke, $(m.subspec)"

"""
`init_model_indices!(m::BayerBornLuetticke)`

Arguments:
`m:: BayerBornLuetticke`: a model object

Description:
Initializes indices for all of `m`'s model states, shocks, and equilibrium conditions.
By model states, we mean the states in the model's reduced form state-space representation,
hence these states include both predetermined states and jumps.
"""
function init_model_indices!(m::BayerBornLuetticke)
    # TODO: maybe want to rename some of the indices to make it explicit
    #       exactly what we're perturbing (e.g. we are perturbing DCT coefficients or CDFs)

    # Predetermined states
    # TODO: delete states that should just be augmented states
    m.state_variables = [:marginal_pdf_m′_t, :marginal_pdf_k′_t, :marginal_pdf_y′_t, :copula′_t,

                         :A′_t, :Z′_t, :Ψ′_t, :RB′_t, :μ_p′_t, :μ_w′_t, :σ′_t, # TODO: rename σ′_t to σ_sq′_t to make it clear it's the variance
                         :Y′_t1, :B′_t1, :T′_t1, :I′_t1, :w′_t1, :q′_t1, :C′_t1,
                         :avg_tax_rate′_t1, :τ_prog′_t1,

                         :G_sh′_t, :P_sh′_t, :R_sh′_t, :S_sh′_t]

#=    m.state_variables = [# Endogenous function-valued states
                         :marginal_pdf_m′_t, :marginal_pdf_k′_t, :marginal_pdf_y′_t, :copula′_t,

                         # Endogenous scalar-valued states (e.g. lags)
                         :union_retained′_t, :retained′_t,
                         :Y′_t1, :B′_t1, :T′_t1, :I′_t1, :w′_t1, :q′_t1, :C′_t1,
                         :avg_tax_rate′_t1, :τ_prog′_t1,

                         # Exogenous scalar-valued states:
                         :A′_t, :Z′_t, :Ψ′_t, :RB′_t, :μ_p′_t, :μ_w′_t, :σ′_t,
                         :G_sh′_t, :P_sh′_t, :R_sh′_t, :S_sh′_t]=#

    m.aggregate_state_variables = m.state_variables[5:end]


    m.jump_variables = [:Vm′_t, :Vk′_t,
                        # Function valued jumps above, Distribution Names
                        :Gini_C′_t, :Gini_X′_t, :I90_share′_t, :I90_share_net′_t,:W90_share′_t, :sd_log_y′_t,
                        # Endogenous scalar-valued jumps
                        :rk′_t, :w′_t, :K′_t, :π′_t, :π_w′_t, :Y′_t, :C′_t, :q′_t, :N′_t, :mc′_t,
                        :mc_w′_t, :u′_t, :Ht′_t, :avg_tax_rate′_t, :T′_t, :I′_t, :B′_t,
                        :BD′_t, :BY′_t, :TY′_t, :mc_w_w′_t, :G′_t, :τ_level′_t, :τ_prog′_t, :Ygrowth′_t,
                        :Bgrowth′_t, :Igrowth′_t, :wgrowth′_t, :Cgrowth′_t,
                        :Tgrowth′_t, :LP′_t, :LP_XA′_t, :union_profits′_t,
                        :profits′_t]


    m.aggregate_jump_variables = m.jump_variables[3:end]

    # Update number of scalar states and jumps
    m <= Setting(:n_scalar_states, length(get_aggregate_state_variables(m)))
    m <= Setting(:n_scalar_jumps, length(get_aggregate_jump_variables(m)))
    m <= Setting(:n_scalar_variables,  get_setting(m, :n_scalar_jumps) + get_setting(m, :n_scalar_states))

    m <= Setting(:PRightAll, Matrix{Float64}(undef, 0, 0))
    m <= Setting(:State2Control, Matrix{Float64}(undef, 0, 0))
    m <= Setting(:LOMstate, Matrix{Float64}(undef, 0, 0))
    m <= Setting(:PRightStates, Matrix{Float64}(undef, 0, 0))

    # Exogenous shocks
    exogenous_shocks = collect([:A_sh, :Z_sh, :Ψ_sh, :μ_p_sh, :μ_w_sh, :G_sh, :R_sh, :S_sh, :P_sh])

    shock2state_map = Dict(:A_sh => :A_t, :Z_sh => :Z_t, :Ψ_sh => :Ψ_t, :μ_p_sh => :μ_p_t, :μ_w_sh => :μ_w_t, :G_sh => :G_sh_t, :P_sh => :P_sh_t, :R_sh => :R_sh_t, :S_sh => :S_sh_t)
    m <= Setting(:shock2state, shock2state_map)

    standard_deviation_dictionary = Dict(:A_sh => :σ_A, :Z_sh => :σ_Z, :Ψ_sh => :σ_Ψ, :μ_p_sh => :σ_μ_p,
                                     :μ_w_sh => :σ_μ_w, :G_sh => :σ_G, :R_sh => :σ_R, :S_sh => :σ_S,
                                     :P_sh => :σ_P)

    ## Check These Values Commented Out the 48, 49 etc. values, NOT SURE WHY THEY WERE SET THIS WAY
  #=  standard_deviation_dictionary = Dict(:A_sh => 48, :Z_sh => 49, :Ψ_sh => 50, :μ_p_sh => 51,
                                     :μ_w_sh => 52, :G_sh => 55, :R_sh => 54, :S_sh => 53,
                                     :P_sh => 56)=#

#standard_deviation_dictionary = Dict(:A_sh => 0.00033, :Z_sh => 0.00033, :Ψ_sh => 0.00033, :μ_p_sh => 0.00033,
                                    # :μ_w_sh => 0.00033, :G_sh => 0.00033, :R_sh => 0.00033, :S_sh => 0.511538,
                                    # :P_sh => 0.00033)

    m <= Setting(:shock_to_deviation_dict, standard_deviation_dictionary)

    # Observables
    observables = keys(m.observable_mappings)

    # Pseudo-Observables
    pseudo_observables = keys(m.pseudo_observable_mappings)

    # Initialize indices for exogenous shocks, observables, and pseudo-observables
    for (i,k) in enumerate(exogenous_shocks);   m.exogenous_shocks[k] = i end
    for (i,k) in enumerate(observables);        m.observables[k]      = i end
    for (i,k) in enumerate(pseudo_observables); m.pseudo_observables[k]      = i end

    # Initialize indices for
    # reduced-form endogenous_states (from gensys notation) and equilibrium conditions
    # setup_indices!(m) # TODO: This might have to be called AFTER steadystate! is called and leave entries empty otherwise

    # Create dict for unreduced endogenous_states
    # to facilitate repeated solutions of steady state
#=    m <= Setting(:n_model_states_unreduced, length(m.endogenous_states),
                 "Number of model states (incl. both predetermined states and jumps) before any state-space reduction")=#

    # Additional states added after solving model, namely
    # lagged states and observables measurement error
    # TODO: add augmented variables back
#=    endogenous_states_augmented = [:C_t1]

    for (i, k) in enumerate(endogenous_states_augmented)
        m.endogenous_states_augmented[k] = i + length(m.endogenous_states[get_setting(m, :jumps)[end]])
    end
    m <= Setting(:n_model_states_augmented, get_setting(m, :n_model_states) +
                 length(m.endogenous_states_augmented))=#
end

# TODO: maybe add coarse as a kwarg
function BayerBornLuetticke(subspec::String="ss1";
                            custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
                            load_steadystate::Bool = false, load_jacobian::Bool = false,
                            testing = false)

    # Model-specific specifications
    spec               = "bayer_born_luetticke"
    subspec            = subspec
    settings           = Dict{Symbol,Setting}()
    test_settings      = Dict{Symbol,Setting}()
    rng                = MersenneTwister(0)

    # initialize empty model
    m = BayerBornLuetticke{Float64}(
            # model parameters and steady state values
            Vector{AbstractParameter{Float64}}(), Vector{Float64}(),
            # grids and keys
            OrderedDict{Symbol,Union{Grid, Array, Float64}}(), OrderedDict{Symbol,Int}(),

            # state_variables, jump_variables,
            Vector{Symbol}(), Vector{Symbol}(),

            # aggregate_state_variables, aggregate_jump_variables
            Vector{Symbol}(), Vector{Symbol}(),

            # model indices
            # endogenous states
            OrderedDict{Symbol, Int}(), # TODO: label these
            OrderedDict{Symbol, UnitRange{Int}}(), # TODO: label these
            OrderedDict{Symbol, Int}(), OrderedDict{Symbol, Int}(),
            OrderedDict{Symbol, UnitRange{Int}}(), OrderedDict{Symbol, Int}(),
            OrderedDict{Symbol, Int}(), OrderedDict{Symbol, Int}(),
            OrderedDict{Symbol, Int}(),

            spec,
            subspec,
            settings,
            test_settings,
            rng,
            testing,
            OrderedDict{Symbol,Observable}(),
            OrderedDict{Symbol,PseudoObservable}())

    default_settings!(m)

    # Set observable transformations
    init_observable_mappings!(m)

    # Set settings
    model_settings!(m)
    for custom_setting in values(custom_settings)
        m <= custom_setting
    end

    if get_setting(m, :original_dataset)
        _init_original_observable_mappings!(m, m.observable_mappings)
    end

    # Initialize model indices
    init_model_indices!(m)

    # Init what will keep track of # of states, jumps, and states/jump indices
    # (need model indices first)
    # init_states_and_jumps!(m, states, jumps)

    # Initialize parameters
    init_parameters!(m)

    # Initialize grids
    init_grids!(m; coarse = !load_steadystate) # if steady state has not been computed, we start from a coarse grid

    # Load the steady state if it has already been computed
    # from the filepath get_setting(m, :steadystate_output_file)
    if load_steadystate
        load_steadystate!(m)
    end
    # steadystate!(m)

    # So that the indices of m.endogenous_states reflect the normalization
    # normalize_model_state_indices!(m)

    #init_subspec!(m)

    # Load Jacobian if it has already been computed
    if load_jacobian
        load_jacobian!(m)
    end

    return m
end

"""
```
init_parameters!(m::BayerBornLuetticke)
```

Initializes the model's parameters, as well as empty values for the steady-state
parameters (in preparation for `steadystate!(m)` being called to initialize
those).
"""
function init_parameters!(m::BayerBornLuetticke)
    ## Compare with BBL m_par (not just what's specified in Parameters.jl because some change based on the mode of the prior distribution)
    ######################################
    # Parameters that affect steady-state
    ######################################

    # HH preferences
    m <= parameter(:ξ, 4., fixed = true,
                   description = "ξ: risk aversion of households",
                   tex_label = "\\xi")
    m <= parameter(:γ, 2., fixed = true,
                   description = "γ: inverse Frisch elasticity",
                   tex_label = "\\gamma")
    m <= parameter(:β, 0.9842, fixed = true,
                   description = "β: discount factor",
                   tex_label = "\\beta")
    m <= parameter(:λ, 0.095, fixed = true,
                   description = "λ: portfolio adjustment probability",
                   tex_label = "\\lambda")
    m <= parameter(:γ_scale, 0.2, fixed = true,
                   description = "γ_scale: disutility of labor",
                   tex_label = "\\gamma_{\\text{scale}}")

    # Individual income process
    m <= parameter(:ρ_h, 0.98, fixed = true,
                   description = "ρ_h: autocorrelation of income shock",
                   tex_label = "\\rho_{h}")
    m <= parameter(:σ_h, 0.12, fixed = true,
                   description = "σ_h: standard deviation of income shock",
                   tex_label = "\\sigma_{h}")
    m <= parameter(:ι, 1. / 16, fixed = true,
                   description = "ι: probability of return to worker",
                   tex_label = "\\iota")
    m <= parameter(:ζ, 1. / 3750., fixed = true,
                   description = "ζ: probability of becoming an entrepreneur",
                   tex_label = "\\zeta")

    # Technological parameters
    m <= parameter(:α, 0.318, fixed = true,
                   description = "Capital share", tex_label = "\\alpha")
    m <= parameter(:δ_0, (.07 + .016) / 4., fixed = true,
                   description = "Depreciation rate", tex_label = "\\delta_0")

    # Technological parameters
    ## Old Value 5. rather than 4.2
    m <= parameter(:δ_s, 4.2, (0., 1e2), (0., 1e2), ModelConstructors.Exponential(),
                   GammaAlt(5., 2.), fixed = false,
                   description = "Depreciation increase from flexible utilization",
                   tex_label = "\\delta_s")
#CHANGE delta_s, phi to fixed = false eventually, may have to modify after when trying to fix

    ## Old Value 4. rather than 3.0
    m <= parameter(:ϕ, 3.0, (0., 1e2), (0., 1e2), ModelConstructors.Exponential(),
                   GammaAlt(4., 2.), fixed = false,
                   description = "Depreciation increase from flexible utilization",
                   tex_label = "\\phi")


    # NK Phillips Curve #BBL's \mu
    m <= parameter(:μ_p, 1.1, fixed = true,
                   description = "Price markup", tex_label = "\\mu_p")

    # NK Phillips Curve
    ## Old Value 1./11. rather than 0.099
    m <= parameter(:κ_p, 0.09900000000000002, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   GammaAlt(0.1, 0.01), fixed = false,
                   description = "Price adjustment cost (Calvo probability)",
                   tex_label = "\\kappa_p")

    m <= parameter(:μ_w, 1.1, fixed = true,
                   description = "Wage markup", tex_label = "\\mu_w")
    ## Old Value 1./11 rather than 0.099
    m <= parameter(:κ_w, 0.09900000000000002, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   GammaAlt(0.1, 0.01), fixed = false,
                   description = "Wage adjustment cost (Calvo probability)",
                   tex_label = "\\kappa_w")



    m <= parameter(:ψ, 0.1, fixed = true,
                   description = "Steady-state bond to capital ratio", tex_label = "\\psi")


    m <= parameter(:τ_lev, 0.825, fixed = true,
                   description = "Steady-state income tax rate level", tex_label = "\\tau^L")
    m <= parameter(:τ_prog, 0.12, fixed = true,
                   description = "Steady-state income tax rate progressivity", tex_label = "\\tau^P")


    # Unused Parameters
    m <= parameter(:Runused, 1.01, fixed = true,
                   description = "Unused", tex_label = "\\R")
    m <= parameter(:Kunused, 40.0, fixed = true,
                   description = "Unused", tex_label = "\\K")


    m <= parameter(:π, 1.0^0.25 , fixed = true,
                   description = "Steady-state inflation", tex_label = "\\pi")

    # Monetary policy
    m <= parameter(:RB, m[:π]*(1.0.^0.25) , fixed = true,
                   description = "Steady-state nominal interest rate", tex_label = "\\RB")


    m <= parameter(:Rbar, (m[:π] * (1.0675 ^ 0.25) - 1.), fixed = true,
                   description = "Borrowing wedge in interest rate", tex_label = "\\bar{R}")




     # Exogenous processes - autocorrelation
    ## Old Value 0.9 rather than 0.5
    m <= parameter(:ρ_A, 0.5, (1e-5, 1. - 1e-5), (1e-5, 1. - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_A: AR(1) coefficient in the bond-spread process.",
                   tex_label = "\\rho_A")


     # Exogenous processes - standard deviations
    ## all rater close to 0.0003 for BBL, so may change to 0 later
    m <= parameter(:σ_A, 0.00033388842631140714, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   InverseGamma(ig_pars(0.001,0.02.^2)...), fixed = false, # Note second tuple is parameterization for Exponential transform
                   description = "σ_A: standard dev. of the bond-spread process.",
                   tex_label = "\\sigma_{A}")

    ## Old Value 0.9 rather than 0.5
    m <= parameter(:ρ_Z, 0.5, (1e-5, 1. - 1e-5), (1e-5, 1. - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_Z: AR(1) coefficient in the technology process.",
                   tex_label = "\\rho_Z")


     m <= parameter(:σ_Z,  0.00033388842631140714, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   InverseGamma(ig_pars(0.001,0.02.^2)...), fixed = false,
                   description = "σ_Z: standard dev. of the process describing the " *
                   "stationary component of productivity.", tex_label = "\\sigma_Z")

    ## Old Value 0.9 rather than 0.5
    m <= parameter(:ρ_Ψ, 0.5, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_Ψ: AR(1) coefficient in marginal efficiency of investment (MEI) process.",
                   tex_label = "\\rho_{\\Psi}")



     m <= parameter(:σ_Ψ, 0.00033388842631140714, (0., 5.), (0., 5.), ModelConstructors.Exponential(), InverseGamma(ig_pars(0.001,0.02.^2)...), fixed = false,description = "σ_Ψ: standard dev. of the exogenous marginal efficiency" *
                   " of investment shock process.", tex_label = "\\sigma_{\\Psi}")


    ## Old Value 0.9 rather than 0.5
    m <= parameter(:ρ_μ_p, 0.5, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_μ_p: AR(1) coefficient in the price mark-up shock process.",
                   tex_label = "\\rho_{\\mu_p}")



 m <= parameter(:σ_μ_p, 0.00033388842631140714, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   InverseGamma(ig_pars(0.001,0.02.^2)...), fixed = false,
                   description = "σ_μ_p: standard dev. of the price mark-up shock process",
                   tex_label = "\\sigma_{\\mu_p}")
    ## Old Value 0.9 rather than 0.5
    m <= parameter(:ρ_μ_w, 0.5, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_μ_w: AR(1) coefficient in the wage mark-up shock process.",
                   tex_label = "\\rho_{\\mu_w}")


   m <= parameter(:σ_μ_w, 0.00033388842631140714, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                  InverseGamma(ig_pars(0.001,0.02.^2)...), fixed = false,
                   description = "σ_μ_w: : standard dev. of the wage mark-up shock process",
                   tex_label = "\\sigma_{\\mu_w}")



    ## Old Value 0.84 rather than 0.878
    m <= parameter(:ρ_S, 0.8777777777777779, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.7, 0.2), fixed = false,
                   description = "ρ_S: AR(1) coefficient in the idiosyncratic income risk process.",
                   tex_label = "\\rho_S")



   ## In the Seven Variable Version, this needs to be fixed at 0 but you let mode get assigned and then fix as of now
if get_setting(m,:seven_var_bbl)
 m <= parameter(:σ_S, 0.0, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   GammaAlt(0.65, 0.3), fixed = false,
                   description = "σ_S: standard dev. of the idiosyncratic income risk shock process",
                   tex_label = "\\sigma_{S}")
else
 m <= parameter(:σ_S, 0.5115384615384616, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   GammaAlt(0.65, 0.3), fixed = false,
                   description = "σ_S: standard dev. of the idiosyncratic income risk shock process",
                   tex_label = "\\sigma_{S}")
end
#=
    m <= parameter(:Σ_n, 0.0, (-1e3, 1e3), (-1e3, 1e3), ModelConstructors.SquareRoot(),
                   Normal(0., 100.), fixed = false,
                   description = "Σ_n: reaction of income risk to employment status",
                   tex_label = "\\Sigma_{n}")
=#
     m <= parameter(:Σ_n, 0.0, (-1000.0, 1000.0), (-1000.0, 1000.0), ModelConstructors.SquareRoot(),
                   Normal(0., 100.), fixed = true,
                   description = "Σ_n: reaction of income risk to employment status",
                   tex_label = "\\Sigma_{n}")
    #m[:Σ_n] = 0.0



 # Monetary policy
    ## Old Value 0.9 rather than 0.5
    m <= parameter(:ρ_R , 0.5, (1e-5, 0.999), (1e-5, 0.999), SquareRoot(),
                   BetaAlt(0.5, 0.20), fixed = false,
                   description = "ρ: The degree of inertia in the monetary policy rule.",
                   tex_label="\\rho_R")



    m <= parameter(:σ_R, 0.00033388842631140714, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                 InverseGamma(ig_pars(0.001,0.02.^2)...), fixed = false,
                   description = "σ_R_ϵ: standard dev. of the monetary policy shock process",
                   tex_label = "\\sigma_{R, \\epsilon}")




    ## Old Value 2. rather than 1.7
    m <= parameter(:θ_π, 1.7, (1., 10.), (1e-5, 10.0), ModelConstructors.Exponential(),
                   Normal(1.7, 0.3), fixed = false, # Note second tuple is parameterization for Exponential transform
                   description = "ψ1: Weight on inflation gap in monetary policy rule.",
                   tex_label = "\\theta_{\\pi}")
    m <= parameter(:θ_Y, 0.125, (-5., 5.), (-5., 5.), SquareRoot(),
                   Normal(0.125, 0.05), fixed = false,
                   description = "ψy: Weight on output gap in monetary policy rule",
                   tex_label = "\\theta_y")

    # Fiscal policy
    ## Old Value 0.2 rather 0.0438
    m <= parameter(:γ_B, 0.04375000000000002, (0., 5.), (0., 5.), SquareRoot(),
                   GammaAlt(0.1, 0.075), fixed = false,
                   description = "γ_B: Reaction of deficit to debt",
                   tex_label = "\\gamma_B")
    ## Old Value -0.1 rather than 0.0, check mean of Normal Dist with BBL
    m <= parameter(:γ_π, 0.0, (-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0.1, 1.), fixed = false,
                   description = "γ_π: Reaction of deficit to inflation",
                   tex_label = "\\gamma_{\\pi}")
    ## Old Value -1. rather than 0.0
    m <= parameter(:γ_Y, 0.0, (-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0.1, 1.), fixed = false,
                   description = "γ_Y: Reaction of deficit to output",
                   tex_label = "\\gamma_{Y}")



    ## Old Value 0.98 rather than 0.5
    m <= parameter(:ρ_G, 0.5, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_G: AR(1) coefficient in the government structural deficit process.",
                   tex_label = "\\rho_G")


     m <= parameter(:σ_G, 0.00033388842631140714, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   InverseGamma(ig_pars(0.001,0.02.^2)...), fixed = false,
                   description = "σ_G: standard dev. of the structural deficit shock process",
                   tex_label = "\\sigma_{G}")
 m <= parameter(:ρ_τ, 0.5, (1e-5, 1. - 1e-5), (1e-5, 1. - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_τ: Persistence in tax level",
                   tex_label = "\\rho_{\\tau}")


  m <= parameter(:γ_B_τ, 0.0, (-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = false,
                   description = "γ_B_τ: Reaction of tax level to debt",
                   tex_label = "\\gamma_{B, \\tau}")
    m <= parameter(:γ_Y_τ, 0.0, (-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = false,
                   description = "γ_Y_τ: Reaction of tax level to output",
                   tex_label = "\\gamma_{Y, \\tau}")

m <= parameter(:ρ_P, 0.5, (1e-5, 1. - 1e-5), (1e-5, 1. - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_P: Persistence in tax level",
                   tex_label = "\\rho_{P}")
 ## In the Seven Variable Version, this needs to be Fixed at 0
if get_setting(m,:seven_var_bbl)
  m <= parameter(:σ_P, 0.0, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   InverseGamma(ig_pars(0.001,0.02.^2)...), fixed = false,
                   description = "σ_P: standard dev. of the tax progressivity shock process",
                   tex_label = "\\sigma_{P}")
else
    m <= parameter(:σ_P, 0.00033388842631140714, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   InverseGamma(ig_pars(0.001,0.02.^2)...), fixed = false,
                   description = "σ_P: standard dev. of the tax progressivity shock process",
                   tex_label = "\\sigma_{P}")
end

  m <= parameter(:γ_B_P, 0., (-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = true,
                   description = "γ_B_P: Reaction of tax level to debt",
                   tex_label = "\\gamma_{B, P}")
    m <= parameter(:γ_Y_P, 0., (-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = true,
                   description = "γ_Y_P: Reaction of tax level to output",
                   tex_label = "\\gamma_{Y, P}")


    # Retained earnings
#=
    m <= parameter(:ω_F, 0.1, (0., 1.), (0., 1.), SquareRoot(),
                   Uniform(0., 1.), fixed = false,
                   description = "fraction of retained earnings (profits) that is disbursed to HH",
                   tex_label = "\\omega_F")
    m <= parameter(:ω_U, 0.1, (0., 1.), (0., 1.), SquareRoot(),
                   Uniform(0., 1.), fixed = false,
                   description = "fraction of retained earnings (wages) that is disbursed to HH",
                   tex_label = "\\omega_U")
=#









    # Auxiliary exogenous processes - autocorrelations (fixed to a very small number in baseline specification)
    m <= parameter(:ρ_R_sh, 1e-8, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = true,
                   description = "ρ_R_ϵ: AR(1) coefficient in the monetary policy shock process.",
                   tex_label = "\\rho_{R, \\epsilon}")
    m <= parameter(:ρ_P_sh, 1e-8, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = true,
                   description = "ρ_P_ϵ: AR(1) coefficient in the tax progressivity shock process.",
                   tex_label = "\\rho_{P, \\epsilon}")
    m <= parameter(:ρ_S_sh, 1e-8, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = true,
                   description = "ρ_S_ϵ: AR(1) coefficient in shock process of the shock in the idiosyncatic income shock process.",
                   tex_label = "\\rho_{S, \\epsilon}")


    ## In the Seven Variable Version, these variables should not be changing but you first unfix as of now and then  fix post mode assignment
if get_setting(m,:seven_var_bbl)
  m <= parameter(:e_W90_share, 0.0, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                 InverseGamma(ig_pars(0.0005,0.001.^2)...), fixed = false,
                   description = "σ_W90_share: standard dev. of measurement error for 90th percentile of wealth distribution",
                   tex_label = "\\sigma_{W^{(90)}}")
    m <= parameter(:e_I90_share, 0.0, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                  InverseGamma(ig_pars(0.0005,0.001.^2)...), fixed = false,
                   description = "σ_I90_share: standard dev. of measurement error for 90th percentile of income distribution",
                   tex_label = "\\sigma_{I^{(90)}}")
    m <= parameter(:e_τ_prog, 0.0, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                    InverseGamma(ig_pars(0.0005,0.001.^2)...), fixed = false,
                   description = "σ_P_me: standard dev. of measurement error for tax progressivity",
                   tex_label = "\\sigma_{P, me}")
    m <= parameter(:e_σ, 0.0, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   InverseGamma(ig_pars(0.05,0.01.^2)...), fixed = false,
                   description = "σ_S_me: standard dev. of measurement error for idiosyncratic income risk",
                   tex_label = "\\sigma_{S, me}")
else
    # Measurement error
    m <= parameter(:e_W90_share, sqrt(3.6982248520710056e-8), (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                 InverseGamma(ig_pars(0.0005,0.001.^2)...), fixed = false,
                   description = "σ_W90_share: standard dev. of measurement error for 90th percentile of wealth distribution",
                   tex_label = "\\sigma_{W^{(90)}}")
    m <= parameter(:e_I90_share,  sqrt(3.6982248520710056e-8), (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                  InverseGamma(ig_pars(0.0005,0.001.^2)...), fixed = false,
                   description = "σ_I90_share: standard dev. of measurement error for 90th percentile of income distribution",
                   tex_label = "\\sigma_{I^{(90)}}")
    m <= parameter(:e_τ_prog,  sqrt(3.6982248520710056e-8), (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                    InverseGamma(ig_pars(0.0005,0.001.^2)...), fixed = false,
                   description = "σ_P_me: standard dev. of measurement error for tax progressivity",
                   tex_label = "\\sigma_{P, me}")
    m <= parameter(:e_σ, sqrt(0.0021556122448979594), (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   InverseGamma(ig_pars(0.05,0.01.^2)...), fixed = false,
                   description = "σ_S_me: standard dev. of measurement error for idiosyncratic income risk",
                   tex_label = "\\sigma_{S, me}")
end



    # Steady-state parameters

    # Aggregate scalars (just initialized here, these will be populated by the steadystate!)
    m <= SteadyStateParameter(:K_star, NaN, description = "Capital stock (steady-state)",
                              tex_label = "K_*")
    m <= SteadyStateParameter(:N_star, NaN, description = "Labor supply (steady-state)",
                              tex_label = "N_*")
    m <= SteadyStateParameter(:Y_star, NaN, description = "Output (steady-state)",
                              tex_label = "Y_*")
    m <= SteadyStateParameter(:G_star, NaN, description = "Government spending (steady-state)",
                              tex_label = "G_*")
    m <= SteadyStateParameter(:w_star, NaN, description = "Wage (steady-state)",
                              tex_label = "w_*")
    m <= SteadyStateParameter(:T_star, NaN, description = "Tax revenue (steady-state)",
                              tex_label = "T_*")
    m <= SteadyStateParameter(:I_star, NaN, description = "Investment (steady-state)",
                              tex_label = "I_*")
    m <= SteadyStateParameter(:B_star, NaN, description = "Bond supply (steady-state)",
                              tex_label = "B_*")
    m <= SteadyStateParameter(:avg_tax_rate_star, NaN, description = "Average tax rate (steady-state)",
                              tex_label = "avgtaxrate_*")

    # Aggregate scalars computed after solving steady state
    m <= SteadyStateParameter(:A_star, NaN, description = "Bond spread (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Z_star, NaN, description = "TFP (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Ψ_star, NaN, description = "MEI (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:RB_star, NaN, description = "MEI (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:μ_p_star, NaN, description = "Price mark-up (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:μ_w_star, NaN, description = "Wage mark-up (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:τ_prog_star, NaN, description = "Tax progressivity (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:τ_level_star, NaN, description = "Tax level (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:σ_star, NaN, description = "Idiosyncratic risk (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:τ_prog_obs_star, NaN, description = "Observed tax progressivity (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:G_sh_star, NaN, description = "Government spending shock (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:R_sh_star, NaN, description = "MP shock (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:P_sh_star, NaN, description = "Progressivity shock (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:S_sh_star, NaN, description = "Idiosyncratic risk shock (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:rk_star, NaN, description = "Rental rate on capital (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:LP_star, NaN, description = "Liquidity premium (ex-post) (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:LP_XA_star, NaN, description = "Liquidity premium (ex-ante) (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:π_star, NaN, description = "Inflation (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:π_w_star, NaN, description = "Wage Inflation (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:BD_star, NaN, description = "Debt (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:C_star, NaN, description = "Consumption (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:q_star, NaN, description = "Capital price (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:mc_star, NaN, description = "Marginal cost (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:mc_w_star, NaN, description = "Wage marginal cost (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:mc_w_w_star, NaN, description = "Wage marginal cost (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:u_star, NaN, description = "Capital utilization rate (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:profits_star, NaN, description = "Profits (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:union_profits_star, NaN, description = "Union Profits (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:BY_star, NaN, description = "Bond to output ratio (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:TY_star, NaN, description = "Tax revenue to output ratio (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:T_l1_star, NaN, description = "Tax revenue first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Y_l1_star, NaN, description = "Output first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:B_l1_star, NaN, description = "Bond supply first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:G_l1_star, NaN, description = "Government spending first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:I_l1_star, NaN, description = "Investment first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:w_l1_star, NaN, description = "Wages first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:q_l1_star, NaN, description = "Capital price first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:C_l1_star, NaN, description = "Consumption first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:avg_tax_rate_l1_star, NaN, description = "Average tax rate first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:τ_prog_l1_star, NaN, description = "Tax progressivity first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Ygrowth_star, NaN, description = "Output growth (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Bgrowth_star, NaN, description = "Bond supply growth (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Igrowth_star, NaN, description = "Investment growth (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:wgrowth_star, NaN, description = "Wages growth (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Cgrowth_star, NaN, description = "Consumption growth (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Tgrowth_star, NaN, description = "Tax revenue growth (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Ht_star, NaN, description = "Ht (steady-state)", tex_label = "")
   #m <= SteadyStateParameter(:retained_star, NaN,
                              #description = "Retained earnings of the monopolisticaly competitive intermediate" *
                             # " firm sector, shifted by 1.0 (steady-state)", tex_label = "")
   # m <= SteadyStateParameter(:firm_profits_star, NaN, description = "Firm profits (steady-state)", tex_label = "")
   # m <= SteadyStateParameter(:union_retained_star, NaN, description = "Retained earnings of the monopolistic union" *
                             # " sector, shifted by 1.0 (steady-state)", tex_label = "")
   # m <= SteadyStateParameter(:union_firm_profits_star, NaN, description = "Union profits (steady-state)", tex_label = "")
   # m <= SteadyStateParameter(:tot_retained_Y_star, NaN, description = "Exponential of the retained earnings to gdp ratio" *
                              #" (equal to zero) (steady-state)", tex_label = "")

    # Scalar summary statistics about the idiosyncratic states (e.g. inequality measures)
    # TODO: add descriptions to these parameters
    m <= SteadyStateParameter(:share_borrower_star, NaN)
    m <= SteadyStateParameter(:Gini_wealth_star, NaN)
    m <= SteadyStateParameter(:W90_share_star, NaN)
    m <= SteadyStateParameter(:I90_share_star, NaN)
    m <= SteadyStateParameter(:I90_share_net_star, NaN)
    m <= SteadyStateParameter(:Gini_income_star, NaN)
    m <= SteadyStateParameter(:P90_minus_P10_income_star, NaN)
    m <= SteadyStateParameter(:sd_log_y_star, NaN)
    m <= SteadyStateParameter(:Gini_X_star, NaN)
    m <= SteadyStateParameter(:sd_log_X_star, NaN)
    m <= SteadyStateParameter(:P90_minus_P10_C_star, NaN)
    m <= SteadyStateParameter(:Gini_C_star, NaN)
    m <= SteadyStateParameter(:sd_log_C_star, NaN)
    m <= SteadyStateParameter(:P10_C_star, NaN)
    m <= SteadyStateParameter(:P50_C_star, NaN)
    m <= SteadyStateParameter(:P90_C_star, NaN)

    # Steady state grids for functional/distributional variables
    total_idio_states = get_setting(m, :nm) * get_setting(m, :nk) * get_setting(m, :ny)
    m <= SteadyStateParameterGrid(:distr_star, Array{Float64, 3}(undef, 0, 0, 0), # populated by init_grids! to ensure it's consistent
                                  description = "Distribution over idiosyncratic states (steady-state)", tex_label = "D_*")
    m <= SteadyStateParameterGrid(:marginal_pdf_m_star, Vector{Float64}(undef, 0), # populated later, just need to have the right typing
                                  description = "Marginal PDF of liquid bonds (steady-state)", tex_label = "D_{m, *}")
    m <= SteadyStateParameterGrid(:marginal_pdf_k_star, Vector{Float64}(undef, 0),
                                  description = "Marginal PDF of illiquid capital (steady-state)", tex_label = "D_{k, *}")
    m <= SteadyStateParameterGrid(:marginal_pdf_y_star, Vector{Float64}(undef, 0),
                                  description = "Marginal PDF of income (steady-state)", tex_label = "D_{y, *}")
    m <= SteadyStateParameterGrid(:marginal_cdf_m_star, Vector{Float64}(undef, 0), # populated later, just need to have the right typing
                                  description = "Marginal CDF of liquid bonds (steady-state)", tex_label = "D_{m, *}")
    m <= SteadyStateParameterGrid(:marginal_cdf_k_star, Vector{Float64}(undef, 0),
                                  description = "Marginal CDF of illiquid capital (steady-state)", tex_label = "D_{k, *}")
    m <= SteadyStateParameterGrid(:marginal_cdf_y_star, Vector{Float64}(undef, 0),
                                  description = "Marginal CDF of income (steady-state)", tex_label = "D_{y, *}")
    m <= SteadyStateParameterGrid(:Vm_star, Array{Float64, 3}(undef, 0, 0, 0),
                                  description = "Marginal value of liquid bonds (steady-state)", tex_label = "V_{m, *}")
    m <= SteadyStateParameterGrid(:Vk_star, Array{Float64, 3}(undef, 0, 0, 0),
                                  description = "Marginal value of illiquid capital (steady-state)", tex_label = "V_{k, *}")

    #May need this to replicate BBL's Fsys
    m <= SteadyStateParameterGrid(:copula_marginal_m, Vector{Float64}(undef, 0), # populated later, just need to have the right typing
                                  description = "Marginal PDF from Copula of liquid bonds (steady-state)", tex_label = "D_{m, *}")
    m <= SteadyStateParameterGrid(:copula_marginal_k, Vector{Float64}(undef, 0), # populated later, just need to have the right typing
                                  description = "Marginal PDF from Copula of illiquid bonds (steady-state)", tex_label = "D_{m, *}")
    m <= SteadyStateParameterGrid(:copula_marginal_y, Vector{Float64}(undef, 0), # populated later, just need to have the right typing
                                  description = "Marginal PDF from Copula of income  (steady-state)", tex_label = "D_{m, *}")


    # Steady state grids for reduction-related variables (indices for perturbation kept elsewhere) # TODO: document where compression indices are
    m <= SteadyStateParameterGrid(:dct_Vm_star, Vector{Float64}(undef, 0),
                                  description = "DCT coefficients of the marginal value of liquid bonds (steady-state)",
                                  tex_label = "\\theta_{Vm, *}")
    m <= SteadyStateParameterGrid(:dct_Vk_star, Vector{Float64}(undef, 0),
                                  description = "DCT coefficients of the marginal value of illiquid capital (steady-state)",
                                  tex_label = "\\theta_{Vk, *}")
    m <= SteadyStateParameterGrid(:dct_copula_star, Vector{Float64}(undef, 0),
                                  description = "DCT coefficients of the copula for distribution " *
                                  "over idiosyncratic states (steady-state)",
                                  tex_label = "\\theta_{D, *}")



    # Jacobians to be updated
    m <= SteadyStateParameterGrid(:A, Matrix{Float64}(undef, 0, 0),
                                  description = "The A matrix computed by jacobian(m)")
    m <= SteadyStateParameterGrid(:B, Matrix{Float64}(undef, 0, 0),
                                  description = "The B matrix computed by jacobian(m)")


    # Should have created the P matrix over here, not in settings

#Permute to BBL space, assign in BBL space, and then permute back
    # Note 22, 21 are unused parameters can set 21, R =1.01 and 22 K = 40.0 have been appended to the end
    permute_to_bbl = [1,2,3,4,5,6,7,8,9,10,11,14,16,23,24,18,19,20,25,12,13,15,17,39,41,42,43,44,45,48,49,50,51,53,54,26,28,30,32,34,36,46,55,56,57,27,29,31,33,35,37,38,40,47,52,58,59,60,61,21,22]
    #permute_to_dsge = sortperm(permute_to_bbl)

   #= params_copy = deepcopy(m.parameters)
    for i = 1:length(m.parameters)
       m.parameters[permute_to_bbl[i]] =  params_copy[i]
    end
   =#
     if  get_setting(m,:load_bbl_posterior_mean)
        params = ModelConstructors.get_values(get_parameters(m))
        params = vcat(params[1:19],1.226, 0.164,0.097,0.110,0.788,2.457,0.115,0.294,-1.078,-0.877,0.401,1.454,2.842,params[33:35],0.943,0.997,0.969,0.899,0.859,params[41],0.994,params[43:45],0.00199,0.00582,0.02158,0.01660,0.05930,params[51:52],0.00257,0.00291,params[55:59],params[60:61])
        params_bbl = zeros(size(params))
        for i = 1:length(params)
             params_bbl[permute_to_bbl[i]] = params[i]
        end

        #params = Vector{Float64}(params)
        ModelConstructors.update!(m.parameters,params_bbl)
  end

    if get_setting(m,:load_bbl_posterior_mode)
       free_para_inds = ModelConstructors.get_free_para_inds(get_parameters(m))
       params = ModelConstructors.get_values(get_parameters(m))
       bbl_mode_values = [0.7636183703453220,0.21549013014460100,0.15371931872327300,0.1187171422957710,0.9853149172233280, 0.0013696783537974800, 0.9987810365685850,0.005798639782148340, 0.9317001063640540, 0.0317843407893977,0.853252644523679,0.0160578177040562,0.875053791042663,0.06135513285055830,0.6687499292961940,0.7261306024492550,0.7792653158271170,0.002847962960365710,2.8544867440905100,0.4006536886674540,0.11353926662043100,-0.32373999178247400,-0.0954842071588715,0.9916447523321050,0.003658930882071120,0.6500090075402100,0.0820409042668231,-0.1389790395351300,0.965342707446865,0.03520129992402920,0.024260417952456800,0.062444003121058300,0.00018730093536733500,0.04713042333874030]

       params[free_para_inds] = bbl_mode_values
       ModelConstructors.update!(m.parameters,params)
    end
    if get_setting(m,:smc_fix_at_mode)
       m[:δ_s].fixed = true

       m[:ϕ].fixed = true

    end

    if get_setting(m,:seven_var_bbl)
       params = ModelConstructors.get_values(get_parameters(m))
       #θ = parameters2namedtuple(m)
       #id = construct_prime_and_noprime_indices(m; only_aggregate = false)

       m[:σ_P] = 0
       params[52]=0.0
       m[:σ_P].fixed = true
       m[:σ_S] = 0
       params[37]=0.0
       m[:σ_S].fixed = true
       m[:e_W90_share] = 0
       params[58]=0.0
       m[:e_W90_share].fixed = true
       m[:e_I90_share] = 0
       params[59]=0.0
       m[:e_I90_share].fixed = true
       m[:e_τ_prog] = 0
       params[60]=0.0
       m[:e_τ_prog].fixed = true
       m[:e_σ] = 0
       params[61]=0.0
       m[:e_σ].fixed = true
       ModelConstructors.update!(m.parameters,params)



    end

end

function model_settings!(m::BayerBornLuetticke)

    ## Defaults and overrides of defaults
    default_settings!(m)

    # Likelihood method
    m <= Setting(:use_chand_recursion, false)

    # Anticipated shocks
    m <= Setting(:n_anticipated_shocks, 0, "Number of anticipated policy shocks")
    m <= Setting(:n_anticipated_shocks_padding, 0, "Padding for anticipated policy shocks")

    ## Numerical settings for steady state

    # Coarse grid settings
    ## Setting to skip coarse grid for time being
    m <= Setting(:skip_coarse_grid, true)

    m <= Setting(:coarse_ϵ,  1e-6, "Steady-state tolerance for coarse grid")
    m <= Setting(:coarse_ny, 4, "Number of idiosyncratic income states for coarse grid")
    m <= Setting(:coarse_nm, 10, "Number of liquid asset (bond) points for coarse grid")
    m <= Setting(:coarse_nk, 10, "Number of illiquid asset (capital) points for coarse grid")
    m <= Setting(:coarse_ymin, 0.5, "Minimum grid value for income states on coarse grid")
    m <= Setting(:coarse_ymax, 1.5, "Maximum grid value for income states on coarse grid")
    m <= Setting(:coarse_mmin, -6.6, "Minimum grid value for liquid assets (bond) on coarse grid")
## Old Value 1000 rather than 1750.
    m <= Setting(:coarse_mmax, 1750., "Maximum grid value for liquid assets (bond) on coarse grid")
    m <= Setting(:coarse_kmin, 0., "Minimum grid value for illiquid assets (capital) on coarse grid")
## Old Value 1750. rather than 2250.
    m <= Setting(:coarse_kmax, 2250., "Maximum grid value for illiquid assets (capital) on coarse grid")

    # Refined grid settings
    m <= Setting(:ϵ, 1e-11, "Steady-state tolerance for refined grid")
    m <= Setting(:ny, 11, "Number of idiosyncratic income states for refined grid")
    ## CHANGING FROM 40 to 50 TO MATCH BBL's SETTINGS
    m <= Setting(:nm, 50 , "Number of liquid asset (bond) points for refined grid")
    m <= Setting(:nk, 50 , "Number of illiquid asset (capital) points for refined grid")
    m <= Setting(:ymin, 0.5, "Minimum grid value for income states on refined grid")
    m <= Setting(:ymax, 1.5, "Maximum grid value for income states on refined grid")
    m <= Setting(:mmin, -6.6, "Minimum grid value for liquid assets (bond) on refined grid")
    ## Old Value 1000 rather than 1750.
    m <= Setting(:mmax, 1750., "Maximum grid value for liquid assets (bond) on refined grid")
    m <= Setting(:kmin, 0., "Minimum grid value for illiquid assets (capital) on refined grid")
    ## Old Value 1750 rater than 2250.0
    m <= Setting(:kmax, 2250., "Maximum grid value for illiquid assets (capital) on refined grid")

    #copula reduction settings (to match BBL code)
    m <= Setting(:reduc_copula, 30)
    m <= Setting(:nm_copula, 10)
    m <= Setting(:nk_copula, 10)
    m <= Setting(:ny_copula, 10)
    m <= Setting(:further_compress_critS, 1e-11) #critical value for eigenvalues for Value functions
    m <= Setting(:further_compress_critC, eps()) #critical value for eigenvalues for copula

#=
    m <= Setting(:coarse_ϵ,  1e-5, "Steady-state tolerance for coarse grid")
    m <= Setting(:coarse_ny, 4, "Number of idiosyncratic income states for coarse grid")
    m <= Setting(:coarse_nm, 10, "Number of liquid asset (bond) points for coarse grid")
    m <= Setting(:coarse_nk, 10, "Number of illiquid asset (capital) points for coarse grid")
    m <= Setting(:coarse_ymin, 0.5, "Minimum grid value for income states on coarse grid")
    m <= Setting(:coarse_ymax, 1.5, "Maximum grid value for income states on coarse grid")
    m <= Setting(:coarse_mmin, -6.6, "Minimum grid value for liquid assets (bond) on coarse grid")
    m <= Setting(:coarse_mmax, 1750., "Maximum grid value for liquid assets (bond) on coarse grid")
    m <= Setting(:coarse_kmin, 0., "Minimum grid value for illiquid assets (capital) on coarse grid")
    m <= Setting(:coarse_kmax, 2250., "Maximum grid value for illiquid assets (capital) on coarse grid")

    # Refined grid settings
    m <= Setting(:ϵ, 1e-10, "Steady-state tolerance for refined grid")
    m <= Setting(:ny, 22, "Number of idiosyncratic income states for refined grid")
    m <= Setting(:nm, 80, "Number of liquid asset (bond) points for refined grid")
    m <= Setting(:nk, 80, "Number of illiquid asset (capital) points for refined grid")
    m <= Setting(:ymin, 0.5, "Minimum grid value for income states on refined grid")
    m <= Setting(:ymax, 1.5, "Maximum grid value for income states on refined grid")
    m <= Setting(:mmin, -6.6, "Minimum grid value for liquid assets (bond) on refined grid")
    m <= Setting(:mmax, 1750., "Maximum grid value for liquid assets (bond) on refined grid")
    m <= Setting(:kmin, 0., "Minimum grid value for illiquid assets (capital) on refined grid")
    m <= Setting(:kmax, 2250., "Maximum grid value for illiquid assets (capital) on refined grid")

=#

    # Consumption policy iteration
    ## Old Value 1000 rather than 10000
    m <= Setting(:max_value_function_iters, 10000,
                 "Maximum number of fixed point iterations for the marginal value functions")

    # Kolmogorov forward equation
    m <= Setting(:kfe_method, :krylov, "Method for solving Kolmogorov forward equation")
    m <= Setting(:n_direct_transition_iters, 10_000,
                 "Number of iterations when approximating stationary distribution " *
                 "directly as a limit of the transition equation")

    # Interval endpoints for Brent's method on refined grid
    #m <= Setting(:brent_interval_endpoints, (0.95, 1.05), "Interval endpoints for Brent's method on refined grid as" *
     #          " multiples of the steady state capital guess")
    m <= Setting(:brent_interval_endpoints, (0.8, 1.2), "Interval endpoints for Brent's method on refined grid as" *
               " multiples of the steady state capital guess")

    # Reduction settings for the following reduction strategy:
    # (1) Keep DCT coefficients of value functions that explain some fraction of total "energy"
    # (2) Perturb marginals, which are mapped into perturbations of the
    #     actual distribution through the copula. However, the copula is fixed in this perturbation.
    # (3) Keep DCT coefficients of distribution over idiosyncratic states to approximate perturbations
    #     in the copula while keeping the marginals fixed.
    # (4) Remove even more basis functions
    m <= Setting(:dct_energy_loss, 1e-6, "Lost fraction of 'energy' in the DCT compression of 'value functions'")
    ## WILL ULTIMATELY NO LONGER USE THIS SETTING
    m <= Setting(:n_copula_dct_coefficients, 10, "Number of coefficients in the DCT compression of the " *
                 "distribution over idiosyncratic states to approximate a perturbation in the copula")

    m <= Setting(:remove_non_volatile_basis_functions, false, "Remove non-volatile basis functions for further compression")

    # Initialize storages for settings/objects related to reduction
    m <= Setting(:dct_compression_indices, Dict{Symbol, Vector{Int}}(), "DCT compression indices")
    m <= Setting(:copula, identity, "Steady-state copula")

    # Whether one wishes to re-compute the steady state
    m <= Setting(:compute_full_steadystate, true, "Flag to avoid re-computing the full steady-state.")
    m <= Setting(:compute_steadystate, true, "Avoid recomputing stedystate (both heterogeneous and aggregate) after first computation.")
    ## Numbers of indices
    #  We declare these settings here to initialize them. The numbers of indices
    #  will be updated during solution due to reduction steps by setup_indices!,
    #  which will also update the mappings from variable names to indices.

    # Update number of scalar states and jumps
    m <= Setting(:n_scalar_jumps, 16, "Number of scalar jumps")
    m <= Setting(:n_scalar_states, 16, "Number of scalar states")
    m <= Setting(:n_scalar_variables,  get_setting(m, :n_scalar_jumps) + get_setting(m, :n_scalar_states),
                 "Number of scalars (jumps and states)")

    m <= Setting(:n_backward_looking_states, 1, "Total number of states after reduction steps") # just initializing, will count later
    m <= Setting(:n_jumps, 1, "Total number of jumps after reduction steps")
    m <= Setting(:n_model_states, get_setting(m, :n_backward_looking_states) + get_setting(m, :n_jumps),
                 "Number of model states (predetermined states and jump variables)")

    # Number of states and jumps
    m <= Setting(:n_predetermined_variables, 0, "Number of predetermined variables after
                 removing extra degrees of freedom from the distribution states.
                 This setting is initialized at 0 as a default value because it will always be
                 overwritten once the Jacobian is calculated.")

    ## Linearization settings
    m <= Setting(:linearize_heterogeneous_block, true, false, "", "Boolean for whether the " *
                 "heterogeneous block of the Jacobians should be linearized")
    m <= Setting(:solution_method, :klein, false, "",
                 "Solution method for obtaining a reduced-form state space representation from equilibrium conditions")
    m <= Setting(:klein_inversion_method, :minimum_norm, false, "",
                 "Inversion method to obtain gx and hx during the Klein algorithm")

    ## Replication-related settings
    #m <= Setting(:replicate_original_output, false, "Use steady state and linearization functions that exactly " *
    #             "replicate output from the original implementation by Bayer, Born, and Luetticke.")
    #m <= Setting(:original_dataset, false, "Load original dataset used by Bayer, Born, and Luetticke for their paper.")

m <= Setting(:replicate_original_output, true, "Use steady state and linearization functions that exactly " *
                "replicate output from the original implementation by Bayer, Born, and Luetticke.")
    m <= Setting(:original_dataset, true, "Load original dataset used by Bayer, Born, and Luetticke for their paper.")

    m <= Setting(:load_bbl_posterior_mean, false, "Load posterior mean parameter values from Bayer, Born, and Luetticke")
     m <= Setting(:load_bbl_posterior_mode, true, "Load posterior mode parameter values from Bayer, Born, and Luetticke")
     m <= Setting(:smc_fix_at_mode,false, "Fix certain parameters that don't seem to travel far enough at mode")
     m <= Setting(:seven_var_bbl, true)
    ## Saving and loading steady state output and Jacobians
    m <= Setting(:save_steadystate, true)
    m <= Setting(:save_jacobian, true)
    if !ispath(rawpath(m, "estimate"))
        mkpath(rawpath(m, "estimate"))
    end
    m <= Setting(:steadystate_output_file, rawpath(m, "estimate", "steadystate.jld2"))
    m <= Setting(:jacobian_output_file, rawpath(m, "estimate", "jacobian.jld2"))

    ## Dates
    m <= Setting(:data_vintage, "210504")
    m <= Setting(:cond_vintage, "210504")
    m <= Setting(:data_id, 1793)
    m <= Setting(:cond_id, 1793)
    m <= Setting(:date_zlb_start, quartertodate("2009-Q1")) # ZLB measured using Wu and Xia (2016) shadow FFR, which starts in 2009:Q1
    m <= Setting(:date_mainsample_start, quartertodate("1954-Q4"))
    m <= Setting(:date_presample_start, quartertodate("1954-Q4"))
    m <= Setting(:date_forecast_start, quartertodate("2020-Q1"))
    m <= Setting(:date_conditional_end, quartertodate("2020-Q1"))

    ## Monetary Policy
    m <= Setting(:n_mon_anticipated_shocks_padding, 0) # anticipated shocks are not used
    m <= Setting(:monetary_policy_shock, :R_sh)

    ## Estimation and Forecasting settings
    # defaults for now
end

"""
```
setup_indices!(m::BayerBornLuetticke)
```
sets up the indices of model states (predetermined states and jumps) and
equilibrium conditions associated with states.

This function is called during the model's initialization and # TODO maybe note initialization
during reduction steps, which changes indices.
"""
function setup_indices!(m::BayerBornLuetticke)
    # Abbreviate some fields
    state_vars = m.state_variables
    jump_vars = m.jump_variables
    endo = m.endogenous_states # note that these are the model states, so they include predetermined states and jumps
    aggr_endo = m.aggregate_endogenous_states # note that these are the model states, so they include predetermined states and jumps
    eqconds = m.equilibrium_conditions
    aggr_eqconds = m.aggregate_equilibrium_conditions

    # Compute size of idiosyncratic state space
    nm, nk, ny = get_idiosyncratic_dims(m)

    ## Populate endo using "next-period" name (i.e., using ′) since
    #  it is easier to remove the ′ than to add it
    n_idio_states            = nm + nk + ny - 3 # subtract 3 for dof
    dof_to_remove            = 3
    endo[:marginal_pdf_m′_t] = 1:(nm - 1)
    endo[:marginal_pdf_k′_t] = (1 + nm - 1):(nm + nk - 2)
    endo[:marginal_pdf_y′_t] = (1 + nm + nk - 2):n_idio_states
    n_dct_copula             = length(get_setting(m,:dct_compression_indices)[:copula])
    n_distr_states           = n_idio_states + n_dct_copula
    ## COMMENTING OUT OLD APPROACH WITH FIXED n_dct to match BBL
    #n_distr_states           = n_idio_states + get_setting(m, :n_copula_dct_coefficients)
    endo[:copula′_t]         = (1 + n_idio_states):n_distr_states
    for (i, k) in enumerate(get_aggregate_state_variables(m))
        endo[k] = (n_distr_states + i):(n_distr_states + i)
        aggr_endo[k] = i
    end

    # Update n_states to be consistent with the number of
    # idiosyncratic states and jumps after reduction
    n_states      = first(endo[state_vars[end]])
    n_aggr_states = n_states - n_distr_states
    m            <= Setting(:n_backward_looking_states, n_states)

    # Now populate jump indices
    n_dct_Vm            = length(get_setting(m, :dct_compression_indices)[:Vm])
    n_dct_Vk            = length(get_setting(m, :dct_compression_indices)[:Vk])
    n_idio_jumps        = n_dct_Vm + n_dct_Vk
    n_states_idio_jumps = n_states + n_idio_jumps

    endo[:Vm′_t] = (n_states + 1):(n_states + n_dct_Vm)
    endo[:Vk′_t] = (n_states + n_dct_Vm + 1):(n_states + n_idio_jumps)
    for (i, k) in enumerate(get_aggregate_jump_variables(m))
        endo[k] = (n_states_idio_jumps + i):(n_states_idio_jumps + i)
        aggr_endo[k] = i + n_aggr_states
    end
    m <= Setting(:n_model_states, first(endo[jump_vars[end]]))
    @show get_setting(m, :n_model_states)
    m <= Setting(:n_jumps, get_setting(m, :n_model_states) - n_states)

    ## Populate equation indices

    # Function blocks which output a function
    eqconds[:eq_marginal_pdf_m] = endo[:marginal_pdf_m′_t] # note that since endo holds UnitRanges, this assignment results in a copy
    eqconds[:eq_marginal_pdf_k] = endo[:marginal_pdf_k′_t]
    eqconds[:eq_marginal_pdf_y] = endo[:marginal_pdf_y′_t]
    eqconds[:eq_copula]         = endo[:copula′_t]
#=    eqconds[:eq_marginal_value_bonds]   = (first(endo[:Vm′_t]) - n_aggr_states):(last(endo[:Vm′_t]) - n_aggr_states)
    eqconds[:eq_marginal_value_capital] = (first(endo[:Vk′_t]) - n_aggr_states):(last(endo[:Vk′_t]) - n_aggr_states)=#
    eqconds[:eq_marginal_value_bonds]   = endo[:Vm′_t]
    eqconds[:eq_marginal_value_capital] = endo[:Vk′_t]

    # Aggregate blocks
    aggr_eqn_names = [# Exogenous shocks
                      :eq_A, :eq_Z, :eq_Ψ, :eq_mp, :eq_μ_p, :eq_μ_w,
                      :eq_σ,

                      :eq_LY, :eq_LB, :eq_LT, :eq_LI, :eq_Lw,
                      :eq_Lq, :eq_LC, :eq_Lavg_tax_rate,
                      :eq_Lτ_prog,

                      :eq_G, :eq_P, :eq_R, :eq_S]
    for (i, name) in enumerate(aggr_eqn_names)
        eqconds[name] = (n_distr_states + i):(n_distr_states + i)
        aggr_eqconds[name] = i
    end

    n_aggr_states = length(aggr_eqn_names)

    for (i, name) in enumerate([# Endogenous model states (for the jumps)
                                :eq_Gini_C, :eq_Gini_X, :eq_I90_share, :eq_I90_share_net, :eq_W90_share, :eq_sd_log_y,
                                :eq_capital_return, :eq_wages_firms_pay, :eq_capital_market_clear,
                                :eq_deficit_rule, :eq_real_wage_inflation,
                                :eq_output, :eq_resource_constraint, :eq_tobins_q,
                                :eq_labor_supply, :eq_price_phillips_curve, :eq_wage_phillips_curve,
                                :eq_capital_util, :eq_Ht, :eq_avg_tax_rate,
                                :eq_tax_revenue, :eq_capital_accum,
                                :eq_bond_market_clear, :eq_debt_market_clear,
                                :eq_bond_output_ratio, :eq_tax_output_ratio,
                                :eq_received_wages, :eq_gov_budget_constraint,
                                :eq_tax_level, :eq_tax_progressivity,
                                :eq_Ygrowth, :eq_Bgrowth, :eq_Igrowth,
                                :eq_wgrowth, :eq_Cgrowth, :eq_Tgrowth,
                                :eq_expost_liquidity_premium,
                                :eq_exante_liquidity_premium,:eq_union_profits,:eq_profits_distr_to_hh])
        eqconds[name] = (n_states_idio_jumps + i):(n_states_idio_jumps + i)
        aggr_eqconds[name] = i + n_aggr_states
    end

    # Augmented states
    m <= Setting(:n_model_states_augmented, get_setting(m, :n_model_states))

#=    # Aggregate blocks
    n_distr_states_idio_jumps = last(eqconds[:eq_marginal_value_capital])
    for (i, name) in enumerate([# Endogenous model states (for the jumps)
                                :eq_mp, :eq_tax_progressivity, :eq_tax_level,
                                :eq_tax_revenue, :eq_avg_tax_rate,
                                :eq_deficit_rule, :eq_gov_budget_constraint,
                                :eq_price_phillips_curve, :eq_wage_phillips_curve,
                                :eq_real_wage_inflation, :eq_capital_util, :eq_capital_return,
                                :eq_received_wages, :eq_wages_firms_pay, :eq_union_firm_profits,
                                :eq_union_profits, :eq_union_retained,
                                :eq_firm_profits, :eq_profits_distr_to_hh, :eq_retained,
                                :eq_tobins_q, :eq_expost_liquidity_premium,
                                :eq_exante_liquidity_premium, :eq_capital_accum,
                                :eq_labor_supply, :eq_output, :eq_resource_constraint,

                                # Growth rates
                                :eq_Ygrowth, :eq_Tgrowth, :eq_Bgrowth,
                                :eq_Igrowth, :eq_wgrowth, :eq_Cgrowth,

                                # Lagged variables
                                :eq_LY, :eq_LB, :eq_LI, :eq_Lw, :eq_LT,
                                :eq_Lq, :eq_LC, :eq_Lavg_tax_rate,
                                :eq_Lτ_prog,

                                # Exogenous shocks
                                :eq_A, :eq_Z, :eq_Ψ, :eq_μ_p, :eq_μ_w,
                                :eq_σ, :eq_G, :eq_P, :eq_R, :eq_S,

                                # Blocks mapping functions to scalars
                                :eq_capital_market_clear,
                                :eq_debt_market_clear, :eq_bond_market_clear,
                                :eq_bond_output_ratio, :eq_tax_output_ratio,
                                :eq_retained_earnings_gdp_ratio, :eq_Ht,
                                :eq_GiniX, :eq_I90_share, :eq_I90_share_net,
                                :eq_W90_share, :eq_sd_log_y, :eq_GiniC])
        eqconds[name] = (n_distr_states_idio_jumps + i):(n_distr_states_idio_jumps + i)
        aggr_eqconds[name] = i
    end=#
end


function parameter_groupings(m::BayerBornLuetticke)
    all_keys = [:ξ, :γ, :β, :λ, :γ_scale, #preferences
                  :ρ_h, :σ_h, :ι, :ζ, #individual income processes
                  :α, :δ_0, # technology
                  :δ_s, :ϕ,
                  :μ_p, :κ_p, :μ_w, :κ_w,#nk phillips
                  :ψ, :τ_lev, :τ_prog,
                  :ρ_A, :σ_A, :ρ_Z,
                  :σ_Z, :ρ_Ψ, :σ_Ψ,
                  :ρ_μ_p, :σ_μ_p, :ρ_μ_w, :σ_μ_w,
                  :ρ_S, :σ_S, :ρ_R, :σ_R, :θ_π, :θ_Y,
                  :γ_B, :γ_π, :γ_Y, :ρ_G, :σ_G, :ρ_τ,
                 :γ_B_τ, :γ_Y_τ, :ρ_P, :σ_P, :γ_B_P, :γ_Y_P]

    all_keys = Vector[all_keys]
    all_params = map(keys -> [m[θ]::Parameter for θ in keys], all_keys)
    descriptions = ["Parameters"]

    groupings = OrderedDict{String, Vector{Parameter}}(zip(descriptions, all_params))

    return groupings
end
