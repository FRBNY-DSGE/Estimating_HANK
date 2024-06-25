"""
```
BondLabor{T} <: AbstractModel{T}
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

* `exogenous_shocks::OrderedDict{Symbol,Int}`: Maps each shock to a column in
  the measurement and equilibrium condition matrices.

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
  parameters from the original model spec (\"ss0\") are initialized
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
mutable struct BondLabor{T} <: AbstractModel{T}
    parameters::ParameterVector{T}                         # vector of all time-invariant model parameters
    steady_state::ParameterVector{T}                       # model steady-state values

    # Temporary to get it to work. Need to
    # figure out a more flexible way to define
    # "grids" that are not necessarily quadrature
    # grids within the model
    grids::OrderedDict{Symbol,Union{Grid, Vector}}

    keys::OrderedDict{Symbol,Int}                          # human-readable names for all the model
                                                           # parameters and steady-states

    state_variables::Vector{Symbol}                        # Vector of symbols of the state variables
    jump_variables::Vector{Symbol}                         # Vector of symbols of the jump variables
    normalized_model_states::Vector{Symbol}                # All of the distributional model
                                                           # state variables that need to be normalized
    endogenous_states_unnormalized::OrderedDict{Symbol,UnitRange} # Vector of unnormalized
                                                           # ranges of indices
    endogenous_states::OrderedDict{Symbol,UnitRange}       # Vector of ranges corresponding
                                                           # to normalized (post Klein solution) indices
    exogenous_shocks::OrderedDict{Symbol,Int}              #
    expected_shocks::OrderedDict{Symbol,Int}               #
    equilibrium_conditions::OrderedDict{Symbol,UnitRange}  #
    observables::OrderedDict{Symbol,Int}                   #

    spec::String                                           # Model specification number (eg "m990")
    subspec::String                                        # Model subspecification (eg "ss0")
    settings::Dict{Symbol,Setting}                         # Settings/flags for computation
    test_settings::Dict{Symbol,Setting}                    # Settings/flags for testing mode
    rng::MersenneTwister                                   # Random number generator
    testing::Bool                                          # Whether we are in testing mode or not

    observable_mappings::OrderedDict{Symbol, Observable}
end

description(m::BondLabor) = "BondLabor, $(m.subspec)"

"""
`init_model_indices!(m::BondLabor)`

Arguments:
`m:: BondLabor`: a model object

Description:
Initializes indices for all of `m`'s states, shocks, and equilibrium conditions.
"""
function init_model_indices!(m::BondLabor)
    # Endogenous states
    endogenous_states = collect([
    # These states corresp. to the following in the original notation
    #    MUP,   ZP,    ELLP,  RP
        :μ′_t, :z′_t, :l′_t, :R′_t])

    # Exogenous shocks
    exogenous_shocks = collect([:z_sh])

    # Equilibrium conditions
    equilibrium_conditions = collect([
        :eq_euler, :eq_kolmogorov_fwd, :eq_market_clearing, :eq_TFP])

    # Observables
    observables = keys(m.observable_mappings)

    ########################################################################################
    # Setting indices of endogenous_states and equilibrium conditions manually for now
    nx = get_setting(m, :nx)
    ns = get_setting(m, :ns)
    endo = m.endogenous_states_unnormalized
    eqconds = m.equilibrium_conditions

    # State variables
    endo[:μ′_t]  = 1:nx*ns
    endo[:z′_t]  = nx*ns+1:nx*ns+1

    # Jump variables
    endo[:l′_t]  = nx*ns+2:2*nx*ns+1
    endo[:R′_t]  = 2*nx*ns+2:2*nx*ns+2

    eqconds[:eq_euler]              = 1:nx*ns
    eqconds[:eq_kolmogorov_fwd]     = nx*ns+1:2*nx*ns
    eqconds[:eq_market_clearing]    = 2*nx*ns+1:2*nx*ns+1
    eqconds[:eq_TFP]                = 2*nx*ns+2:2*nx*ns+2
    ########################################################################################
    m.normalized_model_states = [:μ′_t]
    m.endogenous_states = deepcopy(endo)
    m.state_variables = m.endogenous_states.keys[get_setting(m, :state_indices)]
    m.jump_variables = m.endogenous_states.keys[get_setting(m, :jump_indices)]

    for (i,k) in enumerate(exogenous_shocks);            m.exogenous_shocks[k]            = i end
    for (i,k) in enumerate(observables);                 m.observables[k]                 = i end
end

function BondLabor(subspec::String="ss0";
                   custom_settings::Array{S} where S<:Setting = Array{Setting{Bool}}(undef,0),
                   testing = false)

    # Model-specific specifications
    spec               = "BondLabor"
    subspec            = subspec
    settings           = Dict{Symbol,Setting}()
    test_settings      = Dict{Symbol,Setting}()
    rng                = MersenneTwister(0)

    # initialize empty model
    m = BondLabor{Float64}(
            # model parameters and steady state values
            Vector{AbstractParameter{Float64}}(), Vector{Float64}(),
            # grids and keys
            OrderedDict{Symbol,Grid}(), OrderedDict{Symbol,Int}(),

            # normalized_model_states, state_inds, jump_inds
            Vector{Symbol}(), Vector{Symbol}(), Vector{Symbol}(),

            # model indices
            # endogenous states unnormalized, endogenous states normalized
            OrderedDict{Symbol,UnitRange}(), OrderedDict{Symbol,UnitRange}(),
            OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(),
            OrderedDict{Symbol,UnitRange}(), OrderedDict{Symbol,Int}(),

            spec,
            subspec,
            settings,
            test_settings,
            rng,
            testing,
            OrderedDict{Symbol,Observable}())

    # Set settings
    model_settings!(m)
    # default_test_settings!(m)
    for custom_setting in custom_settings
        m <= custom_setting
    end

    # Set observable transformations
    init_observable_mappings!(m)

    # Initialize parameters
    init_parameters!(m)

    # Initialize grids
    init_grids!(m)

    # Initialize model indices
    init_model_indices!(m)

    # Solve for the steady state
    steadystate!(m)

    # So that the indices of m.endogenous_states reflect the normalization
    normalize_model_state_indices!(m)

    return m
end

"""
```
init_parameters!(m::BondLabor)
```

Initializes the model's parameters, as well as empty values for the steady-state
parameters (in preparation for `steadystate!(m)` being called to initialize
those).
"""
function init_parameters!(m::BondLabor)
    # Initialize parameters
    m <= parameter(:R, 1.04, fixed = true,
                   description = "R: Steady-state gross real interest rate.", tex_label = "R")
    m <= parameter(:γ, 1.0, fixed = true,
                   description = "γ: CRRA Parameter.", tex_label = "\\gamma")
    m <= parameter(:ν, 1.0, fixed = true,
                   description = "Inverse Frisch elasticity of labor supply.", tex_label = "\\nu")
    m <= parameter(:abar, -0.5, fixed = true,
                   description = "Borrowing floor.", tex_label = "\\bar{a}")
    m <= parameter(:ρ_z, 0.95, (1e-5, 0.999), (1e-5, 0.999), SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                   description="ρ_z: AR(1) coefficient in the technology process.",
                   tex_label="\\rho_z")
    m <= parameter(:σ_z, sqrt(.007), (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
                   description="σ_z: The standard deviation of the process describing the stationary component of productivity.",
                   tex_label="\\sigma_{z}")
    m <= parameter(:μ_s, 0., fixed = true, description = "μ_s: Mu of log normal in income")
    m <= parameter(:σ_s, 0.1, fixed = true,
                   description = "σ_s: Sigma of log normal in income")
    m <= parameter(:e_y, 1e-3, fixed = true, description = "e_y: Measurement error on GDP",
                   tex_label = "e_y")

    # Setting steady-state parameters
    nx = get_setting(m, :nx)
    ns = get_setting(m, :ns)

    m <= SteadyStateParameterGrid(:lstar, fill(NaN, nx*ns), description = "Steady-state expected discounted
                                  marginal utility of consumption", tex_label = "l_*")
    m <= SteadyStateParameterGrid(:cstar, fill(NaN, nx*ns), description = "Steady-state consumption",
                                  tex_label = "c_*")
    m <= SteadyStateParameterGrid(:ηstar, fill(NaN, nx*ns), description = "Steady-state
                                  level of labor supply",
                                  tex_label = "\\eta_*")
    m <= SteadyStateParameterGrid(:μstar, fill(NaN, nx*ns), description = "Steady-state cross-sectional
                                  density of cash on hand", tex_label = "\\mu_*")

    # Figure out a better description for this...
    m <= SteadyStateParameterGrid(:χstar, fill(NaN, nx*ns), description = "Steady-state
                                  solution for constrained consumption and labor supply",
                                  tex_label = "\\chi_*")

    m <= SteadyStateParameter(:βstar, NaN, description = "Steady-state discount factor",
                              tex_label = "\\beta_*")
    # m <= SteadyStateParameter(:Lstar, NaN, description = "Steady-state labor", tex_label = "L_*")
    # m <= SteadyStateParameter(:Wstar, NaN, description = "Steady-state wages", tex_label = "W_*")
    # m <= SteadyStateParameterGrid(:KFstar, fill(NaN, (nw, nw)), description = "Steady-state Kolmogorov
                                  # Forward Equation", tex_label = "KF_*")
end

"""
```
init_grids!(m::BondLabor)
```
"""
function init_grids!(m::BondLabor)
    xscale  = get_setting(m, :xscale)
    xlo     = get_setting(m, :xlo)
    xhi     = get_setting(m, :xhi)
    nx      = get_setting(m, :nx)

    ns      = get_setting(m, :ns)
    λ       = get_setting(m, :λ)

    ehi     = get_setting(m, :ehi)

    grids = OrderedDict()

    # Cash on hand grid
    grids[:xgrid] = Grid(uniform_quadrature(xscale), xlo, xhi, nx, scale = xscale)

    # Skill grid
    lsgrid, sprob, sscale = tauchen86(m[:μ_s].value, m[:σ_s].value, ns, λ)
    swts = (sscale/ns)*ones(ns)
    sgrid = exp.(lsgrid) .+ ehi
    grids[:sgrid] = Grid(sgrid, swts, sscale)

    # Density of skill across skill grid
    grids[:ggrid] = sprob./swts

    # Total grid vectorized across both dimensions
    grids[:sgrid_total] = kron(sgrid, ones(nx))
    grids[:xgrid_total] = kron(ones(ns), grids[:xgrid].points)
    grids[:weights_total] = kron(swts, grids[:xgrid].weights)

    m.grids = grids
end

function model_settings!(m::BondLabor)
    default_settings!(m)

    # Defaults
    # Data settings for released and conditional data. Default behavior is to set vintage
    # of data to today's date.
    vint = Dates.format(now(), DSGE_DATE_FORMAT)
    m <= Setting(:data_vintage, vint, true, "vint", "Data vintage")

    saveroot = normpath(joinpath(dirname(@__FILE__), "../../../../","save"))
    datapath = normpath(joinpath(dirname(@__FILE__), "../../../../","save","input_data"))

    m <= Setting(:saveroot, saveroot, "Root of data directory structure")
    m <= Setting(:dataroot, datapath, "Input data directory path")

    # Solution method
    m <= Setting(:solution_method, :klein)

    # Anticipated shocks
    m <= Setting(:n_anticipated_shocks, 0,
                 "Number of anticipated policy shocks")
    # m <= Setting(:n_anticipated_shocks_padding, 20,
                 # "Padding for anticipated policy shocks")

    # Number of states and jumps
    m <= Setting(:normalize_distr_variables, true, "Whether or not to perform the
                 normalization of the distributional states in the Klein solution step")
    m <= Setting(:n_predetermined_variables, 0, "Number of predetermined variables after
                 removing the densities. Calculated with the Jacobian normalization.
                 This set to 0 is just the default setting, since it will always be
                 overwritten once the Jacobian is calculated.")

    m <= Setting(:state_indices, 1:2, "Which indices of m.endogenous_states correspond to
                 backward looking state variables")
    m <= Setting(:jump_indices, 3:4, "Which indices of m.endogenous_states correspond to jump
                 variables")

    # Mollifier setting parameters
    m <= Setting(:In, 0.443993816237631, "Normalizing constant for the mollifier")
    m <= Setting(:elo, 0.0, "Lower bound on stochastic consumption commitments")
    m <= Setting(:ehi, 1.0, "Upper bound on stochastic consumption commitments")

    # x: Cash on Hand Grid Setup
    m <= Setting(:nx, 75, "Cash on hand distribution grid points")
    m <= Setting(:xlo, -0.5 - 1.0, "Lower bound on cash on hand")
    m <= Setting(:xhi, 4.0, "Upper Bound on cash on hand")
    m <= Setting(:xscale, get_setting(m, :xhi) - get_setting(m, :xlo), "Size of the xgrid")

    # s: Skill Distribution/ "Units of effective labor" Grid Setup
    m <= Setting(:ns, 2, "Skill distribution grid points")
    m <= Setting(:λ, 3.0, "The λ parameter in the Tauchen distribution calculation")

    # Total grid x*s
    m <= Setting(:n, get_setting(m, :nx) * get_setting(m, :ns), "Total grid size, multiplying
                 across grid dimensions.")

    # Function-valued variables include distributional variables
    m <= Setting(:n_function_valued_backward_looking_states, 1, "Number of function-valued
                 backward looking state variables")
    m <= Setting(:n_backward_looking_distributional_vars, 1, "Number of state variables that are
                 distributional variables.")
    m <= Setting(:n_function_valued_jumps, 1, "Number of function-valued jump variables")
    m <= Setting(:n_jump_distributional_vars, 0, "Number of jump variables that are distributional
                 variables.")

    # Note, these settings assume normalization.
    # The n degrees of freedom removed depends on the distributions/dimensions
    # of heterogeneity that we have discretized over, in this case,
    # cash on hand and the skill distribution. In general the rule of
    # thumb is, remove one degree of freedom for the first endogenous distribution (cash on
    # hand), then one additional degree of freedom for each exogenous distribution (skill
    # distribution). Multiple endogenous distributions only permit removing a single degree
    # of freedom since it is then non-trivial to obtain the marginal distributions.
    m <= Setting(:n_degrees_of_freedom_removed, 2, "Number of degrees of freedom from the
                 distributional variables to remove.")
    n_dof_removed = get_setting(m, :n_degrees_of_freedom_removed)

    ####################################################
    # Calculating the number of backward-looking states
    ####################################################
    n_backward_looking_distr_vars = get_setting(m, :n_backward_looking_distributional_vars)
    m <= Setting(:backward_looking_states_normalization_factor,
                 n_dof_removed*n_backward_looking_distr_vars, "The number of dimensions removed from the
                 backward looking state variables for the normalization.")

    n = get_setting(m, :n)
    n_backward_looking_vars = length(get_setting(m, :state_indices))
    n_backward_looking_function_valued_vars = get_setting(m, :n_function_valued_backward_looking_states)
    n_backward_looking_scalar_vars = n_backward_looking_vars - n_backward_looking_function_valued_vars

    m <= Setting(:n_backward_looking_states, n*n_backward_looking_distr_vars +
                 n_backward_looking_scalar_vars - get_setting(m, :backward_looking_states_normalization_factor),
                 "Number of state variables, in the true sense (fully
                  backward looking) accounting for the discretization across the grid
                  of function-valued variables and the normalization of
                  distributional variables.")

    ##################################
    # Calculating the number of jumps
    ##################################
    n_jump_distr_vars = get_setting(m, :n_jump_distributional_vars)
    m <= Setting(:jumps_normalization_factor,
                 n_dof_removed*n_jump_distr_vars, "The number of dimensions removed from the
                 jump variables for the normalization.")

    n_jump_vars = length(get_setting(m, :jump_indices))
    n_jump_function_valued_vars = get_setting(m, :n_function_valued_jumps)
    n_jump_scalar_vars = n_jump_vars - n_jump_function_valued_vars

    m <= Setting(:n_jumps, n*n_jump_function_valued_vars +
                 n_jump_scalar_vars - get_setting(m, :jumps_normalization_factor),
                 "Number of jump variables (forward looking) accounting for
                  the discretization across the grid of function-valued variables and the
                  normalization of distributional variables.")

    m <= Setting(:n_model_states, get_setting(m, :n_backward_looking_states) + get_setting(m, :n_jumps),
                 "Number of 'states' in the state space model. Because backward and forward
                 looking variables need to be explicitly tracked for the Klein solution
                 method, we have n_states and n_jumps")
end
