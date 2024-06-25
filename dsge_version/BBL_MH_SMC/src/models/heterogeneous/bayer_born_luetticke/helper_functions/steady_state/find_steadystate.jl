"""
```
find_steadystate(m::BayerBornLuetticke; verbose::Symbol = :none,
                 skip_coarse_grid::Bool = false, parallel::Bool = false)
```
Find the stationary equilibrium capital stock.

### Keyword Arguments
- `verbose`: verbosity of print statements at 3 different levels `[:none, :low, :high]`
- `skip_coarse_grid`: skip using a coarse grid to get close to the steady-state capital stock

### Outputs
- `KSS::Float64`: steady-state capital stock
- `VmSS::Array{Float64,3}`, `VkSS::Array{Float64,3}`: marginal value functions
- `distrSS::Array{Float64,3}`: steady-state distribution of idiosyncratic states, computed by `Ksupply`
"""
function find_steadystate(m::BayerBornLuetticke{T}; verbose::Symbol = :none,
                          use_old_steadystate::Bool = false,
                          skip_coarse_grid::Bool = false, parallel::Bool = false)::Tuple{Float64,AbstractArray{Float64,3},AbstractArray{Float64,3},AbstractArray{Float64,3}} where {T <: Real}



    θ = parameters2namedtuple(m)
    max_value_function_iters = get_setting(m, :max_value_function_iters)
    n_direct_transition_iters = get_setting(m, :n_direct_transition_iters)
    kfe_method = get_setting(m, :kfe_method)
    #=if kfe_method == :slepc
        # @assert false "SLEPc currently is not a working method for solving the KFE"
        ## SlepcInitialize("-eps_nev 1")
        # SlepcInitialize("-eps_max_it 100 -eps_tol 1e-6 -eps_nev 1")
    end=#

    # -------------------------------------------------------------------------------
    ## STEP 1: Find the stationary equilibrium for coarse grid
    # -------------------------------------------------------------------------------
    #-------------------------------------------------------
    # Income Process and Income Grids
    #-------------------------------------------------------

    if skip_coarse_grid || use_old_steadystate
        KSS = exp(m[:K_star]) # use K_star as an initial guess
    else

        # Construct coarse grid based on information from settings
        init_grids!(m; coarse = true)

        if verbose in [:low, :high]
            println("Finding equilibrium capital stock for coarse income grid...")
        end

        # Capital stock guesses
        brent_Kmax = 1.75 * ((m[:δ_0] - .0025 + (1.0 .- m[:β]) / m[:β]) / m[:α])^(1.0 / (m[:α] - 1.0))
        brent_Kmin = 1.0  * ((m[:δ_0] - .0005 + (1.0 .- m[:β]) / m[:β]) / m[:α])^(0.5 / (m[:α] - 1.0))

       # brent_Kmax = 50.0
       # brent_Kmin = 20.0

        # a.) Define excess demand function with coarse = true

        # Initialize matrix which will be used when kfe_method == :direct
        # so that stationary distribution can be updated in-place
        init_distr_guess = get_untransformed_values(m[:distr_star])

        # Additional numerical settings
        ϵ = get_setting(m, :coarse_ϵ)
        nm, nk, ny = get_idiosyncratic_dims(m; coarse = true)

        # Initialize arrays to ensure efficient memory usage during EGM loop
        Vm_tmp = Array{T,3}(undef, nm, nk, ny)
        Vk_tmp = Array{T,3}(undef, nm, nk, ny)
        m_n_star = Array{T,3}(undef, nm, nk, ny)
        m_a_star = Array{T,3}(undef, nm, nk, ny)
        k_a_star = Array{T,3}(undef, nm, nk, ny)
        c_n_star = Array{T,3}(undef, nm, nk, ny)
        c_a_star = Array{T,3}(undef, nm, nk, ny)

        ## Uses coarse grid settings for Kdiff for excess demand function *
        d_coarse(  K, initial::Bool=true,
                   Vm_guess = Array{T,3}(undef, nm, nk, ny),
                   Vk_guess = Array{T,3}(undef, nm, nk, ny),
                   distr_guess = init_distr_guess
                   ) = Kdiff(K, m.grids, θ,
                             initial, Vm_guess, Vk_guess, distr_guess,
                             Vm_tmp, Vk_tmp, m_a_star, m_n_star, k_a_star, c_a_star, c_n_star;
                             verbose = verbose, coarse = true, parallel = parallel,
                             ϵ, max_value_function_iters, n_direct_transition_iters,
                             kfe_method, ny)

        # b.) Find equilibrium capital stock (multigrid on y,m,k)

        CustomBrent(d_coarse, brent_Kmin, brent_Kmax)
        KSS =  CustomBrent(d_coarse, brent_Kmin, brent_Kmax)[1]

        if verbose in [:low, :high]
            println("Capital stock is $(KSS)")
        end
    end

    # -------------------------------------------------------------------------------
    ## STEP 2: Find the stationary equilibrium for final grid
    # -------------------------------------------------------------------------------


    if verbose in [:low, :high]
        println("Finding equilibrium capital stock for refined income grid...")
    end
    init_grids!(m)

    # Find stationary equilibrium for refined economy
    # a.) Define excess demand function with coarse = false

    # Additional numerical settings
    ϵ_fine = get_setting(m, :ϵ)
    nm, nk, ny = get_idiosyncratic_dims(m; coarse = false)

    # Initialize arrays to ensure efficient memory usage during EGM loop
    Vm_tmp = Array{T,3}(undef, nm, nk, ny)
    Vk_tmp = Array{T,3}(undef, nm, nk, ny)
    m_n_star = Array{T,3}(undef, nm, nk, ny)
    m_a_star = Array{T,3}(undef, nm, nk, ny)
    k_a_star = Array{T,3}(undef, nm, nk, ny)
    c_n_star = Array{T,3}(undef, nm, nk, ny)
    c_a_star = Array{T,3}(undef, nm, nk, ny)

    # Initialize matrix which will be used when kfe_method == :direct
    # so that stationary distribution can be updated in-place
    init_distr_guess = get_untransformed_values(m[:distr_star])

    d = if use_old_steadystate
        init_Vm_guess = exp.(get_untransformed_values(m[:Vm_star]))
        init_Vk_guess = exp.(get_untransformed_values(m[:Vk_star]))

        d1(  K, initial::Bool=false,
             Vm_guess = init_Vm_guess,
             Vk_guess = init_Vk_guess,
             distr_guess = init_distr_guess
             ) = Kdiff(K, m.grids, θ, initial, Vm_guess, Vk_guess, distr_guess,
                       Vm_tmp, Vk_tmp, m_a_star, m_n_star, k_a_star, c_a_star, c_n_star;
                       verbose = verbose, coarse = false, parallel = parallel,
                       ϵ = ϵ_fine, max_value_function_iters, n_direct_transition_iters,
                       kfe_method, ny = ny)
    else
        d2(  K, initial::Bool=true,
             Vm_guess = Array{T,3}(undef, nm, nk, ny),
             Vk_guess = Array{T,3}(undef, nm, nk, ny),
             distr_guess = init_distr_guess
             ) = Kdiff(K, m.grids, θ, initial, Vm_guess, Vk_guess, distr_guess,
                       Vm_tmp, Vk_tmp, m_a_star, m_n_star, k_a_star, c_a_star, c_n_star;
                       verbose = verbose, coarse = false, parallel = parallel,
                       ϵ = ϵ_fine, max_value_function_iters, n_direct_transition_iters,
                       kfe_method, ny = ny)
    end


    # b.) Find equilibrium capital stock (multigrid on (m, k, y))
    lower_prop, upper_prop = get_setting(m, :brent_interval_endpoints)
    # TODO: update CustomBrent to check if f(a) > 0 and to return an error otherwise
    # b/c in this case, Kdiff is positive when capital guess is low.
    # Then we should add a loop and an algorithm to lower the lower bound.
    # Similar steps should be taken for the upper loop


    BrentOut = CustomBrent(d, KSS*lower_prop, KSS*upper_prop; tol = get_setting(m, :ϵ))

    KSS      = BrentOut[1]
    VmSS     = BrentOut[3][2]
    VkSS     = BrentOut[3][3]
    distrSS  = BrentOut[3][4]
    if verbose in [:low, :high]
        println("Capital stock is $(KSS)")
    end

    #=if kfe_method == :slepc
        SlepcFinalize()
    end=#

    return KSS, VmSS, VkSS, distrSS



end
