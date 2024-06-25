function original_find_steadystate(m::BayerBornLuetticke{T}; verbose::Symbol = :none,
                                   skip_coarse_grid::Bool = false) where {T <: Real}

    # BLAS.set_num_threads(Threads.nthreads()) # this should be set outside of the function

    # TODO: add option to skip the coarse grid b/c already found the steady state and using it as a guess
    # -------------------------------------------------------------------------------
    ## STEP 1: Find the stationary equilibrium for coarse grid
    # -------------------------------------------------------------------------------
    #-------------------------------------------------------
    # Income Process and Income Grids
    #-------------------------------------------------------

    # Construct coarse grid based on information from settings
    original_init_grids!(m; coarse = true)

#=
    Π = m.grids[:Π]
    grid_y = m.grids[:y_grid]
    bounds_y = m.grids[:y_bin_bounds]
    grid_k = m.grids[:k_grid]
    grid_m = m.grids[:m_grid]
    mesh_y = m.grids[:y_ndgrid]
    mesh_m = m.grids[:m_ndgrid]
    mesh_k = m.grids[:k_ndgrid]
    H = m.grids[:H]
    HW = m.grids[:HW]
    @save "save5_dsge.jld2" Π grid_y bounds_y grid_k grid_m mesh_y mesh_m mesh_k H HW
=#

    if verbose in [:low, :high]
        println("Finding equilibrium capital stock for coarse income grid...")
    end

    # Capital stock guesses
    #brent_Kmax = 1.75 * ((m[:δ_0] - 0.0025 + (1.0 - m[:β]) / m[:β]) / m[:α])^(1.0 / (m[:α] - 1.0))
    #brent_Kmin = 1.0  * ((m[:δ_0] - 0.0005 + (1.0 - m[:β]) / m[:β]) / m[:α])^(0.5 / (m[:α] - 1.0))
    #brent_Kmin = 22.487644
    #brent_Kmax = 50.61927


    rmin = 0.0
    rmax = (1.0 .- m[:β])./m[:β] - .0025


    @inline function capital_intensity(r)
        out = ((r + m[:δ_0]) ./ m[:α] .* m[:μ_p])^(1.0 ./ (m[:α] .- 1))
        return out
    end


    @inline function labor_supply(w)
        out = ((1.0 .- m[:τ_prog]) .* m[:τ_lev]) ^ (1.0 ./ (m[:γ] .+ m[:τ_prog])) .*
        w^((1.0 .- m[:τ_prog]) ./ (m[:γ] .+ m[:τ_prog]))
        return out
    end

    brent_Kmax = capital_intensity(rmin) .* labor_supply(_bbl_wage(capital_intensity(rmin), 1.0 ./ m[:μ_p], 1.0, m[:α])./ m[:μ_w])
    brent_Kmin = capital_intensity(rmax) .* labor_supply(_bbl_wage(capital_intensity(rmax), 1.0 ./ m[:μ_p], 1.0, m[:α])./ m[:μ_w])



    @show brent_Kmax
    @show brent_Kmin

    # a.) Define excess demand function with coarse = true
    init_distr_guess = get_untransformed_values(m[:distr_star])



@inline function d_coarse(  K,
               initial::Bool=true,
               Vm_guess = zeros(1,1,1),
               Vk_guess = zeros(1,1,1),
               distr_guess = init_distr_guess
               )
        out = original_Kdiff(K, m, initial, Vm_guess, Vk_guess, distr_guess;
                             verbose = verbose, coarse = true)
    return out
end



    # b.) Find equilibrium capital stock (multigrid on y,m,k)
    KSS = CustomBrent(d_coarse, brent_Kmin, brent_Kmax)[1]

        @show  brent_Kmin
        @show brent_Kmax
        @show KSS

    if verbose in [:low, :high]
        println("Capital stock is $(KSS)")
    end
    #@assert false
    # -------------------------------------------------------------------------------
    ## STEP 2: Find the stationary equilibrium for final grid
    # -------------------------------------------------------------------------------
    if verbose in [:low, :high]
        println("Finding equilibrium capital stock for refined income grid...")
    end
    original_init_grids!(m)

    # Find stationary equilibrium for refined economy
    # a.) Define excess demand function with coarse = false
@inline function d(  K,
               initial::Bool=true,
               Vm_guess = zeros(1,1,1),
               Vk_guess = zeros(1,1,1),
               distr_guess = init_distr_guess
               )
        out = original_Kdiff(K, m, initial, Vm_guess, Vk_guess, distr_guess;
                             verbose = verbose, coarse = false)
    return out
end

    # KSS = 40.8946
    # b.) Find equilibrium capital stock (multigrid on y,m,k) # TODO: isn't grid on (m, k, y)?
    BrentOut = CustomBrent(d, KSS*.8, KSS*1.2; tol = get_setting(m, :ϵ))
    KSS      = BrentOut[1]
    @show KSS
    VmSS     = BrentOut[3][2]
    VkSS     = BrentOut[3][3]
    distrSS  = BrentOut[3][4]
    if verbose in [:low, :high]
        println("Capital stock is $(KSS)")
    end
    @show size(VmSS)
    @show size(VkSS)
    @show size(distrSS)
    #@assert false
    return KSS, VmSS, VkSS, distrSS
end
