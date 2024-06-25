function original_Ksupply(RB_guess::T, R_guess::T, m::BayerBornLuetticke{T1}, Vm::AbstractArray, Vk::AbstractArray,
                          distr_guess::AbstractArray, inc::AbstractArray, eff_int::AbstractArray;
                          verbose::Symbol = :none, coarse::Bool = false) where {T <: Real, T1 <: Real}

    ## Set up
    # initialize distance variables
    dist                = 9999.0
    dist1               = dist
    dist2               = dist
    Π                   = m.grids[:Π]::Matrix{T1}             # type declarations necessary b/c grids is an OrderedDict =>
    m_grid              = get_gridpts(m, :m_grid)::Vector{T1} # ensures type stability, or else unnecessary allocations are made
    m_ndgrid            = m.grids[:m_ndgrid]::Array{T1, 3}
    k_ndgrid            = m.grids[:k_ndgrid]::Array{T1, 3}
    q                   = 1.0       # price of Capital

    # Map parameter values to NamedTuple
    θ                   = parameters2namedtuple(m) # and pass parameters as a NamedTuple

    #----------------------------------------------------------------------------
    # Iterate over consumption policies
    #----------------------------------------------------------------------------
    count               = 0
    n                   = size(Vm)
    ϵ                   = get_setting(m, coarse ? :coarse_ϵ : :ϵ)

    # containers for policies, initialized here
    # so we have access to them outside the while loop below
    m_n_star            = Vector{T}(undef, 0) # just need to make sure these have the right types
    m_a_star            = Vector{T}(undef, 0)
    k_a_star            = Vector{T}(undef, 0)
    c_a_star            = Vector{T}(undef, 0)
    c_n_star            = Vector{T}(undef, 0)

    while dist > ϵ && count < get_setting(m, :max_value_function_iters) # Iterate consumption policies until convergence
        count          += 1

        # Take expectations for labor income change # TODO: is there a more efficient way to write this expectation w/out using reshape?
#=        EVk             = reshape(reshape(Vk, (n[1] * n[2], n[3])) * Π', (n[1], n[2], n[3]))
        EVm             = reshape((reshape(eff_int, (n[1] * n[2], n[3])) .*
                                   reshape(Vm, (n[1] * n[2], n[3]))) * Π', (n[1], n[2], n[3]))=#



        EVk             = reshape(reshape(Vk, (n[1] .* n[2], n[3])) * Π', (n[1], n[2], n[3]))
        EVm             = reshape((reshape(eff_int, (n[1] .* n[2], n[3])) .*
                                   reshape(Vm, (n[1] .* n[2], n[3]))) * Π', (n[1], n[2], n[3]))
        # Policy update step
        c_a_star, m_a_star, k_a_star, c_n_star, m_n_star =
            original_EGM_policyupdate(EVm, EVk, q, θ[:π], RB_guess, 1.0, inc, θ, m.grids, false)

        # marginal value update step
        Vk_new, Vm_new  = original_updateV(EVk, c_a_star, c_n_star, m_n_star, R_guess - 1.0, q, θ, m_grid, Π)

        # Calculate distance in updates
#=        dist1           = maximum(abs, _bbl_invmutil(Vk_new, θ[:ξ]) - _bbl_invmutil(Vk, θ[:ξ]))
        dist2           = maximum(abs, _bbl_invmutil(Vm_new, θ[:ξ]) - _bbl_invmutil(Vm, θ[:ξ]))=#
        dist1           = maximum(abs, _bbl_invmutil(Vk_new, θ[:ξ]) .- _bbl_invmutil(Vk, θ[:ξ]))
        dist2           = maximum(abs, _bbl_invmutil(Vm_new, θ[:ξ]) .- _bbl_invmutil(Vm, θ[:ξ]))
        dist            = max(dist1, dist2) # distance of old and new policy

        # update policy guess/marginal values of liquid/illiquid assets
        Vm              = Vm_new
        Vk              = Vk_new
    end
    if verbose == :high
        println("Maximum absolute error after completing EGM iterations is $(dist)")
    end

    #------------------------------------------------------
    # Find stationary distribution (Is direct transition better for large model?) (TODO: investigate this question)
    #------------------------------------------------------
    # Define transition matrix

    gridpoints = get_idiosyncratic_gridpts(m)

    #S_a, T_a, W_a, S_n, T_n, W_n    = MakeTransition(m_a_star,  m_n_star, k_a_star, Π, n, get_idiosyncratic_gridpts(m))
     S_a, T_a, W_a, S_n, T_n, W_n    = MakeTransition(m_a_star,  m_n_star, k_a_star, Π, n, gridpoints[1], gridpoints[2], gridpoints[3])
    TransitionMat_a                 = sparse(S_a, T_a, W_a, prod(n), prod(n)) # TODO: faster way to construct this, e.g. BlockBanded?
    TransitionMat_n                 = sparse(S_n, T_n, W_n, prod(n), prod(n))
    # TransitionMat                   = θ[:λ] * TransitionMat_a + (1.0 - θ[:λ]) * TransitionMat_n
    TransitionMat                   = θ[:λ] .* TransitionMat_a .+ (1.0 .- θ[:λ]) .* TransitionMat_n

    if get_setting(m, :kfe_method) == :krylov
        # Calculate left-hand unit eigenvector (uses KrylovKit package)
#=        aux   = real.(eigsolve(TransitionMat', 1)[2][1])
        distr = reshape(vec(aux) ./ sum(aux), n)=#
        aux   = real.(eigsolve(TransitionMat', 1)[2][1])
        distr = reshape((aux[:]) ./ sum((aux[:])), n)
    elseif get_setting(m, :kfe_method) == :direct
        # Direct Transition
        distr = get_untransformed_values(m[:distr])::Array{T1, 3}
        distr, dist, count = original_MultipleDirectTransition(m_a_star, m_n_star, k_a_star, distr, θ[:λ], Π,
                                                               n, get_idiosyncratic_gridpts(m), ϵ;
                                                               iters = get_setting(m, :n_direct_transition_iters))
    else
        error("Solution method for Kolmogorov forward equation $(get_setting(m, :kfe_method)) is not recognized. " *
              "Available methods are [:krylov, :direct]")
    end

    #-----------------------------------------------------------------------------
    # Calculate capital stock
    #-----------------------------------------------------------------------------
    K = sum(distr[:] .* k_ndgrid[:])
    B = sum(distr[:] .* m_ndgrid[:])
#=    K = dot(distr, k_ndgrid) # faster to use dot
    B = dot(distr, m_ndgrid)=#

    return K, B, TransitionMat, TransitionMat_a, TransitionMat_n, c_a_star, m_a_star, k_a_star, c_n_star, m_n_star, Vm, Vk, distr
end
