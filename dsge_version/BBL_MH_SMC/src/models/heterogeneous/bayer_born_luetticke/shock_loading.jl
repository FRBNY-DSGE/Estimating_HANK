function shock_loading(m::BayerBornLuetticke{T}) where {T <: Real}

    n_states = n_backward_looking_states(m)::Int
    n_exo_sh = n_shocks_exogenous(m)::Int
    _RRR     = zeros(T, n_states, n_exo_sh)

    return _shock_loading!(m, _RRR)
end

function shock_loading(m::BayerBornLuetticke{T}, TTT_jump::Matrix{T}) where {T <: Real}

    n_states = n_backward_looking_states(m)::Int
    n_exo_sh = n_shocks_exogenous(m)::Int
    RRR      = Matrix{T}(undef, n_model_states(m)::Int, n_exo_sh)

    # Populate RRR in place
    RRR[1:n_states, :] .= 0. # only a few entries are nonzero
    _shock_loading!(m, view(RRR, 1:n_states, :))

    # Map shocks to jumps
    RRR[n_states+1:end, :] = TTT_jump * view(RRR, 1:n_states, :)

    return RRR
end

function _shock_loading!(m::BayerBornLuetticke{T}, _RRR::AbstractMatrix) where {T <: Real}

    # Set up
    exo  = m.exogenous_shocks
    endo = m.endogenous_states

    # Populate _RRR for the shocks
    _RRR[first(endo[:A′_t]), exo[:A_sh]] = 1.
    _RRR[first(endo[:Z′_t]), exo[:Z_sh]] = 1.
    _RRR[first(endo[:Ψ′_t]), exo[:Ψ_sh]] = 1.
    _RRR[first(endo[:μ_p′_t]), exo[:μ_p_sh]] = 1.
    _RRR[first(endo[:μ_w′_t]), exo[:μ_w_sh]] = 1.
    _RRR[first(endo[:G_sh′_t]), exo[:G_sh]] = 1.
    _RRR[first(endo[:R_sh′_t]), exo[:R_sh]] = 1.
    _RRR[first(endo[:S_sh′_t]), exo[:S_sh]] = 1.
    _RRR[first(endo[:P_sh′_t]), exo[:P_sh]] = 1.

    return _RRR
end
