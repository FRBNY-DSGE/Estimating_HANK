function augment_states(m::BayerBornLuetticke{T}, TTT::AbstractMatrix{T}, TTT_jump::AbstractMatrix{T},
                        RRR::AbstractMatrix{T}, CCC::AbstractVector{T}) where {T <: Real}
    # Do nothing for now. Later, add augment states and measurement error
    return TTT, TTT_jump, RRR, CCC
end



function augment_states(m::BayerBornLuetticke{T}, TTT::AbstractMatrix{T}, RRR::AbstractMatrix{T}, CCC::AbstractVector{T}) where {T <: Real}
    println("unclear if we want this for augment states")
    # Do nothing for now. Later, add augment states and measurement error
    return TTT, RRR, CCC
end
