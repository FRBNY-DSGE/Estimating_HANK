function eval_three_states(f::Function, grid1::AbstractVector{T}, grid2::AbstractVector{T}, grid3::AbstractVector{T}) where {T <: Real}
    return [f(i, j, k) for i in grid1, j in grid2, k in grid3]
end
