"""
```
function copula(μ::AbstractArray{S, N}, quadrature_weights::AbstractArray{S, N} = Array{S, N}(undef, zeros(Int, N)...);
                method::Interpolations.InterpolationType = Gridded(Linear())) where {S <: Real, N <: Int}
```
creates an Interpolations object that is the copula for the discretized probability density function `μ`.
If `c = copula(μ, ...)` where `μ` is a `Matrix`, then `c(x, y)` maps the marginal `x` and marginal `y` cumulative distributions
to the cumulative distribution implied by `μ`.
"""
function copula(μ::AbstractArray{S, N}, quadrature_weights::AbstractArray{S, N} = Array{S, N}(undef, zeros(Int, N)...);
                method::Interpolations.InterpolationType = Gridded(Linear())) where {S <: Real, N}

    # Get overall CDF and CDFs of marginals
    Fμ         = cdf_quadrature(μ, quadrature_weights)
    marginals  = marginal_cdf_quadrature(μ, quadrature_weights)

    # Construct interpolation object via Gridded(Linear())
    return interpolate(marginals, Fμ, method)
end

function marginal_pdf_quadrature(μ::AbstractArray{S, N}, quadrature_weights::AbstractArray{S, N} =
                                 Array{S, N}(undef, zeros(Int, N)...)) where {S <: Real, N}
    marg_pdfs = Tuple(Vector{S}(undef, n) for n in size(μ))
    all_dims  = collect(1:N)

    if !isempty(quadrature_weights)
        # Re-weight distribution to incorporate quadrature.
        # Note that we don't do μ .*= quadrature_weights to keep this an out-of-place function
        μ = μ .* quadrature_weights
    end

    # Create array of tuples of dimensions to sum over for each marginal
    vec_i_dims = [Tuple(setdiff(all_dims, i)) for i in 1:N]
    marg_pdfs  = Tuple(dropdims(sum(μ, dims = i_dims), dims = i_dims) for i_dims in vec_i_dims)

    return marg_pdfs
end

function marginal_cdf_quadrature(μ::AbstractArray{S, N}, quadrature_weights::AbstractArray{S, N} =
                                 Array{S, N}(undef, zeros(Int, N)...)) where {S <: Real, N}
    # Set up
    marg_cdfs = Tuple(Vector{S}(undef, n + 1) for n in size(μ))
    all_dims  = collect(1:N)

    # Get marginal PDFs
    marg_pdfs = marginal_pdf_quadrature(μ, quadrature_weights)

    # Get marginal cdfS
    for (i, marg_pdf, marg_cdf) in zip(all_dims, marg_pdfs, marg_cdfs)
        marg_cdf[1]     = 0. # set first index to zero
        i_dims          = Tuple(setdiff(all_dims, i))
        marg_cdf[2:end] = cumsum(marg_pdf) # cumsum for CDF
    end

    return marg_cdfs
end

function cdf_quadrature(μ::AbstractArray{S, N}, quadrature_weights::AbstractArray{S, N} =
                        Array{S, N}(undef, zeros(Int, N)...)) where {S <: Real, N}
    # Initialize CDF matrix as zeros. Slightly inefficient, but avoid complication of setting boundaries of CDF to zero
    Fμ_sz = size(μ) .+ 1
    Fμ    = zeros(S, size(μ) .+ 1) # CDF of μ
    inds  = CartesianIndices(Tuple(2:i for i in Fμ_sz))

    # Perform integration for CDF
    A     = @view Fμ[inds]                                            # portion of CDF to actually fill in
    A    .= isempty(quadrature_weights) ? μ : μ .* quadrature_weights # multiply by quadrature weights so that cumsum works if needed
    for i in 1:N
        cumsum!(A, A, dims = i) # same as cumsum(cumsum(A, dims = 1), dims = 2) if μ is 2D
    end

    return Fμ
end
