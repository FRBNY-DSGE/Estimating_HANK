#----------------------------------------------------------------------------
# Basic Functions: Return on capital, marginal utility and its inverse
_original_bbl_interest(K::Real, A::Real, N::Real, α::Union{Real, AbstractParameter}, δ_0::Union{Real, AbstractParameter}) = A .* α .* (K ./ N) .^(α - 1.0) .- δ_0
_original_bbl_wage(K::Real, A::Real, N::Real, α::Union{Real, AbstractParameter}) = A .* (1. - α) .* (K./N) .^ α
_original_bbl_output(K::Real, A::Real, N::Real, α::Union{Real, AbstractParameter}) = A .* K .^(α) .* N .^(1 - α)
@inline function _original_bbl_employment(K::Real, A::Real, α::Union{Real, AbstractParameter},
                                 τ_lev::Union{Real, AbstractParameter}, τ_prog::Union{Real, AbstractParameter},
                                 γ::Union{Real, AbstractParameter})
    return (A .* (1.0 - α) .* (τ_lev .* (1.0 - τ_prog)).^(1.0 / (1.0 - τ_prog))
            .* K .^(α)).^((1.0 - τ_prog) ./ (γ + τ_prog + (α) .* (1. - τ_prog)))
end

function original_distrSummaries(distr::AbstractArray, Q, c_a_star::AbstractArray,
                                 c_n_star::AbstractArray, inc::AbstractArray,
                                 incgross::AbstractArray, θ::NamedTuple,
                                 dims::NTuple{3, Int}, grids::OrderedDict)
    # Set up
    nm, nk, ny = dims
    m_grid = get_gridpts(grids, :m_grid)::Vector{Float64}
    k_grid = get_gridpts(grids, :k_grid)::Vector{Float64}
    m_ndgrid = grids[:m_ndgrid]::Array{Float64, 3}

    ## Distributional summaries
    mplusk = Vector{eltype(c_a_star)}(undef, nk * nm)
    @inbounds for k = 1:nk
        for m = 1:nm
            mplusk[m + (k - 1) * nm] = m_grid[m] .+ Q .* k_grid[k]
        end
    end
    IX               = sortperm(mplusk)
    mplusk           = mplusk[IX]
    moneycapital_pdf = sum(distr, dims = 3)
    moneycapital_pdf = moneycapital_pdf[IX]
    moneycapital_cdf = cumsum(moneycapital_pdf)
    S                = [0.; cumsum(moneycapital_pdf .* mplusk)]
    # giniwealth       = 1. - (dot(moneycapital_pdf, (S[1:end-1] + S[2:end])) ./ S[end])
    giniwealth       = 1. - (sum(moneycapital_pdf .* (S[1:end-1] + S[2:end])) ./ S[end])

    distr_m = vec(sum(distr, dims=(2,3)))
    distr_k = vec(sum(distr, dims=(1,3)))
    distr_y = vec(sum(distr, dims=(1,2)))

    share_borrower = sum(distr_m .* (m_grid .< 0)) # sum(distr_m[m_grid .< 0])

    p50             = findfirst(x -> x >= 0.5, moneycapital_cdf)
    p90             = findfirst(x -> x >= 0.9, moneycapital_cdf)
    w9050           = mplusk[p90] / mplusk[p50]
    # FN_wealthshares = cumsum(mplusk .* moneycapital_pdf) ./ dot(mplusk, moneycapital_pdf)
    FN_wealthshares = cumsum(mplusk .* moneycapital_pdf) ./ sum(mplusk .* moneycapital_pdf)
    w90share        = 1.0 - mylinearinterpolate(moneycapital_cdf, FN_wealthshares, [0.9])[1]

    x                   = Array{eltype(c_a_star)}(undef, nm, nk, ny, 2)
    c                   = Array{eltype(c_a_star)}(undef, nm, nk, ny, 2)
    distr_x             = Array{eltype(c_a_star)}(undef, nm, nk, ny, 2)
    x[:, :, :, 1]       = c_a_star
    x[:, :, :, 2]       = c_n_star
    aux_x               = inc[5] # (1 .- θ[:τ_bar]) .* (1.0 ./ θ[:μ_w]) .* w .* N .* grids[:y_ndgrid] ./ (m[:γ] + 1)
    aux_x[:, :, end]   .= 0. # this overwrites inc[5] but it's fine b/c we don't use inc[5] after this function is called
    c[:, :, :, 1]       = view(x, :, :, :, 1) + aux_x # (remaining step is reduction, which doesn't require inc[5])
    c[:, :, :, 2]       = view(x, :, :, :, 2) + aux_x
    distr_x[:, :, :, 1] = θ[:λ] .* distr
    distr_x[:, :, :, 2] = (1. - θ[:λ]) .* distr

    IX                  = sortperm(vec(x))
    x                   = x[IX]
    logx                = log.(x)
    x_pdf               = distr_x[IX]
    S                   = vcat(0., cumsum(x_pdf .* x))
    ginicompconsumption = 1. - (sum(x_pdf .* (S[1:end-1] + S[2:end])) / S[end])
    # ginicompconsumption = 1. - (dot(x_pdf, (S[1:end-1] + S[2:end])) ./ S[end])
    sdlogx              = sqrt(x_pdf[:]' * log.(x[:]).^2 - (x_pdf[:]' * log.(x[:]))^2) # off!
    # sdlogx              = sqrt(dot(x_pdf, logx.^2) - dot(x_pdf, logx)^2)

    IX              = sortperm(vec(c))
    c               = c[IX]
    logc            = log.(c)
    c_pdf           = distr_x[IX]
    S               = vcat(0., cumsum(c_pdf .* c))
    c_cdf           = cumsum(c_pdf)

    p10             = findfirst(x -> x >= 0.1, c_cdf)
    p50             = findfirst(x -> x >= 0.5, c_cdf)
    p90             = findfirst(x -> x >= 0.9, c_cdf)

    c9010           = c[p90] / c[p10]

    p10C            = c[p10]
    p50C            = c[p50]
    p90C            = c[p90]

#=    giniconsumption = 1. - (dot(c_pdf, (S[1:end-1] + S[2:end])) / S[end])
    sdlogc          = sqrt(dot(c_pdf, logc.^2) - dot(c_pdf, logc)^2)=#
    giniconsumption = 1. - (sum(c_pdf .* (S[1:end-1] + S[2:end])) / S[end])
    sdlogc          = sqrt(c_pdf[:]' * log.(c[:]).^2 - (c_pdf[:]' * log.(c[:]))^2) # off!

    Yidio              = inc[6] + inc[2] + inc[3] - m_ndgrid
    # IX                 = sortperm(vec(Yidio))
    IX                 = sortperm(Yidio[:])
    Yidio              = Yidio[IX]
    Y_pdf              = distr[IX]
    Y_cdf              = cumsum(Y_pdf)
    # p10                = findfirst(x -> x >= 0.1, Y_cdf)
    p10 = count(Y_cdf .< 0.1) + 1
    # FN_incomesharesnet = cumsum(Yidio .* Y_pdf) ./ dot(Yidio, Y_pdf)
    FN_incomesharesnet = cumsum(Yidio .* Y_pdf) ./ sum(Yidio .* Y_pdf)
    I90sharenet        = 1.0 .- mylinearinterpolate(Y_cdf, FN_incomesharesnet, [0.9])[1]

    Yidio           = incgross[1] + incgross[2] + incgross[3] - m_ndgrid
    # IX              = sortperm(vec(Yidio))
    IX              = sortperm(Yidio[:])
    Yidio           = Yidio[IX]
    Y_pdf           = distr[IX]
    Y_cdf           = cumsum(Y_pdf)
    # FN_incomeshares = cumsum(Yidio .* Y_pdf) ./ dot(Yidio, Y_pdf)
    FN_incomeshares = cumsum(Yidio .* Y_pdf) ./ sum(Yidio .* Y_pdf)
    I90share        = 1.0 .- mylinearinterpolate(Y_cdf, FN_incomeshares, [0.9])[1]

    S               = vcat(0., cumsum(Y_pdf .* Yidio))
    # giniincome      = 1. - (dot(Y_pdf, (S[1:end-1] + S[2:end])) / S[end])
    giniincome      = 1. - (sum(Y_pdf .* (S[1:end-1] + S[2:end])) / S[end])

    Yidio           = incgross[1]
    Yidio           = Yidio[:, :, 1:end-1]
    logYidio        = log.(Yidio)
    # IX              = sortperm(vec(Yidio))
IX              = sortperm(Yidio[:])
    Yidio           = Yidio[IX]
    distr_aux       = distr[:, :, 1:end-1]
    # distr_aux       = distr_aux ./ sum(distr_aux)
distr_aux       = distr_aux ./ sum(distr_aux[:])
    Y_pdf           = distr_aux[IX]
    Y_cdf           = cumsum(Y_pdf)
    # p10             = findfirst(x -> x >= 0.1, Y_cdf)
    # p50             = findfirst(x -> x >= 0.5, Y_cdf)
p10 = count(Y_cdf .< 0.1) + 1
p50 = count(Y_cdf .< 0.5) + 1
    y5010           = Yidio[p50] ./ Yidio[p10]

    # sdlogy          = sqrt(dot(Y_pdf, logYidio.^2) - dot(Y_pdf, logYidio)^2)
sdlogy          = sqrt(Y_pdf[:]' * log.(Yidio[:]).^2 - (Y_pdf[:]' * log.(Yidio[:]))^2)

    return distr_m, distr_k, distr_y, share_borrower, giniwealth, I90share, I90sharenet, ginicompconsumption, #= # comment used to split the
    =# sdlogx, c9010, giniconsumption, sdlogc, y5010, giniincome, sdlogy, w90share, p10C, p50C, p90C            # return output into two lines
end
