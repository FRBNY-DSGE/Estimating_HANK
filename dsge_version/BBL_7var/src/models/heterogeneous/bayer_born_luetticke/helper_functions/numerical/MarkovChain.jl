"""
```
tauchen86(ρ, N; σ = 1., μ_e = 0.)
```
Generate a discrete approximation to an AR(1) process. This function
is named after Tauchen (1986), but it actually is implementing
the Adda and Cooper (2003) method.

Uses importance sampling: each bin has probability 1/N to realize

# Arguments
- `ρ`: autocorrelation coefficient
- `N`: number of gridpoints
- `σ`: long-run variance (short-run variance is `σ_e = σ * sqrt(1 - ρ^2)`)
- `μ_e`: mean of the AR(1) process

# Returns
- `grid_vec`: state vector grid
- `P`: transition matrix
- `bounds`: bin bounds
"""
function tauchen86(ρ::T, N::Int; σ::T = 1.0, μ_e::T = 0.0) where {T <: Real}
#   Author: Christian Bayer, Uni Bonn, 03.05.2010
#   Modified by William Chen, April 6, 2021.
#   See https://github.com/BenjaminBorn/HANK_BusinessCycleAndInequality for original implementation as "Tauchen"

    dis = Normal()
    pr_ij(x, bound1, bound2, ρ, σ_e) = pdf.(dis, x) .*
        (cdf.(dis, (bound2 - ρ .* x) ./ σ_e) -
         cdf.(dis, (bound1 - ρ .* x) ./ σ_e) )

    grid_probs = range(0.0, stop = 1.0, length = N+1)   # generate equi-likely bins

    bounds = quantile.(dis, grid_probs[1:end])     # corresponding bin bounds

    # replace [-]Inf bounds, by finite numbers
    bounds[1] = bounds[2] - 1.0e2
    bounds[end] = bounds[end-1] + 1.0e2

    # Calculate grid() - centers
    grid_vec = N * (pdf.(dis, bounds[1:end-1]) - pdf.(dis, bounds[2:end]))

    # σ_e = sqrt(1 - ρ^2) # Calculate short run variance # NOTE: this was the original line of code, but I think this assumes σ = 1
    σ_e = sqrt(1 - ρ^2) * σ # Calculate short run variance
    P = fill(0.0, (N, N)) # Initialize Transition Probability Matrix

    for j = 1:N
        p(x) = pr_ij(x,bounds[j], bounds[j+1], ρ, σ_e)
        for i = 1:floor(Int, (N-1)/2)+1 # Exploit Symmetrie to save running time
            P[i, j] = _bbl_gauss_chebyshev_integrate(p, bounds[i], bounds[i+1]) # Evaluate Integral
        end
    end

    # Exploit Symmetrie Part II
    P[floor(Int, (N - 1) / 2) + 2:N, :] = P[(ceil(Int, (N - 1) / 2):-1:1), end:-1:1]

    # Make sure P is a Probability Matrix
    P = P ./ sum(P, dims = 2)

    grid_vec   = grid_vec .* σ .+ μ_e
    lmul!(σ, bounds)
    # bounds = bounds .* σ

    return grid_vec, P, bounds
end

# Does the exact same thing as tauchen86 above except it does not re-create the bin bounds
# and does not create the grid vector implied by the bins. This function here therefore
# only creates the transition matrix. It is primarily used to get the Jacobian
# of the model with respect to an infinitessimal change in the unconditional variance
# of the income process.
function ExTransition(rho::Number,bounds::Array{Float64,1},riskscale::Number)
#similar to TAUCHEN
N = length(bounds)-1
# Assume Importance Sampling
        sigma_e=riskscale*sqrt(1-rho^2)# Calculate short run variance
        P=zeros(typeof(riskscale),N,N) # Initialize Transition Probability Matrix

    # this appears to be a similar integration procedure as above, not sure how it is different yet
        for i=1:floor(Int,(N-1)/2)+1
            nodes, weights = _bbl_qnwcheb(500, bounds[i], bounds[i+1])
            for j=1:N
            p(x) = pr_ij(x,bounds[j],bounds[j+1],rho,sigma_e)
             # Exploit Symmetrie to save running time
                P[i,j]=dot(weights, p.(nodes)) # _bbl_gauss_chebyshev_integrate(p,bounds[i],bounds[i+1]) # Evaluate Integral
            end
        end

       # Exploit Symmetrie Part II
        P[floor(Int,(N-1)/2)+2:N,:]=P[(ceil(Int,(N-1)/2):-1:1),end:-1:1]

# Make sure P is a Probability Matrix
P = P./ sum(P, dims = 2)

return P

end

function pr_ij(x,bound1,bound2,rho,sigma_e)
    mycdf(x) = 0.5 + 0.5 * erf.(x / sqrt(2.0))
    mypdf(x) =  1/sqrt(2*π).*exp.(-x.^2/2.0)
    p = mypdf.(x) .* (mycdf.((bound2 - rho.*x)./sigma_e) - mycdf.((bound1 - rho.*x)./sigma_e) )
return p
end

"""
```
rouwenhorst(N, ρ, σ, μ=0.0)
```
Rouwenhorst's method to approximate AR(1) processes. This code
was directly copied from QuantEcon.jl (and modified to
return the nodes and transition matrix directly).

The process follows
```math
    y_t = mu + rho y_{t-1} + epsilon_t
```
where ``epsilon_t ~ N (0, sigma^2)``
##### Arguments
- `N::Integer` : Number of points in markov process
- `ρ::Real` : Persistence parameter in AR(1) process
- `σ::Real` : Standard deviation of random component of AR(1) process
- `μ::Real` :  Mean of AR(1) process

##### Returns
- `state_values` and `transition_matrix`
"""
function rouwenhorst(N::Integer, ρ::Real, σ::Real, μ::Real=0.0)
    σ_y = σ / sqrt(1-ρ^2)
    p  = (1+ρ)/2
    ψ = sqrt(N-1) * σ_y
    m = μ / (1 - ρ)

    return _rouwenhorst(p, p, m, ψ, N) # returns state_values, transition_matrix
end

function _rouwenhorst(p::Real, q::Real, m::Real, Δ::Real, n::Integer)
    if n == 2
        return [m-Δ, m+Δ],  [p 1-p; 1-q q]
    else
        _, θ_nm1 = _rouwenhorst(p, q, m, Δ, n-1)
        θN = p    *[θ_nm1 zeros(n-1, 1); zeros(1, n)] +
             (1-p)*[zeros(n-1, 1) θ_nm1; zeros(1, n)] +
             q    *[zeros(1, n); zeros(n-1, 1) θ_nm1] +
             (1-q)*[zeros(1, n); θ_nm1 zeros(n-1, 1)]

        θN[2:end-1, :] ./= 2

        return range(m-Δ, stop=m+Δ, length=n), θN
    end
end

function rouwenhorst_transition(N::Integer, ρ::Real, σ_y::Real, μ::Real=0.0)
    # σ_y = σ / sqrt(1-ρ^2)
    p  = (1+ρ)/2
    # ψ = sqrt(N-1) * σ_y
    m = μ / (1 - ρ)

    return _rouwenhorst_transition(p, p, m, N) # returns transition_matrix
end

function _rouwenhorst_transition(p::Real, q::Real, m::Real, n::Integer)
    if n == 2
        return [p 1-p; 1-q q]
    else
        _, θ_nm1 = _rouwenhorst_transition(p, q, m, n-1)
        θN = p    *[θ_nm1 zeros(n-1, 1); zeros(1, n)] +
             (1-p)*[zeros(n-1, 1) θ_nm1; zeros(1, n)] +
             q    *[zeros(1, n); zeros(n-1, 1) θ_nm1] +
             (1-q)*[zeros(1, n); θ_nm1 zeros(n-1, 1)]

        θN[2:end-1, :] ./= 2

        return θN
    end
end
