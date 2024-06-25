# Constructor functions for grid
function _construct_liquid_asset_grid_bbl(mmin::T, mmax::T, nm::Int) where {T <: Real}
    return exp.(range(0, stop = log(mmax - mmin + 1.), length = nm)) .+ mmin .- 1.
end
function _construct_illiquid_asset_grid_bbl(kmin::T, kmax::T, nk::Int) where {T <: Real}
    return exp.(range(log(kmin + 1.), stop = log(kmax + 1.), length = nk)) .- 1.
end

"""
```
init_grids!(m::BayerBornLuetticke; coarse::Bool = false)
```
constructs the grids used for distribution/functional states
and stores them in m.grids. If `coarse = true`, then the
constructed grid constructed has a smaller dimension
and is thus "coarse".
"""
function init_grids!(m::BayerBornLuetticke{T}; coarse::Bool = false) where {T <: Real}

    # Grab dictionary for holding grids
    grids = m.grids

    # Get variable names (depending on whether it's a coarse grid or not)
    ny   = coarse ? get_setting(m, :coarse_ny)   : get_setting(m, :ny)
    kmin = coarse ? get_setting(m, :coarse_kmin) : get_setting(m, :kmin)
    kmax = coarse ? get_setting(m, :coarse_kmax) : get_setting(m, :kmax)
    nk   = coarse ? get_setting(m, :coarse_nk)   : get_setting(m, :nk)
    mmin = coarse ? get_setting(m, :coarse_mmin) : get_setting(m, :mmin)
    mmax = coarse ? get_setting(m, :coarse_mmax) : get_setting(m, :mmax)
    nm   = coarse ? get_setting(m, :coarse_nm)   : get_setting(m, :nm)

    # Initialize transition matrix and income grid
    # recall that for Grid constructor, arg 2 = quadrature weights, arg 3 = "scale" or Lebesgue measure,
    # and that uniform_quadrature's 2nd output is the weights vector

    # Liquid asset grid
    m_grid = _construct_liquid_asset_grid_bbl(mmin, mmax, nm)
    m_grid[findlast(x -> x < 0, m_grid)] = 0.0 # Guarantee there is a zero is on the m grid (liquid asset)
    grids[:m_grid] = Grid(m_grid, uniform_quadrature(mmin, mmax, nm; scale = mmax - mmin)[2], mmax - mmin)

    # Illiquid asset grid
    grids[:k_grid] = Grid(_construct_illiquid_asset_grid_bbl(kmin, kmax, nk),
                          uniform_quadrature(kmin, kmax, nk; scale = kmax - kmin)[2],
                          kmax - kmin)

    ## Idiosyncratic income states for workers + entrepreneurs

    # Construct idiosyncratic income grid for workers alone
    ny_min1             = ny - 1 # subtract 1 b/c want ny - 1 states for workers, last state for entrepreneurs' income
    y_grid_e, Π, bounds = tauchen86(get_untransformed_values(m[:ρ_h]), ny_min1) # Income grid and transitions for workers

    # Update ymin, ymax, and bounds
    ymin, ymax = y_grid_e[1], y_grid_e[end] # y_grid_e is ordered left to right
    m <= Setting(coarse ? :coarse_ymin : :ymin, ymin)
    m <= Setting(coarse ? :coarse_ymax : :ymax, ymax)
    grids[:y_bin_bounds] = bounds

    # Add entrepreneurs into the income transitions => total income states = ny
    Π      = [(Π .* (1. - m[:ζ])) fill(get_untransformed_values(m[:ζ]), ny_min1);
              fill(m[:ι] / ny_min1, 1, ny_min1) (1. - m[:ι])]
    y_grid = vcat(exp.(y_grid_e .* m[:σ_h] ./ sqrt(1. - m[:ρ_h]^2)), (m[:ζ] + m[:ι]) / m[:ζ])

    # Update long-run average level of human capital (of the worker) to be consistent with income process
    Paux       = Π ^ 1000 # compute LR average implied by process
    grids[:H]  = Paux[1, 1:end-1]' * y_grid[1:end-1] # ignore last state b/c last state is entrepreneurs' state
    grids[:HW] = coarse ? 1. : 1. / (1. - Paux[end, end])

    # Construct and store income grid and transition matrix
    y_worker_weights = uniform_quadrature(ymin, ymax, ny_min1; scale = ny_min1 / ny)[2] # weights for worker states
    y_entrep_weights = 1. / ny # weights for entrepreneur's state. Place equal weights; probability handled elsewhere
    grids[:y_grid] = Grid(y_grid, vcat(y_worker_weights, y_entrep_weights), 1.)
    grids[:Π]      = Π
    grids[:Paux]   = Paux # store approximate stationary distribution

    # Construct ndgrids (TODO: delete this and be Julian by never allocating these grids and using list comprehensions, IF this is feasible)
    grids[:m_ndgrid], grids[:k_ndgrid], grids[:y_ndgrid] = ndgrid([get_gridpts(m, x) for x in [:m_grid, :k_grid, :y_grid]]...)
    grids[:weights_ndgrid] = eval_three_states((x, y, z) -> x * y * z, [get_gridwts(m, x) for x in [:m_grid, :k_grid, :y_grid]]...)

    # Create initial guess for distribution
    m[:distr_star].value = fill(1. / (nm * nk * ny), nm, nk, ny)

    return m
end
