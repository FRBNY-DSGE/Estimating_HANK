"""
```
EGM_policyupdate(EVm::Array, EVk::Array, Qminus::Real, πminus::Real,
                 RBminus::Real, Tshock::Real, inc::Array,
                 θ::NamedTuple, grids::OrderedDict, warnme::Bool;
                 parallel::Bool = false)
```
Find optimal policies, given marginal continuation values `EVm`, `EVk`, today's
prices [`Qminus`, `πminus`,`RBminus`], and income [`inc`], using the
Endogenous Grid Method. The inputs `θ` and `grids` map
the names of economic parameters and quantities related
to the idiosyncratic state space to their values, respectively.

Optimal policies are defined on the fixed grid, but optimal asset choices (`m` and `k`)
are off-grid values.

### Returns
- `c_a_star`,`m_a_star`,`k_a_star`,`c_n_star`,`m_n_star`: optimal (on-grid) policies for
    consumption [`c`], liquid [`m`] and illiquid [`k`] asset, with [`a`] or
    without [`n`] adjustment of illiquid asset
"""
function EGM_policyupdate!(EVm::Array,
                          EVk::Array,
                          Qminus::Real,
                          πminus::Real,
                          RBminus::Real,
                          Tshock::Real,
                          inc::Array,
                          θ::NamedTuple,
                          grids::OrderedDict,
                          warnme::Bool,
                          c_a_star::Array,
                          m_a_star::Array,
                          k_a_star::Array,
                          c_n_star::Array,
                          m_n_star::Array;
                          parallel::Bool = false)

# Expensiveness of steps on coarse grid (ny = 6)
# Step 1 => .0004 s (500 alloc)
# Step 2 => .00005  (255 alloc)
# Step 3 => .000025 (100 alloc)
# Step 4 => .0003 s (71 alloc)

    if parallel
        return _parallel_EGM_policyupdate!(EVm, EVk, Qminus, πminus, RBminus, Tshock, inc, θ, grids, warnme,
                                           c_a_star, m_a_star, k_a_star, c_n_star, m_n_star)
    end
    # TODO: add more comments explaining how we figure out policies when households can and cannot adjust portfolios

    ################### Copy/read-out stuff#####################################
    β          = θ[:β]
    borrwedge  = θ[:Rbar] .* Tshock
    # inc[1] = labor income , inc[2] = rental income,
    # inc[3]= liquid assets income, inc[4] = capital liquidation income
    inc_lab    = inc[1] # TODO: do we need to also sign the types of inc?
    inc_rent   = inc[2]
    inc_LA     = inc[3]
    inc_IA     = inc[4]
    n          = size(EVm)
    m_grid     = get_gridpts(grids, :m_grid)::Vector{Float64} # type declarations necessary b/c grids is an OrderedDict =>
    k_grid     = get_gridpts(grids, :k_grid)::Vector{Float64} # ensures type stability, or else unnecessary allocations are made
    m_ndgrid   = grids[:m_ndgrid]::Array{Float64, 3}
    mmax       = m_grid[end]
    kmax       = k_grid[end]

    ############################################################################
    ## EGM Step 1: Find optimal liquid asset holdings in the constrained case ##
    ############################################################################
    EVm       .*= β   # directly adjust EVm and set EMU = EVm
    EMU         = EVm # since we don't need access to the original EVm for the remainder of an EGM loop
    c_star_n    = _bbl_invmutil(EMU, θ[:ξ]) # 6% of time with rolled out power function

    # Calculate assets consistent with choices being [m']
    # Calculate initial money position from the budget constraint
    # that leads to the optimal consumption choice
    m_star_n    = c_star_n + m_ndgrid - inc_lab - inc_rent

    # Apply correct interest rate
    m_star_n   ./= ((RBminus .+ borrwedge .* (m_star_n .< 0)) ./ πminus)  # apply borrowing rate

    # Next step: Interpolate w_guess and c_guess from new k-grids
    # using c[s,h,m"], m(s,h,m")
    # Interpolate grid().m and c_n_aux defined on m_star_n over grid().m

    # Check monotonicity of m_star_n
    if warnme
        m_star_aux = reshape(m_star_n, (n[1], n[2] * n[3]))
        if any(x -> x < 0, diff(m_star_aux, dims = 1))
            @warn "non monotone future liquid asset choice encountered"
        end
    end

    # Policies for tuples (c*,m*,y) are now given. Need to interpolate to return to
    # fixed grid. Note that c_n_star & m_n_star are the policies when HH cannot adjust
    @fastmath @inbounds @views begin
        @simd for jj = 1:n[3] # Loop over income states
            @simd for kk = 1:n[2] # Loop over capital states
                cc, mn = mylinearinterpolate_mult2(m_star_n[:, kk, jj], c_star_n[:, kk, jj], m_grid, m_grid)
                c_n_star[:, kk, jj] = cc
                m_n_star[:, kk, jj] = mn

                # Check for binding borrowing constraints, no extrapolation from grid
                bcpol = m_star_n[1, kk, jj]
                @simd for mm = 1:n[1]
                    if m_ndgrid[mm, kk, jj] < bcpol
                        c_n_star[mm, kk, jj] = inc_lab[mm, kk, jj] + inc_rent[mm, kk, jj] + inc_LA[mm, kk, jj] - m_grid[1]
                        m_n_star[mm, kk, jj] = m_grid[1]
                    end
                    if mmax < m_n_star[mm, kk, jj]
                        m_n_star[mm, kk, jj] = mmax
                    end
                end
            end
        end
    end
    #-------------------------END OF STEP 1-----------------------------

    ############################################################################
    ## EGM Step 2: Find Optimal Portfolio Combinations                        ##
    ############################################################################
    # Find an m_a* for given k' that yield the same expected future marginal value
    # for liquid and illiquid assets:
    term1           = (β / Qminus) * EVk                    # expected marginal value of illiquid investment
    E_return_diff   = term1 - EMU                           # difference conditional on future asset holdings on grid
    m_a_aux1        = Fastroot(m_grid, E_return_diff)       # Find indifferent m by interpolation of two neighboring points a, b ∈ grid_m with: E_return_diff(a) < 0 < E_return_diff(b)
    # (Fastroot does not allow for extrapolation and uses non-negativity constraint and monotonicity)
    m_a_aux         = reshape(m_a_aux1, (n[2], n[3]))

    # Note that we allocate a new matrix for term1 rather than over-write EVk (in contrast to EVm)
    # b/c we want the original EVk available later

    ###########################################################################
    ## EGM Step 3: Constraints for money and capital are not binding         ##
    ###########################################################################
    # Interpolation of psi()-function at m*_n[m,k]
    aux_index       = (0:(n[2] * n[3]) - 1) * n[1]                  # auxiliary to move to linear indexing
    EMU_star        = Matrix{eltype(m_a_aux)}(undef, n[2], n[3])    # container (note it is over capital and income)
    step            = diff(m_grid)                                  # Stepsize on grid()

    # Interpolate EMU[m",k',s'*h',M',K'] over m*_n[k"], m-dim is dropped
    @inbounds @fastmath @simd for j in eachindex(m_a_aux)
        xi          = m_a_aux[j]

        # find indexes on grid next smallest to optimal policy
        if xi > m_grid[n[1] - 1]                                    # policy is larger than highest grid point
            idx     = n[1] - 1
        elseif xi <= m_grid[1]                                      # policy is smaller than lowest grid point
            idx     = 1
        else
            idx     = locate(xi, m_grid)                            # use exponential search to find grid point closest to policy (next smallest)
        end

        s           = (xi - m_grid[idx]) / step[idx]                # Distance of optimal policy to next grid point to get convex weights

        EMU_star[j] = EMU[idx + aux_index[j]] * (1.0 - s) +         # linear interpolation to populate EMU using s as a convex weight
        s * (EMU[idx + aux_index[j] + 1])
    end
c_a_aux         = _bbl_invmutil!(EMU_star, EMU_star, θ[:ξ])

# Resources that lead to capital choice
# k'= c + m*(k") + k" - w*h*N
# = value of todays cap and money holdings
Resource = c_a_aux + m_a_aux + inc_IA[1, :, :] - inc_lab[1, :, :]

# Money constraint is not binding, but capital constraint is binding
m_star_zero     = m_a_aux[1, :] # Money holdings that correspond to k'=0:  m*(k=0)

# Use consumption at k"=0 from constrained problem, when m" is on grid()
aux_c           = reshape(c_star_n[:, 1, :], (n[1], n[3]))
aux_inc         = reshape(inc_lab[1, 1, :], (1, n[3]))
cons_list       = Array{Array{eltype(c_star_n)}}(undef, n[3], 1) # consumption
res_list        = Array{Array{eltype(c_star_n)}}(undef, n[3], 1) # resources
mon_list        = Array{Array{eltype(c_star_n)}}(undef, n[3], 1) # liquid asset choice
cap_list        = Array{Array{eltype(c_star_n)}}(undef, n[3], 1) # capital choice

@inbounds @fastmath @simd for j = 1:n[3] # Iterate over income states
    # When choosing zero capital holdings, HHs might still want to choose money
    # holdings smaller than m*(k'=0)
    if m_star_zero[j] > m_grid[1]
        # Calculate consumption policies, when HHs chooses money holdings
        # lower than m*(k"=0) and capital holdings k"=0 and save them in cons_list
        log_index    = m_grid .< m_star_zero[j] # all indices of m grid points less than m*(k"=0)
        # aux_c is the consumption policy under no cap. adj. (fix k=0), for m<m_a*(k'=0)
        c_k_cons     = aux_c[log_index, j]
        cons_list[j] = c_k_cons # Consumption at k"=0, for all m"<m_a*(0), and income state zⱼ

        # Required Resources: Money choice + Consumption - labor income
        # => this step gets resources that lead to k"=0 and m'<m*(k"=0) when z = zⱼ
        res_list[j]  = m_grid[log_index] .+ c_k_cons .- aux_inc[j]
        mon_list[j]  = m_grid[log_index]
        cap_list[j]  = zeros(eltype(EVm), sum(log_index)) # a bunch of zeros b/c choosing zero capital
    else
        # optimal m* choice is lowest gridpoint on m_grid => constrained in m too, so
        # we postpone handling this case until further down
        cons_list[j] = zeros(eltype(EVm), 0) # Consumption at k"=0, m"<m_a*(0)

        # Required Resources: Money choice + Consumption - labor income
        # Resources that lead to k"=0 and m'<m*(k"=0)
        res_list[j]  = zeros(eltype(EVm), 0)
        mon_list[j]  = zeros(eltype(EVm), 0)
        cap_list[j]  = zeros(eltype(EVm), 0)
    end
end

# Merge lists.
# Recall that k ranges from 0 to some upper limit,
# so when we calculate c_a_aux, Resource, and m_a_aux,
# these are the consumption, resources, and money holdings
# according to the policy function guesses,
# given k > 0, income state z, and m*, where m* is the
# money holding that gives the same expected marginal utility
# as the capital state k.
c_a_aux             = reshape(c_a_aux, (n[2], n[3]))
m_a_aux             = reshape(m_a_aux, (n[2], n[3]))

@inbounds for j = 1:n[3]
    cons_list[j]    = append!(cons_list[j], c_a_aux[:, j]) # c_a_aux[:, j] is consumption over all points on k_grid, fixing income state zⱼ
    res_list[j]     = append!(res_list[j],  Resource[:, j])
    mon_list[j]     = append!(mon_list[j],  m_a_aux[:, j])
    cap_list[j]     = append!(cap_list[j],  k_grid)
end
## TODO: Doesn't this repeat values?

####################################################################
## EGM Step 4: Interpolate back to fixed grid                     ##
####################################################################
Resource_grid       = reshape(inc_IA + inc_LA + inc_rent, (n[1] * n[2], n[3])) # resources according to income
labor_inc_grid      = vec(inc_lab[1, 1, :])

@views @inbounds @fastmath begin
    @simd for j = 1:n[3]
        # Check monotonicity of resources
        if warnme
            if any(x -> x < 0, diff(res_list[j]))
                @warn "non monotone resource list encountered"
            end
        end

        # Find when at most one constraint binds, which is when the required resources # log_index2 is supposed to be for at most 1 constraint
        # for the desired choice of (m, k), given zⱼ, exceeds implied income (see construction of Resource_grid) # but that doesn't seem to actually
        # log_index2 = reshape(Resource_grid[:, j], n[1] * n[2]) .< res_list[j][1] # be when it's used (see below), so . . . what's going on?
        # Note that res_list[j][1] corresponds to m_a"=0 and k_a"=0.
        log_index2 = Resource_grid[:, j] .< res_list[j][1] # so this is when the available resources are below what you'd need to exactly choose m = k = 0 when unconstrained?

        # cons_list[j], mon_list[j], cap_list[j] are vectors over res_list[j] -> form interpolation using res_list[j]
        # and then evaluate over Resource_grid[:, j] using interpolation.
        c_a_star1, m_a_star1, k_a_star1 = mylinearinterpolate_mult3(res_list[j], cons_list[j], mon_list[j], cap_list[j], Resource_grid[:, j])

        # Any resources on grid smaller then res_list imply that HHs consume all
        # resources plus income => choose m = k = 0
        # When both constraints are binding:
        c_a_star1[log_index2]  = Resource_grid[log_index2, j] .+ labor_inc_grid[j] .- m_grid[1]
        m_a_star1[log_index2] .= m_grid[1]
        k_a_star1[log_index2] .= 0.0

        # Update consumption policy to be back on grid and
        # check if policies go beyond largest grid points
        @simd for kk = 1:n[2]
            @simd for mm = 1:n[1]
                runind                  = mm + (kk-1) * n[1] # linear indexing
                mp                      = m_a_star1[runind]  # liquid asset saving
                kp                      = k_a_star1[runind]  # illiquid asset saving
                c_a_star[mm, kk, j]     = c_a_star1[runind]  # consumption policy

                if mp < mmax
                    m_a_star[mm, kk, j] = mp
                else
                    m_a_star[mm, kk, j] = mmax
                end
                if kp < kmax
                    k_a_star[mm, kk, j] = kp
                else
                    k_a_star[mm, kk, j] = kmax
                end
            end
        end
    end
end

    return c_a_star, m_a_star, k_a_star, c_n_star, m_n_star
end

# parallel version of EGM_policyupdate! to make the serial code cleaner
function _parallel_EGM_policyupdate!(EVm::Array,
                                    EVk::Array,
                                    Qminus::Real,
                                    πminus::Real,
                                    RBminus::Real,
                                    Tshock::Real,
                                    inc::Array,
                                    θ::NamedTuple,
                                    grids::OrderedDict,
                                    warnme::Bool,
                                    c_a_star::Array,
                                    m_a_star::Array,
                                    k_a_star::Array,
                                    c_n_star::Array,
                                    m_n_star::Array)

    ################### Copy/read-out stuff#####################################
    β::Float64 = θ[:β]
    borrwedge  = θ[:Rbar] .* Tshock
    # inc[1] = labor income , inc[2] = rental income,
    # inc[3]= liquid assets income, inc[4] = capital liquidation income
    inc_lab    = inc[1] # TODO: do we need to also sign the types of inc?
    inc_rent   = inc[2]
    inc_LA     = inc[3]
    inc_IA     = inc[4]
    n          = size(EVm)
    m_grid     = get_gridpts(grids, :m_grid)::Vector{Float64} # type declarations necessary b/c grids is an OrderedDict =>
    k_grid     = get_gridpts(grids, :k_grid)::Vector{Float64} # ensures type stability, or else unnecessary allocations are made
    m_ndgrid   = grids[:m_ndgrid]::Array{Float64, 3}
    mmax       = m_grid[end]
    kmax       = k_grid[end]

    ############################################################################
    ## EGM Step 1: Find optimal liquid asset holdings in the constrained case ##
    ############################################################################
    EMU         = EVm .* β
    c_star_n    = _bbl_invmutil(EMU, θ[:ξ]) # 6% of time with rolled out power function

    # Calculate assets consistent with choices being [m']
    # Calculate initial money position from the budget constraint
    # that leads to the optimal consumption choice
    m_star_n    = c_star_n + m_ndgrid - inc_lab - inc_rent

    # Apply correct interest rate
    m_star_n   ./= ((RBminus .+ borrwedge .* (m_star_n .< 0)) ./ πminus)  # apply borrowing rate

    # Next step: Interpolate w_guess and c_guess from new k-grids
    # using c[s,h,m"], m(s,h,m")
    # Interpolate grid().m and c_n_aux defined on m_star_n over grid().m

    # Check monotonicity of m_star_n
    if warnme
        m_star_aux = reshape(m_star_n, (n[1], n[2] * n[3]))
        if any(x -> x < 0, diff(m_star_aux, dims = 1))
            @warn "non monotone future liquid asset choice encountered"
        end
    end

    # Policies for tuples (c*,m*,y) are now given. Need to interpolate to return to
    # fixed grid. Note that c_n_star & m_n_star are the policies when HH cannot adjust
    @inbounds @views @fastmath begin
        Threads.@threads for kkjj in CartesianIndices((n[2], n[3]))
            kk, jj = kkjj[1], kkjj[2] # Loop over capital (kk) and income (jj) states
            cc, mn = mylinearinterpolate_mult2(m_star_n[:, kk, jj], c_star_n[:, kk, jj], m_grid, m_grid)
            c_n_star[:, kk, jj] = cc
            m_n_star[:, kk, jj] = mn

            # Check for binding borrowing constraints, no extrapolation from grid
            bcpol = m_star_n[1, kk, jj]
            @simd for mm = 1:n[1]
                # this loop should be fairly fast, so we don't bother with using threads.
                # The reason is that, since this loop depends on the interpolation above,
                # we would need to use `Threads.@spawn` and one of the tools available for
                # ensuring this loop does not run until the interpolation above is finished.
                if m_ndgrid[mm, kk, jj] < bcpol
                    c_n_star[mm, kk, jj] = inc_lab[mm, kk, jj] + inc_rent[mm, kk, jj] + inc_LA[mm, kk, jj] - m_grid[1]
                    m_n_star[mm, kk, jj] = m_grid[1]
                end
                if mmax < m_n_star[mm, kk, jj]
                    m_n_star[mm, kk, jj] = mmax
                end
            end
        end
    end
    #-------------------------END OF STEP 1-----------------------------

    ############################################################################
    ## EGM Step 2: Find Optimal Portfolio Combinations                        ##
    ############################################################################
    # Find an m_a* for given k' that yield the same expected future marginal value
    # for liquid and illiquid assets:
    term1           = (β / Qminus) * EVk                    # expected marginal value of illiquid investment
    E_return_diff   = term1 - EMU                           # difference conditional on future asset holdings on grid
    m_a_aux1        = Fastroot(m_grid, E_return_diff)       # Find indifferent m by interpolation of two neighboring points a, b ∈ grid_m with: E_return_diff(a) < 0 < E_return_diff(b)
    # (Fastroot does not allow for extrapolation and uses non-negativity constraint and monotonicity)
    m_a_aux         = reshape(m_a_aux1, (n[2], n[3]))

    ###########################################################################
    ## EGM Step 3: Constraints for money and capital are not binding         ##
    ###########################################################################
    # Interpolation of psi()-function at m*_n[m,k]
    aux_index       = (0:(n[2] * n[3]) - 1) * n[1]                  # auxiliary to move to linear indexing
    EMU_star        = Matrix{eltype(m_a_aux)}(undef, n[2], n[3])    # container (note it is over capital and income)
    step            = diff(m_grid)                                  # Stepsize on grid()

    # Interpolate EMU[m",k',s'*h',M',K'] over m*_n[k"], m-dim is dropped # TODO: figure out exactly what EMU is. Are we using EVk = EVm or something like it?
    @inbounds @fastmath Threads.@threads for j in eachindex(m_a_aux)
        xi          = m_a_aux[j]

        # find indexes on grid next smallest to optimal policy
        if xi > m_grid[n[1] - 1]                                    # policy is larger than highest grid point
            idx     = n[1] - 1
        elseif xi <= m_grid[1]                                      # policy is smaller than lowest grid point
            idx     = 1
        else
            idx     = locate(xi, m_grid)                            # use exponential search to find grid point closest to policy (next smallest)
        end

        s           = (xi - m_grid[idx]) / step[idx]                # Distance of optimal policy to next grid point to get convex weights

        EMU_star[j] = EMU[idx + aux_index[j]] * (1.0 - s) +         # linear interpolation to populate EMU using s as a convex weight
        s * (EMU[idx + aux_index[j] + 1])
    end
c_a_aux         = _bbl_invmutil!(EMU_star, EMU_star, θ[:ξ]) # TODO: make this inplace

# Resources that lead to capital choice
# k'= c + m*(k") + k" - w*h*N
# = value of todays cap and money holdings
Resource = c_a_aux + m_a_aux + inc_IA[1, :, :] - inc_lab[1, :, :]

# Money constraint is not binding, but capital constraint is binding
m_star_zero     = m_a_aux[1, :] # Money holdings that correspond to k'=0:  m*(k=0)

# Use consumption at k"=0 from constrained problem, when m" is on grid()
aux_c           = reshape(c_star_n[:, 1, :], (n[1], n[3]))
aux_inc         = reshape(inc_lab[1, 1, :], (1, n[3]))
cons_list       = Array{Array{eltype(c_star_n)}}(undef, n[3], 1) # consumption
res_list        = Array{Array{eltype(c_star_n)}}(undef, n[3], 1) # resources
mon_list        = Array{Array{eltype(c_star_n)}}(undef, n[3], 1) # liquid asset choice
cap_list        = Array{Array{eltype(c_star_n)}}(undef, n[3], 1) # capital choice

 @fastmath @inbounds Threads.@threads for j = 1:n[3] # Iterate over income states
    # When choosing zero capital holdings, HHs might still want to choose money
    # holdings smaller than m*(k'=0)
    if m_star_zero[j] > m_grid[1]
        # Calculate consumption policies, when HHs chooses money holdings
        # lower than m*(k"=0) and capital holdings k"=0 and save them in cons_list
        log_index    = m_grid .< m_star_zero[j] # all indices of m grid points less than m*(k"=0)
        # aux_c is the consumption policy under no cap. adj. (fix k=0), for m<m_a*(k'=0)
        c_k_cons     = aux_c[log_index, j]
        cons_list[j] = c_k_cons # Consumption at k"=0, for all m"<m_a*(0), and income state zⱼ

        # Required Resources: Money choice + Consumption - labor income
        # => this step gets resources that lead to k"=0 and m'<m*(k"=0) when z = zⱼ
        res_list[j]  = m_grid[log_index] .+ c_k_cons .- aux_inc[j]
        mon_list[j]  = m_grid[log_index]
        cap_list[j]  = zeros(eltype(EVm), sum(log_index)) # a bunch of zeros b/c choosing zero capital
    else
        # optimal m* choice is lowest gridpoint on m_grid => constrained in m too, so
        # we postpone handling this case until further down
        cons_list[j] = zeros(eltype(EVm), 0) # Consumption at k"=0, m"<m_a*(0)

        # Required Resources: Money choice + Consumption - labor income
        # Resources that lead to k"=0 and m'<m*(k"=0)
        res_list[j]  = zeros(eltype(EVm), 0)
        mon_list[j]  = zeros(eltype(EVm), 0)
        cap_list[j]  = zeros(eltype(EVm), 0)
    end
end

# Merge lists.
# Recall that k ranges from 0 to some upper limit,
# so when we calculate c_a_aux, Resource, and m_a_aux,
# these are the consumption, resources, and money holdings
# according to the policy function guesses,
# given k > 0, income state z, and m*, where m* is the
# money holding that gives the same expected marginal utility
# as the capital state k.
c_a_aux             = reshape(c_a_aux, (n[2], n[3]))
m_a_aux             = reshape(m_a_aux, (n[2], n[3]))

@inbounds Threads.@threads for j = 1:n[3] # thread-safe b/c each loop is independent of others, even though we're calling append!
    cons_list[j]    = append!(cons_list[j], c_a_aux[:, j]) # c_a_aux[:, j] is consumption over all points on k_grid, fixing income state zⱼ
    res_list[j]     = append!(res_list[j],  Resource[:, j])
    mon_list[j]     = append!(mon_list[j],  m_a_aux[:, j])
    cap_list[j]     = append!(cap_list[j],  k_grid)
end

####################################################################
## EGM Step 4: Interpolate back to fixed grid                     ##
####################################################################
Resource_grid       = reshape(inc_IA + inc_LA + inc_rent, (n[1] * n[2], n[3])) # resources according to income
labor_inc_grid      = vec(inc_lab[1, 1, :])

@fastmath @views @inbounds begin
    @Threads.threads for j = 1:n[3]
        # Check monotonicity of resources
        if warnme
            if any(x -> x < 0, diff(res_list[j]))
                @warn "non monotone resource list encountered"
            end
        end

        # Find when at most one constraint binds, which is when the required resources # log_index2 is supposed to be for at most 1 constraint
        # for the desired choice of (m, k), given zⱼ, exceeds implied income (see construction of Resource_grid) # but that doesn't seem to actually
        # log_index2 = reshape(Resource_grid[:, j], n[1] * n[2]) .< res_list[j][1] # be when it's used (see below), so . . . what's going on?
        # Note that res_list[j][1] corresponds to m_a"=0 and k_a"=0.
        log_index2 = Resource_grid[:, j] .< res_list[j][1] # so this is when the available resources are below what you'd need to exactly choose m = k = 0 when unconstrained?

        # cons_list[j], mon_list[j], cap_list[j] are vectors over res_list[j] -> form interpolation using res_list[j]
        # and then evaluate over Resource_grid[:, j] using interpolation.
        c_a_star1, m_a_star1, k_a_star1 = mylinearinterpolate_mult3(res_list[j], cons_list[j], mon_list[j], cap_list[j], Resource_grid[:, j])

        # Any resources on grid smaller then res_list imply that HHs consume all
        # resources plus income => choose m = k = 0
        # When both constraints are binding:
        c_a_star1[log_index2]  = Resource_grid[log_index2, j] .+ labor_inc_grid[j] .- m_grid[1]
        m_a_star1[log_index2] .= m_grid[1]
        k_a_star1[log_index2] .= 0.0

        # Update consumption policy to be back on grid and
        # check if policies go beyond largest grid points
        @simd for kk = 1:n[2]     # this nested loop shouldn't be so slow that it's worth
            @simd for mm = 1:n[1] # going through the hassle of nested multi-threading
                runind                  = mm + (kk-1) * n[1] # linear indexing
                mp                      = m_a_star1[runind]  # liquid asset saving
                kp                      = k_a_star1[runind]  # illiquid asset saving
                c_a_star[mm, kk, j]     = c_a_star1[runind]  # consumption policy

                if mp < mmax
                    m_a_star[mm, kk, j] = mp
                else
                    m_a_star[mm, kk, j] = mmax
                end
                if kp < kmax
                    k_a_star[mm, kk, j] = kp
                else
                    k_a_star[mm, kk, j] = kmax
                end
            end
        end
    end
end

    return c_a_star, m_a_star, k_a_star, c_n_star, m_n_star
end
