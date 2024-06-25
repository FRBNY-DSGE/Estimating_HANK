#hank
# TODO: add an option that constructs nt and id only once rather than repeatedly
using Debugger, CSV, Tables
"""
```
jacobian(m::BayerBornLuetticke)
```
compute the Jacobians of the non-linear difference equations defined
by the functions `Fsys` and `Fsys_agg` using the package `ForwardDiff`.

If the user only wants to update the aggregate block of the Jacobians
without changing the heterogeneous block, then the user
should set `m <= Setting(:linearize_heterogeneous_block, false)`.
In this case, the Jacobians are already stored in `m` and
are updated in place.

### Outputs
- `A::Matrix`,`B::Matrix`: first derivatives of `Fsys` with respect to arguments `X` [`B`] and
    `XPrime` [`A`]
"""
@inline function jacobian(m::BayerBornLuetticke{T}) where {T <: Real}

    if get_setting(m, :replicate_original_output)::Bool
        # This block replicates output from the original implementation by Bayer, Born, and Luetticke
        if get_setting(m, :linearize_heterogeneous_block)::Bool
            return _original_jacobian!(m)
        else
            #return _original_update_aggregate_jacobian!(m, get_untransformed_values(m[:A])::Matrix{T},
                                                       # get_untransformed_values(m[:B])::Matrix{T})
            return _original_update_aggregate_jacobian!(m, m[:A].value::Matrix{T},m[:B].value::Matrix{T})
        end
    else
        if get_setting(m, :linearize_heterogeneous_block)::Bool
            # Compute the Jacobian from scratch (assumes steadystate!(m) has already been called)
             m <= Setting(:linearize_heterogeneous_block, false)
            return _jacobian!(m)
        else
            # Note that since get_untransformed_values(m[:A]) = m[:A].value,
            # the A matrix in m[:A] is being directly updated
            if isempty(m[:A]) || isempty(m[:B])
                return _jacobian!(m)
            else

                return _update_aggregate_jacobian!(m, get_untransformed_values(m[:A])::Matrix{T},
                                               get_untransformed_values(m[:B])::Matrix{T})
            end


          #= try
                result =  _update_aggregate_jacobian!(m, get_untransformed_values(m[:A])::Matrix{T},
                                               get_untransformed_values(m[:B])::Matrix{T})
            catch e
                result = _jacobian!(m)
            end

            return result =#

           #= return _update_aggregate_jacobian!(m, get_untransformed_values(m[:A])::Matrix{T},
                                               get_untransformed_values(m[:B])::Matrix{T})
                =#
        end
    end
end

function _jacobian!(m::BayerBornLuetticke)

    # Information needed from m for set up
    θ = parameters2namedtuple(m)
    nt = construct_steadystate_namedtuple(m) # see helper_functions/steady_state/prepare_linearization.jl
    # id = construct_prime_and_noprime_indices(m; only_aggregate = false)
    id = get_setting(m, :prime_and_noprime_indices)::OrderedDict{Symbol, UnitRange{Int}}
    nm, nk, ny = get_idiosyncratic_dims(m)


    ############################################################################
    # Prepare elements used for uncompression
    ############################################################################
    # Matrices to take care of reduced degree of freedom in marginal distributions
    Γ  = shuffle_matrix(m[:distr_star])

    # Matrices for discrete cosine transforms
    #DC = Vector{Array{Float64, 2}}(undef, 3)
    DC = Vector{Array{Float64, 2}}(undef, 3)
    DC[1]  = mydctmx(nm)
    DC[2]  = mydctmx(nk)
    DC[3]  = mydctmx(ny)

    IDC    = [DC[1]', DC[2]', DC[3]'] # TODO: why do we need to take the transpose?

    DCD = Vector{Array{Float64, 2}}(undef, 3)
    # DCD = Vector{Array{Float64, 2}}(undef, 3)
    ## commenting out fixed approach for dct copula
    #n_copula = get_setting(m, :n_copula_dct_coefficients)
    n_copula = length(get_setting(m,:dct_compression_indices)[:copula])

    nm_copula = get_setting(m,:nm_copula)
    nk_copula = get_setting(m,:nk_copula)
    ny_copula = get_setting(m,:ny_copula)
    DCD[1] = mydctmx(nm_copula)
    DCD[2] = mydctmx(nk_copula)
    DCD[3] = mydctmx(ny_copula)
#=
    DCD[1]  = mydctmx(nm-1)
    DCD[2]  = mydctmx(nk-1)
    DCD[3]  = mydctmx(ny-1)
=#
    IDCD    = [DCD[1]', DCD[2]', DCD[3]']

#=
  DCD[1]  = mydctmx(n_copula)
    DCD[2]  = mydctmx(n_copula)
    DCD[3]  = mydctmx(n_copula)
    IDCD    = [DCD[1]', DCD[2]', DCD[3]']=#


    ############################################################################
    # Check whether steady state solves the difference equation
    # (left here in case the user ever wants to check)
    ############################################################################
    # X0 = zeros(get_setting(m, :n_model_states)) .+ ForwardDiff.Dual(0.0,tuple(zeros(5)...))
    # F  = Fsys(X0, X0, θ, m.grids, id, nt, m.equilibrium_conditions,
    #           get_setting(m, :dct_compression_indices), Γ, DC, IDC, DCD, IDCD)
    # if maximum(abs.(F)) / 10 > get_setting(m, :ϵ)
    #     @warn  "F = 0 is not at required precision"
    # end

    ############################################################################
    # Calculate Jacobians of the Difference equation F
    ############################################################################
    length_X0   = get_setting(m, :n_model_states)::Int
    n_dct_Vm    = length(get_setting(m, :dct_compression_indices)[:Vm]::Vector{Int})
    n_dct_Vk    = length(get_setting(m, :dct_compression_indices)[:Vk]::Vector{Int})
    n_marginals = length(id[:marginal_pdf_y_t]) + length(id[:marginal_pdf_m_t]) + length(id[:marginal_pdf_k_t])
    nxB         = length_X0 - n_dct_Vm - n_dct_Vk
    nxA         = length_X0 - n_marginals
    n_vars      = n_model_states(m)


    # The objective function omits the Vm, Vk in the X vector, but we left off at index id[:Vm_t][1] - 1,
    # so after we use zeros for the Vm, Vk parts of X, we need to start from id[:Vm_t][1]. We then go
    # to nxB since that is the total number of X elements we want to perturb.
    # For XPrime, we ignore the marginal perturbations and then have to start at nxB + 1 since
    # x is a vector of length nxB + nxA.
    BA = zeros(n_vars, nxB + nxA)

    obj_fnct    = (F, x) -> Fsys(F, [x[1:id[:Vm_t][1]-1]; Zeros(n_dct_Vm + n_dct_Vk); x[id[:Vm_t][1]:nxB]],
                                 [Zeros(n_marginals); x[nxB+1:end]],
                                 θ, m.grids, id, nt, m.equilibrium_conditions,
                                 get_setting(m, :dct_compression_indices), Γ, DC, IDC, DCD, IDCD,m)
    @show n_vars



    ForwardDiff.jacobian!(BA, obj_fnct, zeros(n_vars), zeros(nxB+nxA))

    A      = zeros(n_vars, n_vars)
    B      = zeros(n_vars, n_vars)

    B[:,1:id[:Vm_t][1]-1]                 = BA[:,1:id[:Vm_t][1]-1]
    B[:,id[:Vk_t][end]+1:end]             = BA[:,id[:Vm_t][1]:nxB]
    A[:,id[:marginal_pdf_y_t][end]+1:end] = BA[:,nxB+1:end]

    # Make use of the fact that Vk/Vm has no influence on any variable in
    # the system, thus derivative is 1
    for (i, j) in zip(m.equilibrium_conditions[:eq_marginal_value_bonds], id[:Vm_t])
        B[i, j] = 1.0
    end
    for (i, j) in zip(m.equilibrium_conditions[:eq_marginal_value_capital], id[:Vk_t])
        B[i, j] = 1.0
    end

    # Make use of the fact that future distribution has no influence on any variable in
    # the system, thus derivative is Γ
    for (count, i) in enumerate(id[:marginal_pdf_m_t])
        A[id[:marginal_pdf_m_t],i] = -Γ[1][1:end-1,count]
    end
    for (count, i) in enumerate(id[:marginal_pdf_k_t])
        A[id[:marginal_pdf_k_t],i] = -Γ[2][1:end-1,count]
    end
    for (count, i) in enumerate(id[:marginal_pdf_y_t])
        A[id[:marginal_pdf_y_t],i] = -Γ[3][1:end-1,count]
    end

    # Store A and B Jacobians
    m[:A] = A
    m[:B] = B

    if get_setting(m, :save_jacobian)
        save_jacobian(m)
    end

    return A, B
end

function _update_aggregate_jacobian!(m::BayerBornLuetticke{T}, A::Matrix{T}, B::Matrix{T}) where {T <: Real}

    # Information needed from m for set up
    θ = parameters2namedtuple(m)
    nt = construct_steadystate_namedtuple(m) # see helper_functions/steady_state/prepare_linearization.jl
    # id = construct_prime_and_noprime_indices(m; only_aggregate = true)
    # θ  = get_setting(m, :parameters_namedtuple)::NamedTuple
    # nt = get_setting(m, :steadystate_namedtuple)::NamedTuple
    id = get_setting(m, :prime_and_noprime_aggregate_indices)::OrderedDict{Symbol, Int}

    # Get index info
    eqconds          = m.equilibrium_conditions
    endo_states      = m.endogenous_states
    aggr_eqconds     = get_aggregate_equilibrium_conditions(m)
    aggr_endo_states = get_aggregate_endogenous_states(m)

    # We want to exclude aggregate equilibrium conditions/states that
    # are still distributional, e.g. Gini Coefficients
    aggr_eqconds_excl_distr     = deepcopy(aggr_eqconds)
    aggr_endo_states_excl_distr = deepcopy(aggr_endo_states)
    for k in [:eq_Gini_C, :eq_Gini_X, :eq_sd_log_y, :eq_I90_share, :eq_I90_share_net, :eq_W90_share]
        pop!(aggr_eqconds_excl_distr, k)
    end
    for k in [:Gini_C′_t, :Gini_X′_t, :sd_log_y′_t, :I90_share′_t, :I90_share_net′_t, :W90_share′_t]
        pop!(aggr_endo_states_excl_distr, k)
    end

    ############################################################################
    # Calculate derivatives of non-linear difference equation
    ############################################################################

    length_X0                  = length(aggr_eqconds) # num. aggregate variables = num. equilibrium conditions

    # Sparsity differentiation settings
    use_sparse_jac             = haskey(get_settings(m), :use_sparse_jacobian) && get_setting(m, :use_sparse_jacobian)
    has_sparsity_pattern       = haskey(get_settings(m), :sparsity_pattern)

    obj_fnct = (F, x) -> Fsys_agg(F, x[1:length_X0], x[length_X0+1:end], θ,
                                  m.grids, id, nt, aggr_eqconds)



    if use_sparse_jac && has_sparsity_pattern

        sparsity_pattern = get_setting(m, :sparsity_pattern)::SparseMatrixCSC{T,Int}
        colorvec = haskey(get_settings(m), :colorvec) ? get_setting(m, :colorvec)::Vector{Int} : matrix_colors(sparsity_pattern)

        if haskey(get_settings(m), :sparse_aggregate_block_jacobian)

            BA = get_setting(m, :sparse_aggregate_block_jacobian)::SparseMatrixCSC{T,Int}
            input = get_setting(m, :aggregate_block_jacobian_input)::Vector{T}
            output = get_setting(m, :aggregate_block_jacobian_output)::Vector{T}
        else

            BA = similar(sparsity_pattern)
            input = zeros(T, 2 * length_X0)
            output = similar(input, length_X0)
            m <= Setting(:sparse_aggregate_block_jacobian, BA)
            m <= Setting(:aggregate_block_jacobian_input, input)
            m <= Setting(:aggregate_block_jacobian_output, output)
        end

        forwarddiff_color_jacobian!(BA, obj_fnct, input; dx = output,
                                    colorvec = colorvec, sparsity = sparsity_pattern)
    else
        if use_sparse_jac

            warn_str = "No sparsity pattern provided, so a dense Jacobian will be computed via ForwardDiff" *
                " instead of a sparse Jacobian via SparseDiffTools"
            @warn warn_str
        end

        if haskey(get_settings(m), :aggregate_block_jacobian)
           #this is the only block that runs
            BA = get_setting(m, :aggregate_block_jacobian)::Matrix{T}
            input = get_setting(m, :aggregate_block_jacobian_input)::Vector{T}
            output = get_setting(m, :aggregate_block_jacobian_output)::Vector{T}
        else

            BA = Matrix{T}(undef, length_X0, 2 * length_X0)
            input = zeros(T, 2 * length_X0)
            output = similar(input, length_X0)
            m <= Setting(:aggregate_block_jacobian, BA)
            m <= Setting(:aggregate_block_jacobian_input, input)
            m <= Setting(:aggregate_block_jacobian_output, output)
        end
        ForwardDiff.jacobian!(BA, obj_fnct, output, input)
    end

    @show size(BA)
    Aa          = BA[:, length_X0+1:end] # aggregate A       # to create the required Jacobian sparsity pattern
    Ba          = BA[:, 1:length_X0]     # aggregate B

    # Update Jacobians of equilibrium conditions w.r.t. aggregate variables,
    # excluding aggregates that are distributional in nature (e.g. Gini coefficients)

    for (aggr_endo_state_name, j) in aggr_endo_states_excl_distr # endo states are the columns
        _j = first(endo_states[aggr_endo_state_name])
        for (aggr_eqcond_name, i) in aggr_eqconds_excl_distr # eqconds are the rows
            # So the following line populates A in column major order
            _i = first(eqconds[aggr_eqcond_name])
            A[_i, _j] = Aa[i, j]
            B[_i, _j] = Ba[i, j]
        end
    end

    return A, B
end

function _original_jacobian!(m::BayerBornLuetticke)

    # Information needed from m for set up
    θ = parameters2namedtuple(m)
    nt = construct_steadystate_namedtuple(m) # see helper_functions/steady_state/prepare_linearization.jl
    id = construct_prime_and_noprime_indices(m; only_aggregate = false)
    nm, nk, ny = get_idiosyncratic_dims(m)

    ############################################################################
    # Prepare elements used for uncompression
    ############################################################################
    # Matrices to take care of reduced degree of freedom in marginal distributions
    Γ  = shuffle_matrix(m[:distr_star])

    # Matrices for discrete cosine transforms
    DC = Vector{Array{Float64, 2}}(undef, 3)
    DC[1]  = mydctmx(nm)
    DC[2]  = mydctmx(nk)
    DC[3]  = mydctmx(ny)
    IDC    = [DC[1]', DC[2]', DC[3]'] # TODO: why do we need to take the transpose?

    DCD = Vector{Array{Float64, 2}}(undef, 3)
   #= DCD[1]  = mydctmx(nm-1)
    DCD[2]  = mydctmx(nk-1)
    DCD[3]  = mydctmx(ny-1)
=#

    nm_copula = get_setting(m,:nm_copula)
    nk_copula = get_setting(m,:nk_copula)
    ny_copula = get_setting(m,:ny_copula)
    DCD[1] = mydctmx(nm_copula)
    DCD[2] = mydctmx(nk_copula)
    DCD[3] = mydctmx(ny_copula)
#=
    DCD[1]  = mydctmx(nm-1)
    DCD[2]  = mydctmx(nk-1)
    DCD[3]  = mydctmx(ny-1)
=#
    IDCD    = [DCD[1]', DCD[2]', DCD[3]']

#=
    DCD[1]  = mydctmx(n_copula)
    DCD[2]  = mydctmx(n_copula)
    DCD[3]  = mydctmx(n_copula)
    IDCD    = [DCD[1]', DCD[2]', DCD[3]']=#


    ############################################################################
    # Check whether steady state solves the difference equation
    # (left here in case the user ever wants to check)
    ############################################################################
    # X0 = zeros(get_setting(m, :n_model_states)) .+ ForwardDiff.Dual(0.0,tuple(zeros(5)...))
    # F  = Fsys(X0, X0, θ, m.grids, id, nt, m.equilibrium_conditions,
    #           get_setting(m, :dct_compression_indices), Γ, DC, IDC, DCD, IDCD)
    # if maximum(abs.(F)) / 10 > get_setting(m, :ϵ)
    #     @warn  "F = 0 is not at required precision"
    # end

    ############################################################################
    # Calculate Jacobians of the Difference equation F
    ############################################################################
    length_X0   = get_setting(m, :n_model_states)::Int
    n_dct_Vm    = length(get_setting(m, :dct_compression_indices)[:Vm]::Vector{Int})
    n_dct_Vk    = length(get_setting(m, :dct_compression_indices)[:Vk]::Vector{Int})
    n_marginals = length(id[:marginal_pdf_y_t]) + length(id[:marginal_pdf_m_t]) + length(id[:marginal_pdf_k_t])
    nxB         = length_X0 - n_dct_Vm - n_dct_Vk
    nxA         = length_X0 - n_marginals
    n_vars      = n_model_states(m)


    # The objective function omits the Vm, Vk in the X vector, but we left off at index id[:Vm_t][1] - 1,
    # so after we use zeros for the Vm, Vk parts of X, we need to start from id[:Vm_t][1]. We then go
    # to nxB since that is the total number of X elements we want to perturb.
    # For XPrime, we ignore the marginal perturbations and then have to start at nxB + 1 since
    # x is a vector of length nxB + nxA.
    BA = zeros(n_vars, nxB + nxA)

    obj_fnct    = (F, x) -> Fsys(F, [x[1:id[:Vm_t][1]-1]; Zeros(n_dct_Vm + n_dct_Vk); x[id[:Vm_t][1]:nxB]],
                                 [Zeros(n_marginals); x[nxB+1:end]],
                                 θ, m.grids, id, nt, m.equilibrium_conditions,
                                 get_setting(m, :dct_compression_indices), Γ, DC, IDC, DCD, IDCD,m)
    @show n_vars



    ForwardDiff.jacobian!(BA, obj_fnct, zeros(n_vars), zeros(nxB+nxA))

    A      = zeros(n_vars, n_vars)
    B      = zeros(n_vars, n_vars)

    B[:,1:id[:Vm_t][1]-1]                 = BA[:,1:id[:Vm_t][1]-1]
    B[:,id[:Vk_t][end]+1:end]             = BA[:,id[:Vm_t][1]:nxB]
    A[:,id[:marginal_pdf_y_t][end]+1:end] = BA[:,nxB+1:end]

    # Make use of the fact that Vk/Vm has no influence on any variable in
    # the system, thus derivative is 1
    for (i, j) in zip(m.equilibrium_conditions[:eq_marginal_value_bonds], id[:Vm_t])
        B[i, j] = 1.0
    end
    for (i, j) in zip(m.equilibrium_conditions[:eq_marginal_value_capital], id[:Vk_t])
        B[i, j] = 1.0
    end

    # Make use of the fact that future distribution has no influence on any variable in
    # the system, thus derivative is Γ
    for (count, i) in enumerate(id[:marginal_pdf_m_t])
        A[id[:marginal_pdf_m_t],i] = -Γ[1][1:end-1,count]
    end
    for (count, i) in enumerate(id[:marginal_pdf_k_t])
        A[id[:marginal_pdf_k_t],i] = -Γ[2][1:end-1,count]
    end
    for (count, i) in enumerate(id[:marginal_pdf_y_t])
        A[id[:marginal_pdf_y_t],i] = -Γ[3][1:end-1,count]
    end

    # Store A and B Jacobians
    m[:A] = A
    m[:B] = B

    if get_setting(m, :save_jacobian)
        save_jacobian(m)
    end

    return A, B
end

function _update_aggregate_jacobian!(m::BayerBornLuetticke{T}, A::Matrix{T}, B::Matrix{T}) where {T <: Real}

    # Information needed from m for set up
    θ = parameters2namedtuple(m)
    nt = construct_steadystate_namedtuple(m) # see helper_functions/steady_state/prepare_linearization.jl
    # id = construct_prime_and_noprime_indices(m; only_aggregate = true)
    # θ  = get_setting(m, :parameters_namedtuple)::NamedTuple
    # nt = get_setting(m, :steadystate_namedtuple)::NamedTuple
    id = get_setting(m, :prime_and_noprime_aggregate_indices)::OrderedDict{Symbol, Int}

    # Get index info
    eqconds          = m.equilibrium_conditions
    endo_states      = m.endogenous_states
    aggr_eqconds     = get_aggregate_equilibrium_conditions(m)
    aggr_endo_states = get_aggregate_endogenous_states(m)

    # We want to exclude aggregate equilibrium conditions/states that
    # are still distributional, e.g. Gini Coefficients
    aggr_eqconds_excl_distr     = deepcopy(aggr_eqconds)
    aggr_endo_states_excl_distr = deepcopy(aggr_endo_states)
    for k in [:eq_Gini_C, :eq_Gini_X, :eq_sd_log_y, :eq_I90_share, :eq_I90_share_net, :eq_W90_share]
        pop!(aggr_eqconds_excl_distr, k)
    end
    for k in [:Gini_C′_t, :Gini_X′_t, :sd_log_y′_t, :I90_share′_t, :I90_share_net′_t, :W90_share′_t]
        pop!(aggr_endo_states_excl_distr, k)
    end

    ############################################################################
    # Calculate derivatives of non-linear difference equation
    ############################################################################

    length_X0                  = length(aggr_eqconds) # num. aggregate variables = num. equilibrium conditions

    # Sparsity differentiation settings
    use_sparse_jac             = haskey(get_settings(m), :use_sparse_jacobian) && get_setting(m, :use_sparse_jacobian)
    has_sparsity_pattern       = haskey(get_settings(m), :sparsity_pattern)

    obj_fnct = (F, x) -> Fsys_agg(F, x[1:length_X0], x[length_X0+1:end], θ,
                                  m.grids, id, nt, aggr_eqconds)



    if use_sparse_jac && has_sparsity_pattern

        sparsity_pattern = get_setting(m, :sparsity_pattern)::SparseMatrixCSC{T,Int}
        colorvec = haskey(get_settings(m), :colorvec) ? get_setting(m, :colorvec)::Vector{Int} : matrix_colors(sparsity_pattern)

        if haskey(get_settings(m), :sparse_aggregate_block_jacobian)

            BA = get_setting(m, :sparse_aggregate_block_jacobian)::SparseMatrixCSC{T,Int}
            input = get_setting(m, :aggregate_block_jacobian_input)::Vector{T}
            output = get_setting(m, :aggregate_block_jacobian_output)::Vector{T}
        else

            BA = similar(sparsity_pattern)
            input = zeros(T, 2 * length_X0)
            output = similar(input, length_X0)
            m <= Setting(:sparse_aggregate_block_jacobian, BA)
            m <= Setting(:aggregate_block_jacobian_input, input)
            m <= Setting(:aggregate_block_jacobian_output, output)
        end

        forwarddiff_color_jacobian!(BA, obj_fnct, input; dx = output,
                                    colorvec = colorvec, sparsity = sparsity_pattern)
    else
        if use_sparse_jac

            warn_str = "No sparsity pattern provided, so a dense Jacobian will be computed via ForwardDiff" *
                " instead of a sparse Jacobian via SparseDiffTools"
            @warn warn_str
        end

        if haskey(get_settings(m), :aggregate_block_jacobian)
           #this is the only block that runs
            BA = get_setting(m, :aggregate_block_jacobian)::Matrix{T}
            input = get_setting(m, :aggregate_block_jacobian_input)::Vector{T}
            output = get_setting(m, :aggregate_block_jacobian_output)::Vector{T}
        else

            BA = Matrix{T}(undef, length_X0, 2 * length_X0)
            input = zeros(T, 2 * length_X0)
            output = similar(input, length_X0)
            m <= Setting(:aggregate_block_jacobian, BA)
            m <= Setting(:aggregate_block_jacobian_input, input)
            m <= Setting(:aggregate_block_jacobian_output, output)
        end
        ForwardDiff.jacobian!(BA, obj_fnct, output, input)
    end

    @show size(BA)
    Aa          = BA[:, length_X0+1:end] # aggregate A       # to create the required Jacobian sparsity pattern
    Ba          = BA[:, 1:length_X0]     # aggregate B

    # Update Jacobians of equilibrium conditions w.r.t. aggregate variables,
    # excluding aggregates that are distributional in nature (e.g. Gini coefficients)

    for (aggr_endo_state_name, j) in aggr_endo_states_excl_distr # endo states are the columns
        _j = first(endo_states[aggr_endo_state_name])
        for (aggr_eqcond_name, i) in aggr_eqconds_excl_distr # eqconds are the rows
            # So the following line populates A in column major order
            _i = first(eqconds[aggr_eqcond_name])
            A[_i, _j] = Aa[i, j]
            B[_i, _j] = Ba[i, j]
        end
    end


    return A, B
end

function _original_jacobian!(m::BayerBornLuetticke)

    # Information needed from m for set up
    θ = parameters2namedtuple(m)
    nt = construct_steadystate_namedtuple(m) # see helper_functions/steady_state/prepare_linearization.jl
    id = construct_prime_and_noprime_indices(m; only_aggregate = false)
    nm, nk, ny = get_idiosyncratic_dims(m)

    ############################################################################
    # Prepare elements used for uncompression
    ############################################################################
    # Matrices to take care of reduced degree of freedom in marginal distributions
    Γ  = shuffle_matrix(m[:distr_star])

    # Matrices for discrete cosine transforms
    DC = Vector{Array{Float64, 2}}(undef, 3)
    DC[1]  = mydctmx(nm)
    DC[2]  = mydctmx(nk)
    DC[3]  = mydctmx(ny)
    IDC    = [DC[1]', DC[2]', DC[3]'] # TODO: why do we need to take the transpose?

    DCD = Vector{Array{Float64, 2}}(undef, 3)
   #= DCD[1]  = mydctmx(nm-1)
    DCD[2]  = mydctmx(nk-1)
    DCD[3]  = mydctmx(ny-1)
=#

    nm_copula = get_setting(m,:nm_copula)
    nk_copula = get_setting(m,:nk_copula)
    ny_copula = get_setting(m,:ny_copula)
    DCD[1] = mydctmx(nm_copula)
    DCD[2] = mydctmx(nk_copula)
    DCD[3] = mydctmx(ny_copula)


    IDCD    = [DCD[1]', DCD[2]', DCD[3]']

    ############################################################################
    # Check whether steady state solves the difference equation
    # (left here in case the user ever wants to check)
    ############################################################################
    # X0 = zeros(get_setting(m, :n_model_states)) .+ ForwardDiff.Dual(0.0,tuple(zeros(5)...))
    # F  = Fsys(X0, X0, θ, m.grids, id, nt, m.equilibrium_conditions,
    #           get_setting(m, :dct_compression_indices), Γ, DC, IDC, DCD, IDCD)
    # if maximum(abs.(F)) / 10 > get_setting(m, :ϵ)
    #     @warn  "F = 0 is not at required precision"
    # end

    ############################################################################
    # Calculate Jacobians of the Difference equation F
    ############################################################################
    length_X0   = get_setting(m, :n_model_states)::Int
    n_dct_Vm    = length(get_setting(m, :dct_compression_indices)[:Vm]::Vector{Int})
    n_dct_Vk    = length(get_setting(m, :dct_compression_indices)[:Vk]::Vector{Int})
    n_marginals = length(id[:marginal_pdf_y_t]) + length(id[:marginal_pdf_m_t]) + length(id[:marginal_pdf_k_t])
    nxB         = length_X0 - n_dct_Vm - n_dct_Vk
    nxA         = length_X0 - n_marginals

    # The objective function omits the Vm, Vk in the X vector, but we left off at index id[:Vm_t][1] - 1,
    # so after we use zeros for the Vm, Vk parts of X, we need to start from id[:Vm_t][1]. We then go
    # to nxB since that is the total number of X elements we want to perturb.
    # For XPrime, we ignore the marginal perturbations and then have to start at nxB + 1 since
    # x is a vector of length nxB + nxA.
    obj_fnct    = x -> original_Fsys(x[1:length_X0], x[length_X0+1:end],
                                     θ, m.grids, id, nt, m.equilibrium_conditions,                                                                    get_setting(m, :dct_compression_indices), Γ, DC, IDC, DCD, IDCD, m)
    println("obj fnct norm zeros")
    println(norm(obj_fnct(zeros(2*length_X0))))

    # TODO: make Fsys in place? Would it make sense to make Fsys in place in general, particularly w.r.t Fsys_agg?
    # Relatedly, could we use a sparsity pattern to do the autodiffing so we don't need to do a bunch of copying
    # for SGU_estim and can just directly differentiate into the Jacobian by using the correct sparsity matrix?
    BA          = ForwardDiff.jacobian(obj_fnct, zeros(2*length_X0))

    B     = BA[:,1:length_X0]
    A     = BA[:,length_X0+1:end]
    # Store A and B Jacobians
    m[:A] = A
    m[:B] = B
    if get_setting(m,:load_bbl_posterior_mean)
        CSV.write(rawpath(m, "estimate", "DSGE_A_Mat_v3_Post_Mode.csv"),Tables.table(m[:A].value))
        CSV.write(rawpath(m, "estimate", "DSGE_B_Mat_v3_Post_Mode.csv"),Tables.table(m[:A].value))

    else
         CSV.write(rawpath(m, "estimate", "DSGE_A_Mat_v3_Prior_Mode.csv"),Tables.table(m[:A].value))
        CSV.write(rawpath(m, "estimate", "DSGE_B_Mat_v3_Prior_Mode.csv"),Tables.table(m[:A].value))
    end

    return A, B
end

function _original_update_aggregate_jacobian!(m::BayerBornLuetticke{T}, A::Matrix{T}, B::Matrix{T}) where {T <: Real}

    # Information needed from m for set up
    θ = parameters2namedtuple(m)
    nt = construct_steadystate_namedtuple(m) # see helper_functions/steady_state/prepare_linearization.jl
    id = construct_prime_and_noprime_indices(m; only_aggregate = true)

    # Get index info
    eqconds          = m.equilibrium_conditions
    endo_states      = m.endogenous_states
    aggr_eqconds     = get_aggregate_equilibrium_conditions(m)
    aggr_endo_states = get_aggregate_endogenous_states(m)

    # We want to exclude aggregate equilibrium conditions/states that
    # are still distributional, e.g. Gini Coefficients
    aggr_eqconds_excl_distr     = deepcopy(aggr_eqconds)
    aggr_endo_states_excl_distr = deepcopy(aggr_endo_states)
    for k in [:eq_Gini_C, :eq_Gini_X, :eq_sd_log_y, :eq_I90_share, :eq_I90_share_net, :eq_W90_share]
        pop!(aggr_eqconds_excl_distr, k)
    end
    for k in [:Gini_C′_t, :Gini_X′_t, :sd_log_y′_t, :I90_share′_t, :I90_share_net′_t, :W90_share′_t]
        pop!(aggr_endo_states_excl_distr, k)
    end

    ############################################################################
    # Calculate derivatives of non-linear difference equation
    ############################################################################

    length_X0   = length(aggr_eqconds) # num. aggregate variables = num. equilibrium conditions
    BA          = ForwardDiff.jacobian(x -> original_Fsys_agg(x[1:length_X0], x[length_X0+1:end], θ,
                                                              m.grids, id, nt, aggr_eqconds),
                                       zeros(2 * length_X0))
    Aa          = BA[:, length_X0+1:end] # aggregate A
    Ba          = BA[:, 1:length_X0]     # aggregate B

    # Update Jacobians of equilibrium conditions w.r.t. aggregate variables,
    # excluding aggregates that are distributional in nature (e.g. Gini coefficients)
    for (aggr_endo_state_name, j) in aggr_endo_states_excl_distr # endo states are the columns
        _j = first(endo_states[aggr_endo_state_name])
        for (aggr_eqcond_name, i) in aggr_eqconds_excl_distr # eqconds are the rows
            # So the following line populates A in column major order
            _i = first(eqconds[aggr_eqcond_name])
            A[_i, _j] = Aa[i, j]
            B[_i, _j] = Ba[i, j]
        end
    end

    return A, B
end
