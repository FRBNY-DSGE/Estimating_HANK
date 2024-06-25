## Access functions

# aggregate variable names
get_aggregate_state_variables(m::BayerBornLuetticke) = m.aggregate_state_variables
get_aggregate_jump_variables(m::BayerBornLuetticke) = m.aggregate_jump_variables
get_aggregate_endogenous_states(m::BayerBornLuetticke) = m.aggregate_endogenous_states
get_aggregate_equilibrium_conditions(m::BayerBornLuetticke) = m.aggregate_equilibrium_conditions
get_lagged_variables(m::BayerBornLuetticke) = m.state_variables[7:15] # 7 = Y′_tl1, 15 = τ_prog′_t1

# Idiosyncratic grid settings
@inline function get_idiosyncratic_dims(m::BayerBornLuetticke; coarse::Bool = false)
    if coarse
        (get_setting(m, :coarse_nm), get_setting(m, :coarse_nk), get_setting(m, :coarse_ny))
    else
        (get_setting(m, :nm), get_setting(m, :nk), get_setting(m, :ny))
    end
end
get_idiosyncratic_gridpts(m::BayerBornLuetticke) = (m.grids[:m_grid].points, m.grids[:k_grid].points, m.grids[:y_grid].points)
get_idiosyncratic_ndgrids(m::BayerBornLuetticke) = (m.grids[:m_ndgrid], m.grids[:k_ndgrid], m.grids[:y_ndgrid])

# Numbers of states when only tracking backward-looking states
n_states(m::BayerBornLuetticke) = haskey(get_settings(m), :klein_track_backward_looking_states_only) &&
    get_setting(m, :klein_track_backward_looking_states_only) ? n_backward_looking_states(m)::Int :
    sum(map(i -> length(collect(m.endogenous_states)[i][2]), 1:length(keys(m.endogenous_states))))

## Parsing functions

# Map lagged variable to steady state name
@inline function _bbl_parse_endogenous_states(var::Symbol)
    strvar = string(var)
    cutoff_i = if length(strvar) > 3 && strvar[end - 2:end] == "_t1"
        return Symbol(strvar[1:end - 2] * "l1_star")
    else
        return Symbol(strvar[1:end - 1] * "star")
    end
end

## Saving and loading output from steady state and Jacobian
@inline function save_steadystate(m::BayerBornLuetticke, KSS::T, VmSS::AbstractArray{T, 3},
                                  VkSS::AbstractArray{T, 3}, distrSS::AbstractArray{T, 3}) where {T <: Real}
    fp = get_setting(m, :steadystate_output_file)
    if !ispath(dirname(fp))
        mkpath(dirname(fp))
    end
    JLD2.jldopen(fp, true, true, true, IOStream) do file
        write(file, "KSS", KSS)
        write(file, "VmSS", VmSS)
        write(file, "VkSS", VkSS)
        write(file, "distrSS", distrSS)
    end
    nothing
end

@inline function save_jacobian(m::BayerBornLuetticke{T}) where {T <: Real}
    fp = get_setting(m, :jacobian_output_file)
    if !ispath(dirname(fp))
        mkpath(dirname(fp))
    end
    JLD2.jldopen(fp, true, true, true, IOStream) do file
        write(file, "A", m[:A].value::Matrix{T})
        write(file, "B", m[:B].value::Matrix{T})
    end
    nothing
end

@inline function load_steadystate!(m::BayerBornLuetticke)
    out = JLD2.jldopen(get_setting(m, :steadystate_output_file), "r")
    prepare_linearization(m, out["KSS"], out["VmSS"], out["VkSS"], out["distrSS"]; verbose = :none)
    m <= Setting(:compute_full_steadystate, false) # set to false to avoid re-computing full steadystate
    m
end

@inline function load_jacobian!(m::BayerBornLuetticke)
    # loading in a pre-computed Jacobian, so it is assumed
    # we don't need to linearize the heterogeneous block
    m <= Setting(:linearize_heterogeneous_block, false)
    out = JLD2.jldopen(get_setting(m, :jacobian_output_file), "r")
    m[:A] = out["A"]
    m[:B] = out["B"]
    m
end

@inline function compute_and_save_irfs(m::BayerBornLuetticke,T,fp)
    system_main = compute_system(m)
    θ = parameters2namedtuple(m)
    shocks = collect(keys(m.exogenous_shocks))
    shock2deviation_dict = get_setting(m,:shock_to_deviation_dict)
    sd_shocks = zeros(length(shocks))
    ct = 1
    for  i in shocks
       sd_shocks[ct] = θ[shock2deviation_dict[i]]
       ct+=1
    end
    states, pseudo, obs = impulse_responses(m,system_main, T, shocks,sd_shocks)
    JLD2.jldopen(fp,true, true,true,IOStream) do file
      write(file,"states",states)
      write(file,"obs",obs)
      write(file,"pseudo",pseudo)
    end
    nothing
end


@doc raw"""
    ig_pars(igmean,igvariance)
Compute the location and shape parameter of the Inverse Gamma distribution from the mean and variance.
# Arguments
- `igmean`: prior mean of inverse gamma distribution [scalar]
- `igvariance`: prior variance of inverse gamma distribution [scalar]
Ouputs:
- `a`: location parameter of inverse gamma distribution [scalar]
- `b`: shape parameter of inverse gamma distribution [scalar]
"""
@inline function ig_pars(igmean, igvariance)
    a = igmean^2 / igvariance + 2
    b = igmean * (igmean^2 / igvariance + 1)

    return a, b
end
