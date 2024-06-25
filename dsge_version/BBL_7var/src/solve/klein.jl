"""
```
klein(m::AbstractModel{T}; minimum_inversion_tol::Float64 = 1e-4, verbose::Symbol = :none) where {T <: Real}
```
solves a first-order linear stochastic difference equation using the
Klein (2004) algorithm.

### Output
- `gx::Matrix{T}`: mapping from states to jumps, i.e. `jumps = gx * states`
- `hx::Matrix{T}`: transition equations for state variables, i.e. `states′_t = hx * states_t`
- `eu::Int`: return code specifying existence and uniqueness of solution
    * 1: exists and is unique
    * -1: local indeterminacy
    * -2: local non-existence
    * -3: numerical error during matrix inversions
"""
#using CSV, Tables
function klein(m::AbstractModel{T}; minimum_inversion_tol::Float64 = 1e-4, verbose::Symbol = :none) where {T <: Real}

    #################
    # Linearization
    #################
    # CHANGE BACK
    #if !get_setting(m,:linearize_heterogeneous_block)
    jac_out = jacobian(m)

    # A and B are defined in the first condition

    # Get A and B matrices for Klein
    if isa(jac_out, Tuple)
        # If jac_out is a Tuple, it is assumed that it is a 2-length tuple.
        # The first element is the A matrix, the second element is the B matrix,
        # and we will take the generalized Schur decomposition of A and -B.
        A = jac_out[1]::Matrix{T} # Need to get dense matrices out for schur
        B = -jac_out[2]::Matrix{T}
        #println("A preliminary value")
        #println(m[:A].value[1:5,1:5])
        n = size(A, 1)
    else
        Jac1 = jacobian(m)
        #CHECK ON THIS n approach
        #Jac1 = jac_out
        A = Jac1[:, 1:n]::Matrix{T}
        B = -Jac1[:, n+1:2*n]::Matrix{T}
    end


    NK = n_backward_looking_states(m)::Int

    ##################################################################################
    # Klein Solution Method---apply generalized Schur decomposition a la Klein (2000)
    ##################################################################################

    # Apply generlaized Schur decomposition
    # A ≈ QZ[:Q]*QZ[:S]*QZ[:Z]'
    # B ≈ QZ[:Q]*QZ[:T]*QZ[:Z]'


    QZ = schur!(deepcopy(A),deepcopy(B))
    #println("A post schur")
    #println(m[:A].value[1:5,1:5])

    # Reorder so that stable comes first

    alpha::Vector{complex(promote_type(eltype(A), eltype(B)))} = QZ.α
    beta::Vector{complex(promote_type(eltype(A), eltype(B)))} = QZ.β
	eigs = QZ.β ./ QZ.α #real(QZ.β ./ QZ.α)
	eigselect::AbstractArray{Bool} = abs.(eigs) .< 1 # returns false for NaN gen. eigenvalue which is correct here bc they are > 1
	ordschur!(QZ, eigselect)

    # Check that number of stable eigenvalues equals the number of predetermined state variables
	nk = sum(eigselect)
	eu = if nk > NK
        # Equilibrium is locally indeterminate
        -1
	elseif nk < NK
	    # No local equilibrium exists
        -2
    else
        1
	end

    inv_method = haskey(get_settings(m), :klein_inversion_method) ?
        get_setting(m, :klein_inversion_method) : :minimum_norm


    if inv_method == :minimum_norm
        gx_coef, hx_coef, invert_success = klein_minimum_norm_inversion(QZ, n, NK; tol = minimum_inversion_tol)
    elseif inv_method == :direct
        gx_coef, hx_coef, invert_success = klein_direct_inversion(QZ, n, NK)
    else
        throw(ArgumentError("Inversion method $(inv_method) is not recognized. Available ones are [:minimum_norm, :direct]"))
    end

    if invert_success != 1
        eu = -3
    end

	# next, want to represent policy functions in terms of meaningful things
	# gx_fval = Qy'*gx_coef*Qx
	# hx_fval = Qx'*hx_coef*Qx



    if(typeof(m) <: BayerBornLuetticke)
       #perhaps change the way this is used
       # m <= Setting(:State2Control, Matrix{Float64}(undef,0,0))
       # m <= Setting(:LOMstate, Matrix{Float64}(undef,0,0))
        m <= Setting(:State2Control, gx_coef)
        m <= Setting(:LOMstate, hx_coef)

        if get_setting(m,:linearize_heterogeneous_block)

            println("computing reduction")
            compute_reduction(m)

            m <= Setting(:linearize_heterogeneous_block, false)
            gx, hx, eu = klein(m)

            return gx,hx,eu
        end


    end
    if !get_setting(m,:linearize_heterogeneous_block)

        return gx_coef, hx_coef, eu
    end
end

# Need an additional transition_equation function to properly stack the
# individual state and jump transition matrices/shock mapping matrices to
# a single state space for all of the model_states
function klein_transition_matrices(m::AbstractModel{T}, TTT_state::Matrix{T}, TTT_jump::Matrix{T}) where {T <: Real}

    TTT = Matrix{T}(undef, n_model_states(m), n_model_states(m))

    n_states = n_backward_looking_states(m)::Int

    # Loading mapping time t states to time t+1 states
    TTT[1:n_states, 1:n_states] = TTT_state

    # Loading mapping time t jumps to time t+1 states
    TTT[1:n_states, n_states+1:end] .= 0.

    # Loading mapping time t states to time t+1 jumps
    TTT[n_states+1:end, 1:n_states] = TTT_jump * TTT_state

    # Loading mapping time t jumps to time t+1 jumps
    TTT[n_states+1:end, n_states+1:end] .= 0.

    RRR = shock_loading(m, TTT_jump)

    return TTT, RRR
end

function klein_minimum_norm_inversion(QZ::GeneralizedSchur, n::Int, NK::Int; tol::Float64 = 1e-4)


	U::Matrix{Float64} = QZ.Z'
	T::Matrix{Float64} = QZ.T
	S::Matrix{Float64} = QZ.S


    #OLD MINIMUM NORM INVERSION CODE
    U11 =  Matrix{Float64}(undef, NK, NK)
    U12 = Matrix{Float64}(undef, NK, NK-2)
    U21 = Matrix{Float64}(undef, NK-2, NK)
    U22 = Matrix{Float64}(undef, NK-2, NK-2)

    U11 = U[1:NK, 1:NK]
    U12 = U[1:NK, NK+1:end]
    U21 = U[NK+1:end, 1:NK]
    U22 = U[NK+1:end, NK+1:end]

    #"NEW" MINIMUM NORM INVERSION CODE
   # U_sz = size(U)
   # U22_prime = view(U, NK+1:U_sz[1], NK+1:U_sz[2])

	S11 = view(S, 1:NK, 1:NK)
	T11 = view(T, 1:NK, 1:NK)



   #= U_sz = size(U)
    U11 = view(U, 1:NK,1:NK)
	U12 = view(U, 1:NK, NK+1:U_sz[2])

	U21 = view(U, NK+1:U_sz[1], 1:NK)
	U22 = view(U, NK+1:U_sz[1], NK+1:U_sz[2])

	S11 = view(S, 1:NK, 1:NK)
	T11 = view(T, 1:NK, 1:NK)=#

    # Find minimum norm solution to U₂₁ + U₂₂*g_x = 0 (more numerically stable than -U₂₂⁻¹*U₂₁)
    gx_coef = Matrix{Float64}(undef, n-NK, NK)
	gx_coef = try
        -U22' * pinv(U22 * U22') * U21
    catch ex
        if isa(ex, LinearAlgebra.LAPACKException)
            # @info "LAPACK exception thrown while computing pseudo inverse of U22*U22'"
            return gx_coef, Array{Float64, 2}(undef, NK, NK), -1
        else
            rethrow(ex)
        end
    end


    #gx_coef_prime = -U22_prime' * pinv(U22_prime * U22_prime') * U21


    # Solve for h_x (in a more numerically stable way)
	S11invT11 = S11 \ T11
	Ustuff = U11 + U12 * gx_coef


    #THIS IS WHERE WE GET STUCK
	invterm = try
        pinv(I + gx_coef' * gx_coef)

    catch ex

        if isa(ex, LinearAlgebra.LAPACKException)
            # @info "LAPACK exception thrown while computing pseudo inverse of eye(NK) + gx_coef'*gx_+coef"
            return gx_coef, Array{Float64, 2}(undef, NK, NK), -1
        else
            rethrow(ex)
        end
    end


	hx_coef = invterm * Ustuff' * S11invT11 * Ustuff

	# Ensure that hx and S11invT11 should have same eigenvalues
	# (eigst,valst) = eig(S11invT11);
	# (eighx,valhx) = eig(hx_coef);
	eigst = eigvals(S11invT11)
    eighx = eigvals(hx_coef)
    invert_success = abs(norm(eighx, Inf) - norm(eigst, Inf)) > tol ? 1 : -1
		# @warn "max abs eigenvalue of S11invT11 and hx are different!"

    return gx_coef, hx_coef, invert_success
end

# Direct inversion method copied from SolveDiffEq in https://github.com/BenjaminBorn/HANK_BusinessCycleAndInequality
function klein_direct_inversion(Schur_decomp::GeneralizedSchur, n::Int, nk::Int)



   z21 = view(Schur_decomp.Z, (nk+1):n, 1:nk)
   z11 = view(Schur_decomp.Z, 1:nk, 1:nk)
   s11 = view(Schur_decomp.S, 1:nk, 1:nk)
   t11 = view(Schur_decomp.T, 1:nk, 1:nk)

    if rank(z11) < nk
        hx = Array{Float64}(undef, nk, nk) # change: original code use n_states, but nk = n_states when saddle-path stability satisfied
        gx = Array{Float64}(undef, n-nk, nk) # change: original code uses n_jumps, but n_jumps = n-nk when saddle-path stability satisfied
        return gx, hx, -1
    end
    z11i = z11 \ I # I is the identity matrix -> doesn't allocate an array!
    gx = real(z21 * z11i)
    hx = real(z11 * (s11 \ t11) * z11i)

    return gx, hx, 1
end



#=

function klein_minimum_norm_inversion(QZ::GeneralizedSchur, n::Int, NK::Int; tol::Float64 = 1e-4)


	U::Matrix{Float64} = QZ.Z'
	T::Matrix{Float64} = QZ.T
	S::Matrix{Float64} = QZ.S

#=    U11 = Matrix{Float64}(undef, NK, NK)
    U12 = Matrix{Float64}(undef, NK, NK-2)
    U21 = Matrix{Float64}(undef, NK-2, NK)
    U22 = Matrix{Float64}(undef, NK-2, NK-2)
    S11 = Matrix{Float64}(undef, NK, NK)
    T11 = Matrix{Float64}(undef, NK, NK)=#

    U_sz = size(U)
    U11 = view(U, 1:NK,1:NK)
	U12 = view(U, 1:NK, NK+1:U_sz[2])
	U21 = view(U, NK+1:U_sz[1], 1:NK)
	U22 = view(U, NK+1:U_sz[1], NK+1:U_sz[2])

	S11 = view(S, 1:NK, 1:NK)
	T11 = view(T, 1:NK, 1:NK)

    # Find minimum norm solution to U₂₁ + U₂₂*g_x = 0 (more numerically stable than -U₂₂⁻¹*U₂₁)
    gx_coef = Matrix{Float64}(undef, n-NK, NK)
	gx_coef = try
        -U22' * pinv(U22 * U22') * U21
    catch ex
        if isa(ex, LinearAlgebra.LAPACKException)
            # @info "LAPACK exception thrown while computing pseudo inverse of U22*U22'"
            return gx_coef, Array{Float64, 2}(undef, NK, NK), -1
        else
            rethrow(ex)
        end
    end

    # Solve for h_x (in a more numerically stable way)
	S11invT11 = S11 \ T11
	Ustuff = U11 + U12 * gx_coef



    #THIS IS WHERE WE GET STUCK
	invterm = try

        # pinv(eye(NK) + gx_coef' * gx_coef)
        pinv(I + gx_coef' * gx_coef)


    catch ex

        if isa(ex, LinearAlgebra.LAPACKException)
            # @info "LAPACK exception thrown while computing pseudo inverse of eye(NK) + gx_coef'*gx_+coef"
            return gx_coef, Array{Float64, 2}(undef, NK, NK), -1
        else
            rethrow(ex)
        end
    end


	hx_coef = invterm * Ustuff' * S11invT11 * Ustuff

	# Ensure that hx and S11invT11 should have same eigenvalues
	# (eigst,valst) = eig(S11invT11);
	# (eighx,valhx) = eig(hx_coef);
	eigst = eigvals(S11invT11)
    eighx = eigvals(hx_coef)
    invert_success = abs(norm(eighx, Inf) - norm(eigst, Inf)) > tol ? 1 : -1
		# @warn "max abs eigenvalue of S11invT11 and hx are different!"

    gx_coef, hx_coef, invert_success
end

=#
