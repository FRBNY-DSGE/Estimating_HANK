@doc raw"""
    SolveDiffEq(A, B, n_par; estim)

Calculate the solution to the linearized difference equations defined as
P'*B*P x_t = P'*A*P x_{t+1}, where `P` is the (ntotal x r) semi-unitary model reduction matrix
`n_par.PRightAll` of potentially reduced rank r.

# Arguments
- `A`,`B`: matrices with first derivatives
- `n_par::NumericalParameters`: `n_par.sol_algo` determines
    the solution algorithm, options are:
    * `litx`:  Linear time iteration (implementation follows Reiter)
    * `schur`: Klein's algorithm (preferable if number of controls is small)

# Returns
- `gx`,`hx`: observation equations [`gx`] and state transition equations [`hx`]
- `alarm_sgu`,`nk`: `alarm_sgu=true` when solving algorithm fails, `nk` number of
    predetermined variables
"""
function solveDiffEQ(m::AbstractModel{T},  estim=false)
    lit_fail = false
    jac1 = jacobian(m)
    A = jac_out[1]::Matrix{T}
    B = -jac_out[2]::Matrix{T}


        alarm_sgu = false
        Schur_decomp, slt, nk, λ = complex_schur(A, -B) # first output is generalized Schur factorization
        # Check for determinacy and existence of solution
        if n_par.nstates_r != nk
            if estim # return zeros if not unique and determinate
                hx = zeros(eltype(A), n_par.nstates_r, n_par.nstates_r)
                gx = zeros(eltype(A), n_par.ncontrols_r, n_par.nstates_r)
                alarm_sgu = true
                return gx, hx, alarm_sgu, nk, A, B
            else # debug mode/ allow IRFs to be produced for roughly determinate system
                ind = sortperm(abs.(λ); rev = true)
                slt = zeros(Bool, size(slt))
                slt[ind[1:n_par.nstates_r]] .= true
                alarm_sgu = true
                @warn "critical eigenvalue moved to:"
                print(λ[ind[n_par.nstates_r-5:n_par.nstates_r+5]])
                print(λ[ind[1]])
                nk = n_par.nstates_r
            end
        # in-place reordering of eigenvalues for decomposition
        ordschur!(Schur_decomp, slt)


        z11i = z11 \ I # I is the identity matrix -> doesn't allocate an array!
        gx = real(z21 * z11i)
        hx = real(z11 * (s11 \ t11) * z11i)
    end
    if n_par.sol_algo != :schur && n_par.sol_algo != :litx && n_par.sol_algo != :litx_s
        error("Solution algorithm not defined!")
    end

    return gx, hx, alarm_sgu, nk
end
