# This file holds a bunch of helper functions related to the kolmogorov forward equation

"""
```
stationary_eigenvector(A::AbstractMatrix, method::Symbol = :krylov, kwargs...)
stationary_eigenvector(A::AbstractMatrix, c::S, method::Symbol = :krylov, tol = 1e-2, kwargs...) where {S <: Real}
```

calculates the eigenvector representing the stationary solution of the Kolmgorov forward equation
represented by the transition matrix `A`. Typically, this eigenvector corresponds to the largest
eigenvalue of `A`.

Further, if `A` is properly normalized, then this eigenvalue will also be close to 1.
However, in some applications, it is computationally convenient to calculate the eigenvector of
the matrix `B ≡ cA`, where c is some scaling constant. For this reason, we provide a
second method whose second input argument is the scaling constant `c`. The function then
checks whether the eigenvalue corresponding to the stationary eigenvector
is close to `1 / c`, given some tolerance `tol`.
"""
function stationary_eigenvector(A::AbstractMatrix, method::Symbol = :krylov, kwargs...)

    if method == :krylov
        λs, Vs = eigsolve(A, 1, :LM, kwargs...)
        return abs(λs[1]), real(Vs[1])
    elseif method == :arnoldi
        λs, Vs = partialeigen(partialschur(transition_mat; nev = 1, which = LM(), kwargs...)[1])
        return abs(λs[1]), real(Vs[1])
    elseif method == :eigen
        λs, Vs = (eigen(transition_mat)..., )
        max_λ = argmax(abs.(λs))
        return abs(λs[max_λ]), real(Vs[:, max_λ])
    else
        error("Method $(method) not available. Available methods to solve " *
              "for the stationary eigenvector are [:krylov, :arnoldi, :eigen].")
    end
end

function stationary_eigenvector(A::AbstractMatrix, c::S = 1., method::Symbol = :krylov; tol::S = 1e-2, kwargs...) where {S <: Real}
    λ, V = stationary_eigenvector(A, method; kwargs...)

    if abs(λ - 1 / c) > tol
       throw(KolmogorovForwardError("Your eigenvalue ($(round(λ, digits = 5))) is too far from $(1/c)"))
    end

    return λ, V
end

"""
```
KolmogorovForwardError <: Exception
```
A `KolmogorovForwardError` indicates an error encountered during solution of the Kolmogorov Forward equation, typically
in the steady state.
"""
mutable struct KolmogorovForwardError <: Exception
    msg::String
end
KolmogorovForwardError() = KolmogorovForwardError("")
Base.showerror(io::IO, ex::KolmogorovForwardError) = print(io, ex.msg)
