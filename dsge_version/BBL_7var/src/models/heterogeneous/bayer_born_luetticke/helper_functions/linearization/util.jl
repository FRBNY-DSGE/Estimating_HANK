"""
```
shuffle_matrix(distr)
```
computes a matrix (linear transformation)
which removes one degree of freedom from a distribution over
3 idiosyncratic states
"""
function shuffle_matrix(distr)
    nm, nk, ny = size(distr)

    sum_distr = sum(distr)
    distr_m = dropdims(sum(sum(distr,dims=3),dims=2)./sum_distr, dims = (2, 3))
    distr_k = dropdims(sum(sum(distr,dims=3),dims=1)./sum_distr, dims = (1, 3))
    distr_y = dropdims(sum(sum(distr,dims=2),dims=1)./sum_distr, dims = (1, 2))

    Γ    = Array{Array{Float64,2},1}(undef,3)
    Γ[1] = zeros(Float64,(nm,nm-1))
    Γ[2] = zeros(Float64,(nk,nk-1))
    Γ[3] = zeros(Float64,(ny,ny-1))
    for j=1:nm-1
        Γ[1][:,j] = -distr_m
        Γ[1][j,j] = 1-distr_m[j]
        Γ[1][j,j] = Γ[1][j,j] - sum(view(Γ[1], :,j))
    end
    for j=1:nk-1
        Γ[2][:,j] = -distr_k
        Γ[2][j,j] = 1-distr_k[j]
        Γ[2][j,j] = Γ[2][j,j] - sum(view(Γ[2], :,j))
    end
    for j=1:ny-1
        Γ[3][:,j] = -distr_y
        Γ[3][j,j] = 1-distr_y[j]
        Γ[3][j,j] = Γ[3][j,j] - sum(view(Γ[3], :,j))
    end

    return Γ
end

# Helper functions for linearization
function tot_dual(x::ForwardDiff.Dual)
    a = sum(ForwardDiff.partials(x,:))
    return a
end
tot_dual(x::T) where {T <: Real} = zero(T)

function realpart(x::ForwardDiff.Dual)
    a = ForwardDiff.value(x)
    return a
end
realpart(x::T) where {T <: Real} = x
