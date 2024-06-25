function original_MultipleDirectTransition(m_a_star::AbstractArray{T,3},
                                           m_n_star::AbstractArray{T,3},
                                           k_a_star::AbstractArray{T,3},
                                           distr::AbstractArray{T,3},
                                           λ::T,
                                           Π::AbstractArray{T,2}, dims::NTuple{3, Int},
                                           grids::NTuple{3, <: AbstractVector{T}}, ϵ::T;
                                           iters::Int = 10000) where {T <: Real}

    # unpack dims and grids
    nm,     nk,     ny     = dims
    m_grid, k_grid, y_grid = grids

    # Make linear interpolation weights
    idk_a, wR_k_a, wL_k_a = MakeWeights(k_a_star,k_grid)
    idm_a, wR_m_a, wL_m_a = MakeWeights(m_a_star,m_grid)
    idm_n, wR_m_n, wL_m_n = MakeWeights(m_n_star,m_grid)
    dist =1.0
    count = 1
    blockindex = (0:ny-1)*nk*nm
    while (dist>ϵ) && (count<iters)
        dPrime = zeros(eltype(distr),size(distr))
        for zz = 1:ny # all current income states
            for kk = 1:nk # all current illiquid asset states
                #idk_n = kk
                for mm = 1:nm
                    dd=distr[mm,kk,zz]
                    IDD_a = idm_a[mm,kk,zz].+(idk_a[mm,kk,zz]-1).*nm
                    IDD_n = idm_n[mm,kk,zz].+(kk-1).*nm
                    DLL_a = dd.*wL_k_a[mm,kk,zz].*wL_m_a[mm,kk,zz]
                    DLR_a = dd.*wL_k_a[mm,kk,zz].*wR_m_a[mm,kk,zz]
                    DRL_a = dd.*wR_k_a[mm,kk,zz].*wL_m_a[mm,kk,zz]
                    DRR_a = dd.*wR_k_a[mm,kk,zz].*wR_m_a[mm,kk,zz]
                    DL_n  = dd.*wL_m_n[mm,kk,zz]
                    DR_n  = dd.*wR_m_n[mm,kk,zz]
                    for yy = 1:ny
                        id_a = IDD_a +blockindex[yy]
                        id_n = IDD_n +blockindex[yy]
                        fac = λ.*Π[zz,yy]
                        dPrime[id_a]            += fac.*DLL_a
                        dPrime[id_a+1]          += fac.*DLR_a
                        dPrime[id_a+nm]   += fac.*DRL_a
                        dPrime[id_a+nm+1] += fac.*DRR_a
                        dPrime[id_n]            += (1.0-λ).*Π[zz,yy].*DL_n
                        dPrime[id_n+1]          += (1.0-λ).*Π[zz,yy].*DR_n
                    end
                end
            end
        end
        dist  = maximum(abs.(vec(dPrime - distr)))
        distr = dPrime
        count = count+1
    end
    return distr, dist, count
end
