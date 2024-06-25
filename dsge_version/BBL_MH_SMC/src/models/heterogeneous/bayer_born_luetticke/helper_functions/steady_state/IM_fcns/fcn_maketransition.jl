function MakeTransition(m_a_star::AbstractArray{T,3},
                        m_n_star::AbstractArray{T,3},
                        k_a_star::AbstractArray{T,3},
                        Π::AbstractArray{T,2}, dims::NTuple{3, Int64},
                        m_grid::AbstractVector{T}, k_grid::AbstractVector{T},
                        y_grid::AbstractVector{T}; parallel::Bool = false) where {T <: Real}

    if parallel
        return _parallel_MakeTransition(m_a_star, m_n_star, k_a_star, Π, dims, m_grid, k_grid, y_grid)
    end

    # unpack dims
    nm, nk, ny = dims
    mky_CI     = collect(enumerate(CartesianIndices((nm, nk, ny))))

    # create linear interpolation weights from policy functions
    idk_a, weightright_k_a, weightleft_k_a = MakeWeights(k_a_star,k_grid)
    idm_a, weightright_m_a, weightleft_m_a = MakeWeights(m_a_star,m_grid)
    idm_n, weightright_m_n, weightleft_m_n = MakeWeights(m_n_star,m_grid)

    # Adjustment case
    weight      = Array{typeof(k_a_star[1]),3}(undef, 4,ny,nk* nm*ny)
    targetindex = zeros(Int,4,ny,nk* nm*ny)
    startindex  = zeros(Int,4,ny,nk* nm*ny)
    blockindex  = (0:ny-1)*nk*nm

    @inbounds @fastmath begin
        # can't use @simd on following loop unless we handle the linear index separately
        for (runindex, mmkkzz) in mky_CI # loop over all current income states, illiquid asset states, and liquid asset states
            mm, kk, zz = mmkkzz[1], mmkkzz[2], mmkkzz[3]
            WLL      = weightleft_m_a[mm,kk,zz] .* weightleft_k_a[mm,kk,zz]
            WRL      = weightright_m_a[mm,kk,zz].* weightleft_k_a[mm,kk,zz]
            WLR      = weightleft_m_a[mm,kk,zz] .* weightright_k_a[mm,kk,zz]
            WRR      = weightright_m_a[mm,kk,zz].* weightright_k_a[mm,kk,zz]
            IDD      = idm_a[mm,kk,zz].+(idk_a[mm,kk,zz]-1).*nm
            @simd for jj = 1:ny
                pp                         = Π[zz,jj]
                bb                         = blockindex[jj]
                weight[1,jj,runindex]      = WLL .* pp
                weight[2,jj,runindex]      = WRL .* pp
                weight[3,jj,runindex]      = WLR .* pp
                weight[4,jj,runindex]      = WRR .* pp
                targetindex[1,jj,runindex] = IDD .+ bb
                targetindex[2,jj,runindex] = IDD + 1 .+ bb
                targetindex[3,jj,runindex] = IDD + nm .+ bb
                targetindex[4,jj,runindex] = IDD + nm + 1 .+ bb
                startindex[1,jj,runindex]  = runindex
                startindex[2,jj,runindex]  = runindex
                startindex[3,jj,runindex]  = runindex
                startindex[4,jj,runindex]  = runindex
            end
        end
    end
    S_a          = vec(startindex)
    T_a          = vec(targetindex)
    W_a          = vec(weight)

    # Non-Adjustment case
    weight2      = zeros(typeof(k_a_star[1]), 2,ny,nk* nm*ny)
    targetindex2 = zeros(Int, 2,ny,nk* nm*ny)
    startindex2  = zeros(Int,2,ny,nk* nm*ny)
    @inbounds @fastmath begin
        for (runindex, mmkkzz) in mky_CI # loop over all current income states, illiquid asset states, and liquid asset states
            mm, kk, zz = mmkkzz[1], mmkkzz[2], mmkkzz[3]
            WL       = weightleft_m_n[mm,kk,zz]
            WR       = weightright_m_n[mm,kk,zz]
            CI       = idm_n[mm,kk,zz].+(kk-1).*nm
            @simd for jj = 1:ny
                pp                          = Π[zz,jj]
                weight2[1,jj,runindex]      = WL .* pp
                weight2[2,jj,runindex]      = WR .* pp
                targetindex2[1,jj,runindex] = CI .+ blockindex[jj]
                targetindex2[2,jj,runindex] = CI .+ 1 .+blockindex[jj]
                startindex2[1,jj,runindex]  = runindex
                startindex2[2,jj,runindex]  = runindex
            end
        end
    end
    S_n        = vec(startindex2)
    T_n        = vec(targetindex2)
    W_n        = vec(weight2)

    return S_a, T_a, W_a, S_n, T_n, W_n
end

function _parallel_MakeTransition(m_a_star::AbstractArray{T,3},
                                  m_n_star::AbstractArray{T,3},
                                  k_a_star::AbstractArray{T,3},
                                  Π::AbstractArray{T,2}, dims::NTuple{3, Int64},
                                  m_grid::AbstractVector{T}, k_grid::AbstractVector{T},
                                  y_grid::AbstractVector{T}) where {T <: Real}

    # unpack dims and grids
    nm, nk, ny = dims
    mky_CI     = collect(enumerate(CartesianIndices((nm, nk, ny)))) # have to write it this way to make threads work

    # create linear interpolation weights from policy functions
    idk_a, weightright_k_a, weightleft_k_a = MakeWeights(k_a_star,k_grid)
    idm_a, weightright_m_a, weightleft_m_a = MakeWeights(m_a_star,m_grid)
    idm_n, weightright_m_n, weightleft_m_n = MakeWeights(m_n_star,m_grid)

    # Adjustment case
    weight      = Array{typeof(k_a_star[1]),3}(undef, 4, ny, nm * nk * ny)
    targetindex = Array{Int,3}(undef, 4, ny, nm * nk * ny)
    startindex  = Array{Int,3}(undef, 4, ny, nm * nk * ny)
    blockindex  = (0:ny-1)*nk*nm

    @inbounds @fastmath begin
         Threads.@threads for (runindex, mmkkzz) in mky_CI
            mm, kk, zz = mmkkzz[1], mmkkzz[2], mmkkzz[3]
            WLL      = weightleft_m_a[mm,kk,zz] .* weightleft_k_a[mm,kk,zz]
            WRL      = weightright_m_a[mm,kk,zz].* weightleft_k_a[mm,kk,zz]
            WLR      = weightleft_m_a[mm,kk,zz] .* weightright_k_a[mm,kk,zz]
            WRR      = weightright_m_a[mm,kk,zz].* weightright_k_a[mm,kk,zz]
            IDD      = idm_a[mm,kk,zz].+(idk_a[mm,kk,zz]-1).*nm
            @simd for jj = 1:ny
                pp                         = Π[zz,jj]
                bb                         = blockindex[jj]
                weight[1,jj,runindex]      = WLL .* pp
                weight[2,jj,runindex]      = WRL .* pp
                weight[3,jj,runindex]      = WLR .* pp
                weight[4,jj,runindex]      = WRR .* pp
                targetindex[1,jj,runindex] = IDD .+ bb
                targetindex[2,jj,runindex] = IDD + 1 .+ bb
                targetindex[3,jj,runindex] = IDD + nm .+ bb
                targetindex[4,jj,runindex] = IDD + nm + 1 .+ bb
                startindex[1,jj,runindex]  = runindex
                startindex[2,jj,runindex]  = runindex
                startindex[3,jj,runindex]  = runindex
                startindex[4,jj,runindex]  = runindex
            end
        end
    end
    S_a          = vec(startindex)
    T_a          = vec(targetindex)
    W_a          = vec(weight)

    # Non-Adjustment case
    weight2      = Array{typeof(k_a_star[1]),3}(undef, 2, ny, nm * nk * ny)
    targetindex2 = Array{Int,3}(undef, 2, ny, nm * nk * ny)
    startindex2  = Array{Int,3}(undef, 2, ny, nm * nk * ny)
    @inbounds @fastmath begin
        Threads.@threads for (runindex, mmkkzz) in mky_CI
            mm, kk, zz = mmkkzz[1], mmkkzz[2], mmkkzz[3]
            WL       = weightleft_m_n[mm,kk,zz]
            WR       = weightright_m_n[mm,kk,zz]
            CI       = idm_n[mm,kk,zz].+(kk-1).*nm
            @simd for jj = 1:ny
                pp                          = Π[zz,jj]
                weight2[1,jj,runindex]      = WL .* pp
                weight2[2,jj,runindex]      = WR .* pp
                targetindex2[1,jj,runindex] = CI .+ blockindex[jj]
                targetindex2[2,jj,runindex] = CI .+ 1 .+blockindex[jj]
                startindex2[1,jj,runindex]  = runindex
                startindex2[2,jj,runindex]  = runindex
            end
        end
    end
    S_n        = vec(startindex2)
    T_n        = vec(targetindex2)
    W_n        = vec(weight2)

    return S_a, T_a, W_a, S_n, T_n, W_n
end
