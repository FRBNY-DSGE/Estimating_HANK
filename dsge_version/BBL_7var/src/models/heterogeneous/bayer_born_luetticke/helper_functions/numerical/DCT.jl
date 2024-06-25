#---------------------------------------------------------
# Discrete Cosine Transform
#---------------------------------------------------------
function  mydctmx(n::Int)
    DC::Array{Float64,2} =zeros(n,n);
    for j=0:n-1
        DC[1,j+1]=float(1/sqrt(n));
        for i=1:n-1
            DC[i+1,j+1]=(pi*float((j+1/2)*i/n));
            DC[i+1,j+1] = sqrt(2/n).*cos.(DC[i+1,j+1])
        end
    end
    return DC
end

function produceCompMat(DC,compressionIndexes,dims)
    IDD = DC[1]'
    for j=2:dims
        IDD = kron(DC[j]',IDD)
    end
    UnCompMat = IDD[:,compressionIndexes]
    CompMat   = inv(UnCompMat'*UnCompMat)*UnCompMat'
    return CompMat, UnCompMat
end

function uncompress(compressionIndexes, XC, DC, IDC, grid_dims::NTuple{3, Int})
    # POTENTIAL FOR SPEEDUP BY SPLITTING INTO DUAL AND REAL PART AND USE BLAS
    nm, nk, ny = grid_dims
    θ1 =zeros(eltype(XC),nm,nk,ny)
    for j  =1:length(XC)
        θ1[compressionIndexes[j]] = copy(XC[j])
    end
    @views for yy = 1:ny
        θ1[:,:,yy] = IDC[1]*θ1[:,:,yy]*DC[2]
    end
    @views for mm = 1:nm
        θ1[mm,:,:] = θ1[mm,:,:]*DC[3]
    end
    θ = reshape(θ1,nm * nk * ny)
    return θ
end

function uncompressD(compressionIndexes, XC, DC,IDC, grid_dims::NTuple{3, Int})
    nm, nk, ny = grid_dims
    # POTENTIAL FOR SPEEDUP BY SPLITTING INTO DUAL AND REAL PART AND USE BLAS
    θ1 =zeros(eltype(XC),nm-1,nk-1,ny-1)
    for j  =1:length(XC)
        θ1[compressionIndexes[j]] = copy(XC[j])
    end
    @views for yy = 1:ny-1
        θ1[:,:,yy] = IDC[1]*θ1[:,:,yy]*DC[2]
    end
    @views for mm = 1:nm-1
        θ1[mm,:,:] = θ1[mm,:,:]*DC[3]
    end
    θ = reshape(θ1,(nm-1)*(nk-1)*(ny-1))
    return θ
end


function compress(compressionIndexes::AbstractArray, XU::AbstractArray,
                  DC::AbstractArray, IDC::AbstractArray, grid_dims::NTuple{3, Int})
    nm, nk, ny = grid_dims
    θ   = zeros(eltype(XU),length(compressionIndexes))
    XU2 = zeros(eltype(XU),size(XU))
    # preallocate mm and kk (subs from comressionIndexes)
    mm  = zeros(Int,length(compressionIndexes))
    kk  = zeros(Int,length(compressionIndexes))
    zz  = zeros(Int,length(compressionIndexes))
    # Use the fact that not all elements of the grid are used in the compression index
    for j  = 1:length(compressionIndexes) # index to subs
        zz[j] = div(compressionIndexes[j]-1,nm*nk) +1
        kk[j] = div(compressionIndexes[j]- (zz[j]-1)*nm*nk-1, nm) +1
        mm[j] = compressionIndexes[j] - (zz[j]-1)*nm*nk -(kk[j]-1)*nm
    end

    # for j  = 1:length(compressionIndexes) # index to subs
        # zz[j] = CartesianIndices(mesh_m)[compressionIndexes[j]][3]
        # kk[j] = CartesianIndices(mesh_m)[compressionIndexes[j]][2]
        # mm[j] = CartesianIndices(mesh_m)[compressionIndexes[j]][1]
    # end
    # unique(mm) unique(kk) identify the grid elements retained in the
    # discrete cosine transformation

    # Eliminate unused rows/columns from the transformation matrix
    KK   = unique(kk)
    MM   = unique(mm)


    dc1  = DC[1][MM,:]
    dc2  = DC[2][KK,:]

    # Perform the DCT
    @inbounds @views begin
        for mmm in MM
           XU2[mmm,KK,:] .= (dc2*XU[mmm,:,:])*IDC[3]
        end
        for yy in unique(zz)
           XU2[MM,KK,yy] .= (dc1*XU2[:,KK,yy])#*idc2
        end

        for j  =1:length(compressionIndexes)
            θ[j] = XU2[compressionIndexes[j]]
            ## SBS CHANGED THIS PART TO UNDERSTAND INDEXING ERROR, NEEDS DOCUMENTATION
            #=zz[j] = div(compressionIndexes[j],(nm-1)*(nk-1)) +1
            kk[j] = div(compressionIndexes[j]- (zz[j]-1)*(nm-1)*(nk-1), nk-1) +1
            mm[j] = compressionIndexes[j] - (zz[j]-1)*(nm-1)*(nk-1) -(kk[j]-1)*(nk-1)
            =#
            zz[j] = div(compressionIndexes[j]-1,(nm)*(nk)) +1
            kk[j] = div(compressionIndexes[j]- (zz[j]-1)*(nm)*(nk)-1, nm) +1
            mm[j] = compressionIndexes[j] - (zz[j]-1)*(nm)*(nk) -(kk[j]-1)*(nm)

            if mm[j] == 0
                println("mm computation subtracted amount")
                println( (zz[j]-1)*(nm-1)*(nk-1) -(kk[j]-1)*(nk-1))
                println("test zero index compression \n")
                println(compressionIndexes[j])
                println("index inside compression index vector \n")
                println(j)
                println("length of compression index \n")
                println(length(compressionIndexes))
                println("zz_j \n")
                println(zz[j])
                println("kk_j \n")
                println(kk[j])
            end
        end
    end


    # for j  = 1:length(compressionIndexes) # index to subs
        # zz[j] = CartesianIndices(mesh_m)[compressionIndexes[j]][3]
        # kk[j] = CartesianIndices(mesh_m)[compressionIndexes[j]][2]
        # mm[j] = CartesianIndices(mesh_m)[compressionIndexes[j]][1]
    # end
    # unique(mm) unique(kk) identify the grid elements retained in the
    # discrete cosine transformation

    # Eliminate unused rows/columns from the transformation matrix
    KK   = unique(kk)
    MM   = unique(mm)
    println("see MM")
    println(MM)
    println("see KK")
    println(KK)

    dc1  = DC[1][MM,:]
    dc2  = DC[2][KK,:]

    # Perform the DCT
    @inbounds @views begin
        for mmm in MM
           XU2[mmm,KK,:] .= (dc2*XU[mmm,:,:])*IDC[3]
        end
        for yy in unique(zz)
           XU2[MM,KK,yy] .= (dc1*XU2[:,KK,yy])#*idc2
        end

        for j  =1:length(compressionIndexes)
            θ[j] = XU2[compressionIndexes[j]]
        end
    end
    return θ
end



## SBS added a preliminary compressD function to try and deal with the separate copula reduction step
function compressD(compressionIndexes::AbstractArray, XU::AbstractArray,
                  DC::AbstractArray, IDC::AbstractArray, grid_dims::NTuple{3, Int})
    println("Testing compressD function")
    println("Size DC1")
    println(size(DC[1]))
    nm, nk, ny = grid_dims
    θ   = zeros(eltype(XU),length(compressionIndexes))
    XU2 = zeros(eltype(XU),size(XU))
    # preallocate mm and kk (subs from comressionIndexes)
    mm  = zeros(Int,length(compressionIndexes))
    kk  = zeros(Int,length(compressionIndexes))
    zz  = zeros(Int,length(compressionIndexes))
    # Use the fact that not all elements of the grid are used in the compression index
    ## SBS modified to test about last row selection
    println("test new indexing")
    for j  = 1:length(compressionIndexes) # index to subs
        zz[j] = div(compressionIndexes[j]-1 ,(nm-1)*(nk-1)) +1
        kk[j] = div(compressionIndexes[j]- (zz[j]-1)*((nm-1)*(nk-1))-1, nm-1) +1
        mm[j] = compressionIndexes[j] - (zz[j]-1)*((nm-1)*(nk-1)) -(kk[j]-1)*(nm-1)
        if mm[j] == 0 ||  mm[j]>39
            println("Compression Index")
            println(compressionIndexes[j])
            println("j")
            println(j)
            println("z")
            println(zz[j])
            println("k")
            println(kk[j])
        end
    end

    # for j  = 1:length(compressionIndexes) # index to subs
        # zz[j] = CartesianIndices(mesh_m)[compressionIndexes[j]][3]
        # kk[j] = CartesianIndices(mesh_m)[compressionIndexes[j]][2]
        # mm[j] = CartesianIndices(mesh_m)[compressionIndexes[j]][1]
    # end
    # unique(mm) unique(kk) identify the grid elements retained in the
    # discrete cosine transformation

    # Eliminate unused rows/columns from the transformation matrix
    KK   = unique(kk)
    MM   = unique(mm)
    println("see MM")
    println(MM)
    println("see KK")
    println(KK)

    dc1  = DC[1][MM,:]
    dc2  = DC[2][KK,:]

    # Perform the DCT
    @inbounds @views begin
        for mmm in MM
           XU2[mmm,KK,:] .= (dc2*XU[mmm,:,:])*IDC[3]
        end
        for yy in unique(zz)
           XU2[MM,KK,yy] .= (dc1*XU2[:,KK,yy])#*idc2
        end

        for j  =1:length(compressionIndexes)
            θ[j] = XU2[compressionIndexes[j]]
            #=
            if j == 2
            # automatically starts accessing elements from flattened grid size
            println("sizes of XU2 and  theta assignment")
            println(compressionIndexes[j])
            print(size(XU2[13]))
            println(size(XU2))
            println(size(θ))
            end
            =#
            ## SBS CHANGED THIS PART TO UNDERSTAND INDEXING ERROR, NEEDS DOCUMENTATION
            zz[j] = div(compressionIndexes[j]-1,(nm-1)*(nk-1)) +1
            kk[j] = div(compressionIndexes[j]- (zz[j]-1)*(nm-1)*(nk-1)-1, nm-1) +1
            mm[j] = compressionIndexes[j] - (zz[j]-1)*(nm-1)*(nk-1) -(kk[j]-1)*(nm-1)
            #=
            zz[j] = div(compressionIndexes[j],(nm)*(nk)) +1
            kk[j] = div(compressionIndexes[j]- (zz[j]-1)*(nm)*(nk), nk) +1
            mm[j] = compressionIndexes[j] - (zz[j]-1)*(nm)*(nk) -(kk[j]-1)*(nk)
            =#
            if mm[j] == 0
                println("mm computation subtracted amount")
                println( (zz[j]-1)*(nm-1)*(nk-1) -(kk[j]-1)*(nk-1))
                println("test zero index compression \n")
                println(compressionIndexes[j])
                println("index inside compression index vector \n")
                println(j)
                println("length of compression index \n")
                println(length(compressionIndexes))
                println("zz_j \n")
                println(zz[j])
                println("kk_j \n")
                println(kk[j])
            end
        end
    end


    # for j  = 1:length(compressionIndexes) # index to subs
        # zz[j] = CartesianIndices(mesh_m)[compressionIndexes[j]][3]
        # kk[j] = CartesianIndices(mesh_m)[compressionIndexes[j]][2]
        # mm[j] = CartesianIndices(mesh_m)[compressionIndexes[j]][1]
    # end
    # unique(mm) unique(kk) identify the grid elements retained in the
    # discrete cosine transformation

    # Eliminate unused rows/columns from the transformation matrix
    KK   = unique(kk)
    MM   = unique(mm)
    println("see MM")
    println(MM)
    println("see KK")
    println(KK)

    dc1  = DC[1][MM,:]
    dc2  = DC[2][KK,:]

    # Perform the DCT
    @inbounds @views begin
        for mmm in MM
           XU2[mmm,KK,:] .= (dc2*XU[mmm,:,:])*IDC[3]
        end
        for yy in unique(zz)
           XU2[MM,KK,yy] .= (dc1*XU2[:,KK,yy])#*idc2
        end

        for j  =1:length(compressionIndexes)
            θ[j] = XU2[compressionIndexes[j]]
        end
    end
    println("size theta dimension check D_thet")
    println(size(θ))
    #println("theta")
    #println(θ)
    return θ
end


function bbl_uncompress(compressionIndexes, XC, DC,IDC)
    nm = size(DC[1],1)
    nk = size(DC[2],1)
    ny = size(DC[3],1)
    # POTENTIAL FOR SPEEDUP BY SPLITTING INTO DUAL AND REAL PART AND USE BLAS
    θ1 =zeros(eltype(XC),nm,nk,ny)
    for j  =1:length(XC)
        θ1[compressionIndexes[j]] = copy(XC[j])
    end
    @views for yy = 1:ny
        θ1[:,:,yy] = IDC[1]*θ1[:,:,yy]*DC[2]
    end
    @views for mm = 1:nm
        θ1[mm,:,:] = θ1[mm,:,:]*DC[3]
    end
    θ = reshape(θ1,(nm)*(nk)*(ny))
    return θ
end

function bbl_compress(compressionIndexes::AbstractArray, XU::AbstractArray,
    DC::AbstractArray,IDC::AbstractArray)
    nm, nk, ny=size(XU)
    θ   = zeros(eltype(XU),length(compressionIndexes))
    XU2 = zeros(eltype(XU),size(XU))
    @inbounds @views for m = 1:nm
        XU2[m,:,:] = DC[2]*XU[m,:,:]*IDC[3]
    end
    @inbounds @views  for y = 1:ny
        XU2[:,:,y] = DC[1]*XU2[:,:,y]
    end
    θ = XU2[compressionIndexes]
    return θ
end
