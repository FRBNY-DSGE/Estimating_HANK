function MakeWeights(xpol,grid)
    idx         = Array{Int}(undef, size(xpol))
    weightright = Array{eltype(xpol)}(undef, size(xpol))
    weightleft  = Array{eltype(xpol)}(undef, size(xpol))
    dx          = diff(grid)

    for i = 1:length(xpol)
        if xpol[i] <= grid[1]
            idx[i] = 1
        elseif xpol[i] >= grid[end]
            idx[i] = length(grid) - 1
        else
            idx[i] = locate(xpol[i], grid)
        end
        weightright[i] = (xpol[i] - grid[idx[i]]) / dx[idx[i]]
        if weightright[i] >= 1.0
            weightright[i] = 1.0 - 1.0e-14
        elseif weightright[i] <= 0.0
            weightright[i] = 1.0e-14
        end
        weightleft[i]      = 1.0 - weightright[i]
    end
    return idx, weightright, weightleft
end

function MakeWeightsLight(xpol,grid)
    idx         = Array{Int}(undef, size(xpol))
    weightright = Array{eltype(xpol)}(undef, size(xpol))
    dx          = diff(grid)
    @fastmath @inbounds begin
        for i in eachindex(xpol)
            if xpol[i] <= grid[1]
                idx[i]     = 1
            elseif xpol[i] >= grid[end]
                idx[i]     = length(grid) - 1
            else
                idx[i]     = locate(xpol[i], grid)
            end
            weightright[i] = (xpol[i] - grid[idx[i]]) / dx[idx[i]]
            if weightright[i] >= 1.0
                weightright[i] = 1.0 - 1.0e-14
            elseif weightright[i] <= 0.0
                weightright[i] = 1.0e-14
            end
        end
    end
    return idx, weightright
end
