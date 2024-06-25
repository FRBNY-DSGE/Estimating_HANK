# include("original_aggregate_equations.jl") # Don't include b/c we want to use it as a "model file" and call it via @include
include("FSYS_agg.jl")
include("FSYS.jl")
include("compute_reduction.jl")
