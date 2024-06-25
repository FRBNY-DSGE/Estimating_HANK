# TODO: add docstrings for macros
@inline function variable2index2value(x::AbstractVector{S}, d::AbstractDict{Symbol, <: UnitRange}, ::Val{k}) where {S <: Number, k}
    return length(d[k]) > 1 ? (@view x[d[k]]) : x[d[k][1]]
end


@inline function variable2index2value(x::AbstractVector{S}, d::AbstractDict{Symbol, Int}, ::Val{k}) where {S <: Number, k}
    return x[d[k]]
end

macro variables2indices2values(args) # Based on @unpack from UnPack
    varnames, x_idict = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, idict = x_idict.args
    x_instance = gensym()
    idict_instance = gensym()
    kd = [:( $key = $variable2index2value($x_instance, $idict_instance, Val{$(Expr(:quote, key))}()) ) for key in varnames]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $x_instance = $x # handles if x_instance is not a variable but an expression
        local $idict_instance = $idict # handles if idict_instance is not a variable but an expression
        $kdblock
    end
    esc(expr)
end

@inline function sslogdeviation2level(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, <: UnitRange},
                                      nt::NamedTuple, ::Val{k}) where {k}
    return (length(d[k]) > 1 ? exp.((@view x[d[k]]) + nt[k]) : exp(x[d[k][1]] + nt[k]))
end

@inline function sslogdeviation2level(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, Int},
                                      nt::NamedTuple, ::Val{k}) where {k}
    return exp(x[d[k]] + nt[k])
end

@inline function sslogdeviation2level_unprimekeys(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, <: UnitRange},
                                                  nt::NamedTuple, ::Val{k}) where {k}
    kunprime = unprime(k) # keys of nt are assumed to not have primes on them, but we still want to use prime keys for d
    return (length(d[k]) > 1 ? exp.((@view x[d[k]]) + nt[kunprime]) : exp(x[d[k][1]] + nt[kunprime]))
end

@inline function sslogdeviation2level_unprimekeys(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, Int},
                                                  nt::NamedTuple, ::Val{k}) where {k}
    kunprime = unprime(k) # keys of nt are assumed to not have primes on them, but we still want to use prime keys for d
    return exp(x[d[k]] + nt[kunprime])
end

@inline function sslogdeviation2log(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, <: UnitRange},
                                    nt::NamedTuple, ::Val{k}) where {k}
    return (length(d[k]) > 1 ? (@view x[d[k]]) : x[d[k][1]]) + nt[k]
end

@inline function sslogdeviation2log(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, Int},
                                    nt::NamedTuple, ::Val{k}) where {k}
    return x[d[k]] + nt[k]
end

@inline function ssdeviation2level(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, <: UnitRange},
                                   nt::NamedTuple, ::Val{k}) where {k}
    return (length(d[k]) > 1 ? (@view x[d[k]]) : x[d[k][1]]) + nt[k]
end

@inline function ssdeviation2level(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, Int},
                                   nt::NamedTuple, ::Val{k}) where {k}
    return x[d[k]] + nt[k]
end

@inline function get_deviation(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, <: UnitRange}, ::Val{k}) where {k}
    return length(d[k]) > 1 ? (@view x[d[k]]) : x[d[k][1]]
end

@inline function get_deviation(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, Int}, ::Val{k}) where {k}
    return x[d[k]]
end

macro sslogdeviations2levels(args) # Based on @unpack from UnPack
    varnames, x_idict_nt = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, idict, nt = x_idict_nt.args
    x_instance = gensym()
    idict_instance = gensym()
    nt_instance = gensym()
    kd = [:( $key = $sslogdeviation2level($x_instance, $idict_instance, $nt_instance, Val{$(Expr(:quote, key))}()) ) for key in varnames]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $x_instance = $x # handles if x_instance is not a variable but an expression
        local $idict_instance = $idict # handles if idict_instance is not a variable but an expression
        local $nt_instance = $nt # handles if nt_instance is not a variable but an expression
        $kdblock
    end
    esc(expr)
end

macro sslogdeviations2levels_unprimekeys(args) # Based on @unpack from UnPack
    varnames, x_idict_nt = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, idict, nt = x_idict_nt.args
    x_instance = gensym()
    idict_instance = gensym()
    nt_instance = gensym()
    kd = [:( $key = $sslogdeviation2level_unprimekeys($x_instance, $idict_instance, $nt_instance, Val{$(Expr(:quote, key))}()) ) for key in varnames]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $x_instance = $x # handles if x_instance is not a variable but an expression
        local $idict_instance = $idict # handles if idict_instance is not a variable but an expression
        local $nt_instance = $nt # handles if nt_instance is not a variable but an expression
        $kdblock
    end
    esc(expr)
end

macro sslogdeviations2logs(args) # Based on @unpack from UnPack
    varnames, x_idict_nt = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, idict, nt = x_idict_nt.args
    x_instance = gensym()
    idict_instance = gensym()
    nt_instance = gensym()
    kd = [:( $key = $sslogdeviation2log($x_instance, $idict_instance, $nt_instance, Val{$(Expr(:quote, key))}()) ) for key in varnames]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $x_instance = $x # handles if x_instance is not a variable but an expression
        local $idict_instance = $idict # handles if idict_instance is not a variable but an expression
        local $nt_instance = $nt # handles if nt_instance is not a variable but an expression
        $kdblock
    end
    esc(expr)
end

macro ssdeviations2levels(args) # Based on @unpack from UnPack
    varnames, x_idict_nt = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, idict, nt = x_idict_nt.args
    x_instance = gensym()
    idict_instance = gensym()
    nt_instance = gensym()
    kd = [:( $key = $ssdeviation2level($x_instance, $idict_instance, $nt_instance, Val{$(Expr(:quote, key))}()) ) for key in varnames]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $x_instance = $x # handles if x_instance is not a variable but an expression
        local $idict_instance = $idict # handles if idict_instance is not a variable but an expression
        local $nt_instance = $nt # handles if nt_instance is not a variable but an expression
        $kdblock
    end
    esc(expr)
end

macro sslogdeviations2logs(args) # Based on @unpack from UnPack
    varnames, x_idict_nt = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, idict, nt = x_idict_nt.args
    x_instance = gensym()
    idict_instance = gensym()
    nt_instance = gensym()
    kd = [:( $key = $sslogdeviation2log($x_instance, $idict_instance, $nt_instance, Val{$(Expr(:quote, key))}()) ) for key in varnames]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $x_instance = $x # handles if x_instance is not a variable but an expression
        local $idict_instance = $idict # handles if idict_instance is not a variable but an expression
        local $nt_instance = $nt # handles if nt_instance is not a variable but an expression
        $kdblock
    end
    esc(expr)
end

macro get_deviations(args) # Based on @unpack from UnPack
    varnames, x_idict = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, idict = x_idict.args
    x_instance = gensym()
    idict_instance = gensym()
    kd = [:( $key = $get_deviation($x_instance, $idict_instance, Val{$(Expr(:quote, key))}()) ) for key in varnames]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $x_instance = $x # handles if x_instance is not a variable but an expression
        local $idict_instance = $idict # handles if idict_instance is not a variable but an expression
        $kdblock
    end
    esc(expr)
end

"""
```
@unpack_and_first
```
unpacks a dictionary and calls `first` on the unpacked value.
The principle use of this macro is for unpacking a dictionary
whose values are 1-length `UnitRange` instances.
"""
@inline unpack_and_first(x::AbstractDict{Symbol, UnitRange{Int}}, ::Val{k}) where {k} = first(x[k])
@inline unpack_and_first(x::AbstractDict{Symbol, Int}, ::Val{k}) where {k} = x[k] # avoid calling first when unnecessary
macro unpack_and_first(args)
    args.head!=:(=) && error("Expression needs to be of form `a, b = c`")
    items, suitecase = args.args
    items = isa(items, Symbol) ? [items] : items.args
    suitecase_instance = gensym()
    kd = [:( $key = $unpack_and_first($suitecase_instance, Val{$(Expr(:quote, key))}()) ) for key in items]
    kdblock = Expr(:block, kd...)
    expr = quote
        $suitecase_instance = $suitecase # handles if suitecase is not a variable but an expression
        $kdblock
        $suitecase_instance # return RHS of `=` as standard in Julia
    end
    esc(expr)
end


"""
```
@include(filename::AbstractString)
```
mirrors the behavior of include but in a local scope.
See the discussion https://discourse.julialang.org/t/how-to-include-into-local-scope/34634/10.
"""
macro include(filename::AbstractString)
    path = joinpath(dirname(String(__source__.file)), filename)
    return esc(Meta.parse("quote; " * read(path, String) * "; end").args[1])
end
