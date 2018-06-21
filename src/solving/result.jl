export AffineResult, ProjectiveResult, nresults, nfinite, nsingular, natinfinity, nfailed, nnonsingular,
    finite, results, failed, atinfinity, singular, nonsingular, seed, solutions

"""
    Result

Abstract super type of [`ProjectiveResult`](@ref) and [`AffineResult`](@ref).
"""
abstract type Result end

"""
    ProjectiveResult <: Result

The result of a homogenous system of polynomials.
"""
struct ProjectiveResult{T1, T2, T3} <: Result
    pathresults::Vector{PathResult{T1, T2, T3}}
    seed::Int
end

"""
    AffineResult <: Result

The result of an (non-homogenous) system of polynomials.
"""
struct AffineResult{T1, T2, T3} <: Result
    pathresults::Vector{PathResult{T1, T2, T3}}
    seed::Int
end

Base.length(r::Result) = length(r.pathresults)
Base.getindex(r::Result, I) = getindex(r.pathresults, I)

Base.start(r::Result) = start(r.pathresults)
Base.next(r::Result, state) = next(r.pathresults, state)
Base.done(r::Result, state) = done(r.pathresults, state)
Base.endof(r::Result) = endof(r.pathresults)
Base.iteratorsize(::Type{<:Result}) = Base.HasLength()
Base.eltype(r::Result) = eltype(r.pathresults)

"""
    nresults(result)

The number of proper solutions.
"""
nresults(R::Result) = count(issuccess, R)

"""
    nresults(affineresult)

The number of finite solutions.
"""
nfinite(R::AffineResult) = count(isfinite, R)

"""
    nsingular(result; tol=1e10)

The number of singular solutions. A solution is considered singular
if its windingnumber is larger than 1 or the condition number is larger than `tol`.
"""
nsingular(R::Result; tol=1e10) = count(R) do r
    issingular(r, tol)
end

"""
    natinfinity(affineresult)

The number of solutions at infinity.
"""
natinfinity(R::AffineResult) = count(isatinfinity, R)

"""
    nafailed(result)

The number of failed paths.
"""
nfailed(R::Result) = count(isfailed, R)

"""
    nnonsingular(result; tol=1e-10)

The number of non-singular solutions.
"""
nnonsingular(R::Result; tol = 1e10) = count(r -> isnonsingular(r, tol), R)

"""
    seed(result)

The random seed used in the computation.
"""
seed(result::Result) = result.seed

const AffineResults = Union{AffineResult, Vector{<:PathResult}}
const ProjectiveResults = Union{ProjectiveResult, Vector{<:PathResult}}
const Results = Union{Result, Vector{<:PathResult}}

# Filtering
"""
    results(result; onlyreal=false, realtol=1e-6, onlynonsingular=false, singulartol=1e10, onlyfinite=true)

Return all `PathResult`s for which the given conditions apply.

    results(f::Function, result; kwargs...)

Additionally you can apply a transformation `f` on each result.

## Example

```julia
R = solve(F)

# This gives us all solutions considered real (but still as a complex vector).
realsolutions = results(solution, R, onlyreal=true)
```
"""
results(R::Result; kwargs...) = results(identity, R; kwargs...)
function results(f::Function, R::Result;
    onlyreal=false, realtol=1e-6, onlynonsingular=false, singulartol=1e10,
    onlyfinite=true)
    [f(r) for r in R if
        (!onlyreal || isreal(r, realtol)) &&
        (!onlynonsingular || isnonsingular(r, singulartol)) &&
        (!onlyfinite || isfinite(r) || isprojective(r))]
end

"""
    solutions(result, onlyreal=Val{false};realtol=1e-6, onlynonsingular=false, singulartol=1e10, onlyfinite=true)

Return all solution (as `Vector`s) for which the given conditions apply. If `onlyreal` is
`Val{true}` this returns the *real* solutions. This is different than applying `results(solution, result, onlyreal=true)`.

## Example
```julia
julia> @polyvar x y
julia> result = solve([(x-2)y, y+x+3]);
julia> solutions(result, Val{true})
[[2.0, -5.0], [-3.0, 0.0]]
julia> solutions(result)
[[2.0+0.0im, -5.0+0.0im], [-3.0+0.0im, 0.0+0.0im]]
```
"""
function solutions(result::Result, onlyreal=Val{false}; kwargs...)
    if onlyreal === Val{true}
        results(r -> real.(solution(r)), result; onlyreal=true, kwargs...)
    else
        results(r -> solution(r), result; onlyreal=false, kwargs...)
    end
end

"""
    nonsingular(result::AffineResult)

Return all `PathResult`s for which the solution is non-singular.
"""
nonsingular(R::Results; kwargs...) = nonsingular(identity, R; kwargs...)
function nonsingular(f::Function, R::Results; tol=1e10)
    [f(r) for r in R if isnonsingular(r, tol)]
end

"""
    finite(result::AffineResult)

Return all `PathResult`s for which the result is successfull and
the contained `solution` is indeed a solution of the system.

    finite(f::Function, result)

Additionally you can apply a transformation `f` on each result.
"""
finite(R::Results; kwargs...) = finite(identity, R; kwargs...)
function finite(f::Function, R::AffineResults; onlynonsingular=false, tol=1e10)
    if !onlynonsingular
        [f(r) for r in R if isfinite(r)]
    else
        [f(r) for r in R if isfinite(r) && isnonsingular(r, tol)]
    end
end


"""
    singular(result; tol=1e10)

Get all singular solutions. A solution is considered singular
if its windingnumber is larger than 1 or the condition number is larger than `tol`.
"""
singular(R::Result; tol=1e10) = [r for r in R if (issuccess(r) && issingular(r, tol))]


"""
    real(result, tol=1e-6)

Get all results where the solutions are real with the given tolerance `tol`.
See [`isreal`](@ref) for details regarding the determination of 'realness'.
"""
Base.real(R::Results; tol=1e-6) = [r for r in R if isreal(r, tol)]

"""
    failed(result)

Get all results where the path tracking failed.
"""
failed(R::Result) = [r for r in R if isfailed(r)]

"""
    atinfinity(result::AffineResult)

Get all results where the solutions is at infinity.
"""
function atinfinity(R::AffineResults)
    [r for r in R if isatinfinity(r)]
end

function Base.show(io::IO, r::AffineResult)
    println(io, "-----------------------------------------------")
    println(io, "Paths tracked: $(length(r))")
    println(io, "# non-singular finite solutions:  $(nnonsingular(r))")
    println(io, "# singular finite solutions:  $(nsingular(r))")
    println(io, "# solutions at infinity:  $(natinfinity(r))")
    println(io, "# failed paths:  $(nfailed(r))")
    println(io, "Random seed used: $(seed(r))")
    println(io, "-----------------------------------------------")
end

function Base.show(io::IO, r::ProjectiveResult)
    println(io, "-----------------------------------------------")
    println(io, "Paths tracked: $(length(r))")
    println(io, "# non-singular solutions:  $(nnonsingular(r))")
    println(io, "# singular solutions:  $(nsingular(r))")
    println(io, "# failed paths:  $(nfailed(r))")
    println(io, "Random seed used: $(seed(r))")
    println(io, "-----------------------------------------------")
end


function Juno.render(i::Juno.Inline, r::AffineResult)
        t = Juno.render(i, Juno.defaultrepr(r))
        t[:children] = [
            Juno.render(i, Text("Paths tracked → $(length(r))"))]
        #
        n = nnonsingular(r)
        if n > 0
            t_result = Juno.render(i, finite(r))
            t_result[:head] = Juno.render(i, Text("$n finite non-singular $(plural("solution", n))"))
            push!(t[:children], t_result)
        end

        n = nsingular(r)
        if n > 0
            t_result = Juno.render(i, finite(r))
            t_result[:head] = Juno.render(i, Text("$n finite singular $(plural("solution", n))"))
            push!(t[:children], t_result)
        end

        n = natinfinity(r)
        if n > 0
            t_result = Juno.render(i, atinfinity(r))
            t_result[:head] = Juno.render(i, Text("$n $(plural("solution", n)) at ∞"))
            push!(t[:children], t_result)
        end

        n = nfailed(r)
        if n > 0
            t_result = Juno.render(i, failed(r))
            t_result[:head] = Juno.render(i, Text("$n paths failed"))
            push!(t[:children], t_result)
        end
        push!(t[:children], Juno.render(i, Text("Random seed used → $(seed(r))")))

        return t
end

function Juno.render(i::Juno.Inline, r::ProjectiveResult)
        t = Juno.render(i, Juno.defaultrepr(r))
        t[:children] = [
            Juno.render(i, Text("Paths tracked → $(length(r))"))]
        #
        n = nnonsingular(r)
        if n > 0
            t_result = Juno.render(i, nonsingular(r))
            t_result[:head] = Juno.render(i, Text("$n non-singular $(plural("solution", n))"))
            push!(t[:children], t_result)
        end

        n = nsingular(r)
        if n > 0
            t_result = Juno.render(i, singular(r))
            t_result[:head] = Juno.render(i, Text("$n singular $(plural("solution", n))"))
            push!(t[:children], t_result)
        end

        n = nfailed(r)
        if n > 0
            t_result = Juno.render(i, failed(r))
            t_result[:head] = Juno.render(i, Text("$n $(plural("path", n)) failed"))
            push!(t[:children], t_result)
        end
        push!(t[:children], Juno.render(i, Text("Random seed used → $(seed(r))")))

        return t
end

plural(singularstr, n) = n > 1 ? singularstr * "s" : singularstr