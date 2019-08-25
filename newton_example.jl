using Test, LinearAlgebra, Random
using DynamicPolynomials, HomotopyContinuation, StaticArrays
import TreeViews, ProjectiveVectors, PolynomialTestSystems

import PolynomialTestSystems: cyclic, cyclooctane, katsura, equations, ipp2, heart, griewank_osborne
const HC = HomotopyContinuation

@polyvar x y
F = SPSystem([1.00001(x-√2)*(x-√2+1e-4)^2, 1e6(x*y+2)])
s = [√2, -√2]
x₀ = s .+ 1e-5

c = HC.newton_cache(F, x₀)
x̂, r = newton(F, x₀, HC.InfNorm(), c; tol=1e-15,
			maxiters=10, precision=PRECISION_FIXED_64, update_all_steps=true,
			jacobian_monitor_update=HC.JAC_MONITOR_UPDATE_ALL)

s - x̂


big_r = evaluate(F, big.(x₀))
big_J = jacobian(F, big.(x₀))

big_d̂ = big_J \ big_r
big_r2 = big_J * big_d̂ -  big_r
big_d̂2 = big_J \ big_r2

r = evaluate(F, x₀)
J = jacobian(F, x₀)

d̂ = J \ r
r2 = J * d̂ -  r
d̂2 = J \ r2
