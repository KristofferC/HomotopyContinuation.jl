using Pkg;
Pkg.activate(@__DIR__)

using Test, LinearAlgebra, Random
using DynamicPolynomials, HomotopyContinuation, StaticArrays
import TreeViews, ProjectiveVectors, PolynomialTestSystems

import PolynomialTestSystems: cyclic, cyclooctane, katsura, equations, ipp2, heart, griewank_osborne
const HC = HomotopyContinuation

F = equations(griewank_osborne())

prob, starts = problem_startsolutions(F; seed=12345)
S = collect(starts)

x₀ = S[1] .+ 1e-4
t = 1.0
H = HC.HomotopyWithCache(prob.homotopy, x₀, t)
corr_cache = HC.cache(HC.NewtonCorrector2(), H, x₀, t)

JM = HC.JacobianMonitor(jacobian(H, x₀, t))
x̄ = similar(x₀)

R1 = HC.newton!(x̄, H, x₀, t, JM, HC.InfNorm(), corr_cache; tol=1e-3)
@test R1.iters == 1
@test R1.accuracy ≤ 1.1e-4
@test R1.norm_Δx₀ < 1.1e-4
@test isnan(R1.ω)
@test norm(x̄ - S[1]) < 1e-6


R2 = HC.newton!(x̄, H, x₀, t, JM, HC.InfNorm(), corr_cache; tol=1e-7)
@test R2.return_code == HC.NEWT_CONVERGED
@test R2.iters == 2
@test R2.accuracy < 1e-7
@test !isnan(R2.ω + R2.ω₀ + R2.θ + R2.θ₀)
@test R2.norm_Δx₀ < 1.1e-4
@test norm(x̄ - S[1]) < 1e-10


R3 = HC.newton!(x̄, H, [1e-5, 1e-5], 0.0, JM, HC.InfNorm(), corr_cache)
@test R3.return_code == HC.NEWT_TERMINATED




@polyvar x y
G = [x^2+2, y^2+4]
tracker, starts = coretracker2_startsolutions(G; seed=12345,
    predictor=Pade21(), auto_scaling=true)
S = collect(starts)
track(tracker, S[1])

tracker.state.jacobian
