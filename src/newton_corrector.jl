struct NewtonCorrector2 end


struct NewtonCorrector2Cache{T}
    Δx::Vector{Complex{T}}
    r::Vector{Complex{T}}
end

function NewtonCorrector2Cache(H::HomotopyWithCache, x::AbstractVector, t::Number)
    rx = evaluate(H, x, t)
    T = complex(float(eltype(rx)))
    Δx = Vector{T}(undef, length(x))
    r = Vector{T}(undef, length(rx))
    NewtonCorrector2Cache(Δx, r)
end

cache(::NewtonCorrector2, H, x, t) = NewtonCorrector2Cache(H, x, t)

@doc """
    NewtonCorrector2ReturnCode

The possible return codes of Newton's method

* `NEWT_CONVERGED`
* `NEWT_TERMINATED`
* `NEWT_MAX_ITERS`
"""
@enum NewtonCorrector2Codes begin
    NEWT_CONVERGED
    NEWT_TERMINATED
    NEWT_MAX_ITERS
end

struct NewtonCorrector2Result{T}
    return_code::NewtonCorrector2Codes
    accuracy::T
    iters::Int
    ω₀::Float64
    ω::Float64
    θ₀::Float64
    θ::Float64
    norm_Δx₀::T
end

Base.show(io::IO, ::MIME"application/prs.juno.inline", r::NewtonCorrector2Result) = r
Base.show(io::IO, result::NewtonCorrector2Result) = print_fieldnames(io, result)
is_converged(R::NewtonCorrector2Result) = R.return_code == NEWT_CONVERGED

function newton!(x̄::AbstractVector,
    H::HomotopyWithCache,
    x₀::AbstractVector,
    t::Number,
    JM::JacobianMonitor,
    norm::AbstractNorm,
    cache::NewtonCorrector2Cache;
    tol::Float64 = 1e-6,
    max_iters::Int = 3,
    debug::Bool = true)

    # Setup values
    x̄ .= x₀
    acc = limit_acc = norm_Δxᵢ = norm_Δxᵢ₋₁ = norm_Δx₀ = Inf
    ω = ω₀ = θ₀ = θ = NaN
    @unpack Δx, r = cache
    # alias to make logic easier
    xᵢ₊₁ = xᵢ = x̄

    for i ∈ 1:max_iters
        debug && println("i = ", i)

        compute_jacobian = i == 1 || i < max_iters
        if compute_jacobian
            evaluate_and_jacobian!(r, jacobian(JM), H, xᵢ, t)
            updated!(JM)
        else
            evaluate!(r, H, xᵢ, t)
        end

        # Update cond info etc?
        LA.ldiv!(Δx, JM, r, norm, JAC_MONITOR_UPDATE_NOTHING)

        norm_Δxᵢ₋₁ = norm_Δxᵢ
        norm_Δxᵢ = Float64(norm(Δx))

        debug && println("||Δxᵢ|| = ", norm_Δxᵢ)

        xᵢ₊₁ .= xᵢ .- Δx

        if i == 1
            acc = norm_Δx₀ = norm_Δxᵢ
            if acc ≤ tol
                return NewtonCorrector2Result(NEWT_CONVERGED, acc, i, ω₀, ω, θ₀, θ, norm_Δx₀)
            end
            continue
        end

        if i == 2
            θ = θ₀ = norm_Δxᵢ / norm_Δxᵢ₋₁
            @show θ
            ω = ω₀ = 2θ / norm_Δxᵢ₋₁
        elseif i > 2
            θ = norm_Δxᵢ / norm_Δxᵢ₋₁
            ω = max(ω, 2θ / norm_Δxᵢ₋₁)
        end
        acc = norm_Δxᵢ / (1.0 - min(0.5, 2θ^2))
        if acc ≤ tol
            return NewtonCorrector2Result(NEWT_CONVERGED, acc, i, ω₀, ω, θ₀, θ, norm_Δx₀)
        end
        if θ ≥ 0.5
            return NewtonCorrector2Result(NEWT_TERMINATED, acc, i, ω₀, ω, θ₀, θ, norm_Δx₀)
        end
    end

    return return NewtonCorrector2Result(NEWT_MAX_ITERS, acc, max_iters, ω₀, ω, θ₀, θ, norm_Δx₀)
end
