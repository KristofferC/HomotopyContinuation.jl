export CoreTracker2, CoreTracker2Result, CoreTracker2Status, CoreTracker2Options,
        coretracker2, coretracker2_startsolutions, affine_tracking,
        track, track!, setup!, iterator,
        current_x, current_t, current_Δt, iters, status, accuracy, max_corrector_iters,
        max_step_size , refinement_accuracy, max_refinement_iters,
        set_accuracy!, set_max_corrector_iters!, set_refinement_accuracy!,
        set_max_refinement_iters!, set_max_step_size!, digits_lost, options



####################
# CoreTracker2State #
####################

@doc """
    enum CoreTracker2Status

The possible states the `CoreTracker2` can achieve are

* `CT_SUCCESS`: Indicates a successfull completed tracking
* `CT_TRACKING`: The tracking is still in progress
* `CT_TERMINATED_MAX_ITERS`: Tracking terminated since maximal iterations reached.
* `CT_TERMINATED_INVALID_STARTVALUE`: Tracking terminated since the provided start value was invalid.
* `CT_TERMINATED_STEP_SIZE_TOO_SMALL`
* `CT_TERMINATED_SINGULARITY`
* `CT_TERMINATED_ILL_CONDITIONED`
"""
@enum CoreTracker2Status begin
    CT_SUCCESS
    CT_TRACKING
    CT_TERMINATED_MAX_ITERS
    CT_TERMINATED_INVALID_STARTVALUE
    CT_TERMINATED_STEP_SIZE_TOO_SMALL
    CT_TERMINATED_SINGULARITY
    CT_TERMINATED_ILL_CONDITIONED
end

is_success(S::CoreTracker2Status) = S == CT_SUCCESS
is_terminated(S::CoreTracker2Status) = S ≠ CT_TRACKING

####################
# CoreTracker2Result #
####################

"""
     CoreTracker2Result{V<:AbstractVector}

Containing the result of a tracked path. The fields are
* `returncode::CTStatus` If the tracking was successfull then it is `CT_SUCCESS`.
* `x::V` The result.
* `t::ComplexF64` The `t` when the path tracker stopped.
* `accuracy::Float64`: The estimated accuracy of `x`.
"""
struct CoreTracker2Result{V <: AbstractVector}
    returncode::CoreTracker2Status
    x::V
    t::ComplexF64
    accuracy::Float64
    accepted_steps::Int
    rejected_steps::Int
end

function CoreTracker2Result(tracker)
    state = tracker.state
    CoreTracker2Result(state.status,
          copy(state.x), state.segment[state.s],
          state.accuracy,
          state.accepted_steps,
          state.rejected_steps)
end

Base.show(io::IO, result::CoreTracker2Result) = print_fieldnames(io, result)
Base.show(io::IO, ::MIME"application/prs.juno.inline", result::CoreTracker2Result) = result


##################
## PATH TRACKER ##
##################

mutable struct StepSizeModel
    ω::Float64
    expected_Δx₀::Float64
end
StepSizeModel() = StepSizeModel(NaN, NaN)

function init!(model::StepSizeModel)
    model.ω = model.expected_Δx₀ = NaN
    model
end
Base.show(io::IO, opts::StepSizeModel) = print_fieldnames(io, opts)
Base.show(io::IO, ::MIME"application/prs.juno.inline", opts::StepSizeModel) = opts

######################
# CoreTracker2Options #
######################


"""
    CoreTracker2Options

The set of options set for a [`CoreTracker2`](@ref). See the description of [`CoreTracker2`](@ref)
for all possible options.
"""
mutable struct CoreTracker2Options
    accuracy::Float64
    max_corrector_iters::Int
    refinement_accuracy::Float64
    max_refinement_iters::Int
    max_steps::Int
    initial_step_size::Float64
    min_step_size::Float64
    max_step_size::Float64
    simple_step_size_alg::Bool
    update_patch::Bool
    max_lost_digits::Float64
    terminate_ill_conditioned::Bool
    steps_jacobian_info_update::Int
    # Not changeable options
    precision::PrecisionOption
    logarithmic_time_scale::Bool
end

function CoreTracker2Options(;
    accuracy = 1e-7,
    initial_step_size = 0.1,
    max_corrector_iters::Int = 2,
    max_refinement_iters = 5,
    max_step_size = Inf,
    min_step_size = 1e-14,
    parameter_homotopy = false,
    max_steps = parameter_homotopy ? 10_000 : 1_000,
    precision::PrecisionOption = PRECISION_FIXED_64,
    max_lost_digits = default_max_lost_digits(precision, accuracy),
    refinement_accuracy = 1e-8,
    simple_step_size_alg = false,
    steps_jacobian_info_update::Int = 3,
    terminate_ill_conditioned::Bool = true,
    update_patch = true,
    logarithmic_time_scale = false)

    CoreTracker2Options(accuracy, max_corrector_iters, refinement_accuracy,
            max_refinement_iters, max_steps, initial_step_size, min_step_size,
            max_step_size, simple_step_size_alg, update_patch, max_lost_digits, terminate_ill_conditioned,
            steps_jacobian_info_update, precision, logarithmic_time_scale)
end

Base.show(io::IO, opts::CoreTracker2Options) = print_fieldnames(io, opts)
Base.show(io::IO, ::MIME"application/prs.juno.inline", opts::CoreTracker2Options) = opts

const eps_double_64 = Float64(eps(Double64))
function default_max_lost_digits(prec::PrecisionOption, accuracy::Float64)
    if prec == PRECISION_FIXED_64
        # maximal_digits_available - digits necessary - buffer
        -log10(eps()) + log10(accuracy) + 1
    else
        # if higher precision is available we will more like be constrained
        # by the fact that the jacobian cannot be too ill-conditioned
        min(-log10(eps()) - 3, -log10(eps_double_64) + log10(accuracy) - 3)
    end
end

####################
# CoreTracker2State #
####################

mutable struct CoreTracker2State{T,AV <: AbstractVector{Complex{T}},AN <: AbstractNorm,MaybePatchState <: Union{AbstractAffinePatchState,Nothing}}
    x::AV # current x
    x̂::AV # last prediction
    x̄::AV # candidate for new x
    ẋ::Vector{Complex{T}} # derivative at current x

    segment::ComplexSegment
    s::Float64 # current step length (0 ≤ s ≤ length(segment))
    Δs::Float64 # current step size
    Δs_prev::Float64 # previous step size

    norm::AN
    step_size::StepSizeModel
    accuracy::Float64
    jacobian::JacobianMonitor{T}
    steps_jacobian_info_update::Int
    status::CoreTracker2Status
    patch::MaybePatchState

    accepted_steps::Int
    rejected_steps::Int
    last_step_failed::Bool
    consecutive_successfull_steps::Int
end

function CoreTracker2State(H, x₁::AbstractVector, t₁, t₀, options::CoreTracker2Options,
                           patch::Union{Nothing,AbstractAffinePatchState},
                           norm::AbstractNorm)
    x = isa(x₁, SVector) ? Vector(x₁) : copy(x₁)
    x̂, x̄ = copy(x), copy(x)
    ẋ = Vector(x)
    step_size_model = StepSizeModel()
    segment = ComplexSegment(promote(complex(t₁), complex(t₀))...)
    s = 0.0
    Δs = convert(Float64, min(options.initial_step_size, length(segment), options.max_step_size))
    Δs_prev = 0.0
    accuracy = 0.0
    JM = JacobianMonitor(jacobian(H, x, t₁))
    steps_jacobian_info_update = 0
    accepted_steps = rejected_steps = 0
    status = CT_TRACKING
    last_step_failed = false
    consecutive_successfull_steps = 0
    CoreTracker2State(x, x̂, x̄, ẋ, segment, s, Δs, Δs_prev, norm, step_size_model, accuracy,
        JM, steps_jacobian_info_update, status, patch,
        accepted_steps, rejected_steps, last_step_failed, consecutive_successfull_steps)
end

@inline function Base.getproperty(state::CoreTracker2State, sym::Symbol)
    if sym === :t
        return state.segment[state.s]
    elseif sym === :Δt
        return state.segment[state.Δs] - state.segment.start
    else # fallback to getfield
        return getfield(state, sym)
    end
end

Base.show(io::IO, state::CoreTracker2State) = print_fieldnames(io, state)
Base.show(io::IO, ::MIME"application/prs.juno.inline", state::CoreTracker2State) = state


##################
## CoreTracker2 ##
##################
    """
    CoreTracker2(problem::AbstractProblem, x₁; kwargs...)

Construct a `CoreTracker2` from the given `problem` to track elements of type `x₁`.
The path is tracked using a predictor-corrector scheme. The recommended methods to construct
a `CoreTracker2` are [`coretracker2`](@ref) and [`coretracker2_startsolutions`](@ref).
Note that a `CoreTracker2` is also a (mutable) iterator.

## Keyword arguments
* `accuracy=1e-7`: The accuracy required during the path tracking.
* `corrector::AbstractCorrector`: The corrector used during in the predictor-corrector scheme. The default is [`NewtonCorrector`](@ref).
* `initial_step_size=0.1`: The size of the first step.
* `max_corrector_iters=2`: The maximal number of correction steps in a single step.
* `max_lost_digits::Real`: The tracking is terminated if we estimate that we loose more than `max_lost_digits` in the linear algebra steps. This threshold depends on the `precision` argument.
* `max_refinement_iters=5`: The maximal number of correction steps used to refine the final value.
* `max_steps=1_000`: The maximal number of iterations the path tracker has available. Note that this changes to `10_000` for parameter homotopies.
* `max_step_size=Inf`: The maximal step size.
* `min_step_size=1e-14`: The minimal step size.
* `precision::PrecisionOption=PRECISION_FIXED_64`: The precision used for evaluating the residual in Newton's method.
* `predictor::AbstractPredictor`: The predictor used during in the predictor-corrector scheme. The default is [`Heun`](@ref)()`.
* `refinement_accuracy=1e-8`: The precision required for the final value.
* `simple_step_size_alg=false`: Use a more simple step size algorithm.
* `steps_jacobian_info_update::Int=1`: Every n-th step a linear system will be solved using a QR factorization to obtain an estimate for the condition number of the Jacobian.
* `terminate_ill_conditioned::Bool=true`: Indicates whether the path tracking should be terminated for ill-conditioned paths. A path is considerd ill-conditioned if the condition number of the Jacobian is larger than ≈1e14 or if it is larger than 1e`max_lost_digits`.
"""
struct CoreTracker2{T,AV <: AbstractVector{Complex{T}},AN <: AbstractNorm,P <: AbstractPredictorCache,AP <: Union{Nothing,AbstractAffinePatchState},H <: HomotopyWithCache}
    homotopy::H
    predictor::P
    corrector::NewtonCorrector2Cache{T}
    # these are mutable
    state::CoreTracker2State{T,AV,AN,AP}
    options::CoreTracker2Options
    # buffer
    dt::Vector{Complex{T}}
end

function CoreTracker2(prob::AbstractProblem, x₁; kwargs...)
    CoreTracker2(prob.homotopy, embed(prob, x₁), complex(1.0), complex(0.0), prob; kwargs...)
end

# Tracking in affine space
function CoreTracker2(homotopy::AbstractHomotopy,
    x₁::AbstractVector,
    t₁::Number,
    t₀::Number,
    prob::AbstractProblem;
    patch = nothing,
    corrector::NewtonCorrector2 = NewtonCorrector2(),
    predictor::AbstractPredictor = default_predictor(x₁),
    logarithmic_time_scale = false,
    auto_scaling = true,
    norm::AbstractNorm = InfNorm(),
    kwargs...)

    patch === nothing || throw(ArgumentError("You can only pass `patch=$(patch)` if `affine_tracking=false`."))

    options = CoreTracker2Options(;
        parameter_homotopy = isa(homotopy, ParameterHomotopy),
        kwargs...)

    # We close over the homotopy and its cache to be able to pass things around more easily
    H = HomotopyWithCache(homotopy, Vector(x₁), t₁)

    used_norm = auto_scaling ? WeightedNorm(norm, x₁) : norm
    # We have to make sure that the element type of x is invariant under evaluation
    state = CoreTracker2State(H, indempotent_x(H, x₁, t₁), t₁, t₀, options, nothing, used_norm)

    pred_cache = cache(predictor, H, state.x, state.ẋ, t₁)
    corr_cache = cache(corrector, H, state.x, t₁)
    dt = Vector{eltype(state.ẋ)}(undef, first(size(H)))

    CoreTracker2(H, pred_cache, corr_cache, state, options, dt)
end


# Tracking in Projective Space
function CoreTracker2(homotopy::AbstractHomotopy, x₁::ProjectiveVectors.PVector, t₁, t₀, prob::AbstractProblem;
    patch = has_dedicated_homvars(prob.vargroups) ? EmbeddingPatch() : OrthogonalPatch(),
    corrector::NewtonCorrector2 = NewtonCorrector2(),
    predictor::AbstractPredictor = default_predictor(x₁),
    auto_scaling = true,
    norm::AbstractNorm = InfNorm(),
    simple_step_size_alg = !isa(patch, EmbeddingPatch),
    kwargs...)

    options = CoreTracker2Options(;
                parameter_homotopy = isa(homotopy, ParameterHomotopy),
                kwargs...)

    if homotopy isa PatchedHomotopy
        error("You cannot pass a `PatchedHomotopy` to CoreTracker2. Instead pass the homotopy and patch separate.")
    end

    patch_state = state(patch, x₁)

    # We close over the patch state, the homotopy and its cache
    # to be able to pass things around more easily
    H = HomotopyWithCache(PatchedHomotopy(homotopy, patch_state), x₁, t₁)

    if is_global_patch(patch) && auto_scaling
        used_norm = WeightedNorm(norm, x₁)
    else
        used_norm = norm
    end
    # We have to make sure that the element type of x is invariant under evaluation
    state = CoreTracker2State(H, indempotent_x(H, x₁, t₁), t₁, t₀, options, patch_state, used_norm)

    pred_cache = cache(predictor, H, state.x, state.ẋ, t₁)
    corr_cache = cache(corrector, H, state.x, t₁)
    dt = Vector{eltype(state.ẋ)}(undef, first(size(H)))

    CoreTracker2(H, pred_cache, corr_cache, state, options, dt)
end


default_predictor(x::AbstractVector) = Pade21()
# Do not really understand this but Heun doesn't work that great for multi-homogeneous tracking
default_predictor(x::ProjectiveVectors.PVector{T,1}) where {T} = Pade21()
default_predictor(x::ProjectiveVectors.PVector{T,N}) where {T,N} = Euler()

"""
indempotent_x(H, x₁, t₁)

This returns a vector similar to `x₁` but with an element type which is invariant under evaluation.
"""
function indempotent_x(H::AbstractHomotopy, x₁, t₁)
    u = Vector{Any}(undef, size(H)[1])
    evaluate!(u, H, x₁, t₁)

    if isa(x₁, SVector)
        indem_x = Vector{promote_type(typeof(u[1]), ComplexF64)}(undef, length(x₁))
    else
        indem_x = similar(x₁, promote_type(typeof(u[1]), ComplexF64))
    end
    indem_x .= x₁
    indem_x
end

Base.show(io::IO, C::CoreTracker2) = print(io, "CoreTracker2 tracking a path of type $(typeof(C.state.x))")
Base.show(io::IO, ::MIME"application/prs.juno.inline", x::CoreTracker2) = x


###########
## Alias ##
###########
const CT = CoreTracker2
const CTS = CoreTracker2State
const CTO = CoreTracker2Options
const CTR = CoreTracker2Result

#######################
## Correct & Tangent ##
#######################

function compute_ẋ!(tracker::CT; update_jacobian::Bool = true)
    @unpack homotopy, state, options, dt = tracker

    if update_jacobian
        jacobian_and_dt!(jacobian(state.jacobian), dt, homotopy, state.x, state.t)
        updated!(state.jacobian)
    else
        dt!(dt, homotopy, state.x, state.t)
    end
    # flip sign of entries in dt
    for i in eachindex(dt); dt[i] = -dt[i]; end
    # solve the linear system to compute ẋ
    LA.ldiv!(state.ẋ, state.jacobian, dt)
    state.ẋ
end


function correct!(
    x̄::AbstractVector,
    tracker::CT,
    x::AbstractVector = tracker.state.x,
    t::Number = tracker.state.t,
    norm::AbstractNorm = tracker.state.norm;
    accuracy::Float64 = tracker.options.accuracy,
    max_iters::Int = tracker.options.max_corrector_iters+1,
    precision::PrecisionOption = tracker.options.precision,
    update_jacobian_infos::Bool = false)
    newton!(
        x̄, tracker.homotopy, x, t, tracker.state.jacobian,
        norm, tracker.corrector;
        tol=accuracy, max_iters=max_iters)
end


####################
## INITIALIZATION ##
####################

embed!(x::ProjectiveVectors.PVector, y) = ProjectiveVectors.embed!(x, y)
embed!(x::AbstractVector, y) = x .= y

function check_start_value(tracker::CT, x, t)
    embed!(tracker.state.x̄, x)
    init!(tracker.state.norm, tracker.state.x̄)
    result = correct!(tracker.state.x̄, tracker, x, t; update_jacobian_infos = true)
    is_converged(result)
end

function check_start_value!(tracker::CT)
    result = correct!(tracker.state.x̄, tracker; update_jacobian_infos = true)
    if is_converged(result)
        tracker.state.x .= tracker.state.x̄
    else
        tracker.state.status = CT_TERMINATED_INVALID_STARTVALUE
    end
    nothing
end


"""
init!(tracker::CT, x₁, t₁, t₀;
      setup_patch::Bool = tracker.options.update_patch,
      loop::Bool = false,
      check_start_value::Bool = !loop)

Setup `tracker` to track `x₁` from `t₁` to `t₀`. Use this if you want to use the
`tracker` as an iterator.
"""
function init!(tracker::CT, x₁::AbstractVector, t₁, t₀;
    setup_patch::Bool = tracker.options.update_patch,
    loop::Bool = false,
    check_start_value::Bool = !loop)
    @unpack state, predictor, homotopy = tracker

    init!(state, x₁, t₁, t₀, tracker.options, setup_patch, loop)
    check_start_value && check_start_value!(tracker)
    if !loop
        compute_ẋ!(tracker; update_jacobian=true)
        init!(predictor, homotopy, state.x, state.ẋ, state.t, state.jacobian)
    end
    tracker
end

function init!(state::CTS, x₁::AbstractVector, t₁, t₀, options::CTO, setup_patch::Bool, loop::Bool)
    state.segment = ComplexSegment(t₁, t₀)
    state.s = 0.0
    state.Δs = min(options.initial_step_size, length(state.segment), options.max_step_size)
    state.Δs_prev = 0.0
    state.status = CT_TRACKING
    embed!(state.x, x₁)
    setup_patch && state.patch !== nothing && init!(state.patch, state.x)
    state.last_step_failed = false
    state.consecutive_successfull_steps = 0
    if !loop
        state.accuracy = 0.0
        state.accepted_steps = state.rejected_steps = 0
        init!(state.step_size)
        init!(state.norm, state.x)
        init!(state.jacobian)
        state.steps_jacobian_info_update = 0
    end

    state
end



##############
## TRACKING ##
##############

"""
track(tracker, x₁, t₁=1.0, t₀=0.0; options...)::CTR

Track a value `x₁` from `t₁` to `t₀` using the given `CoreTracker2` `tracker`.
This returns a `CoreTracker2Result`. This modifies `tracker`.
See [`track!`](@ref) for the possible options.
"""
function track(tracker::CT, x₁::AbstractVector, t₁ = 1.0, t₀ = 0.0; kwargs...)
    track!(tracker, x₁, t₁, t₀; kwargs...)
    CoreTracker2Result(tracker)
end

"""
 track!(tracker, x₁, t₁=1.0, t₀=0.0; setup_patch=true, check_start_value=true, loop::Bool=false)::CTR

Track a value `x₁` from `t₁` to `t₀` using the given `CoreTracker2` `tracker`.
Returns one of the enum values of `CoreTracker2Status` indicating the status.
If the tracking was successfull it is `CT_SUCCESS`.
If `setup_patch` is `true` then [`setup!`](@ref) is called at the beginning
of the tracking.

track!(x₀, tracker, x₁, t₁=1.0, t₀=0.0; options...)::CTStatus

Additionally also stores the result in `x₀` if the tracking was successfull.
"""
function track!(x₀, tracker::CT, x₁, t₁ = 1.0, t₀ = 0.0; kwargs...)
    track!(tracker, x₁, t₁, t₀; kwargs...)
    retcode = status(tracker)
    if is_success(status(tracker))
        x₀ .= current_x(tracker)
    end
    retcode
end
function track!(tracker::CT, x₁, t₁ = 1.0, t₀ = 0.0;
    setup_patch::Bool = tracker.options.update_patch,
    loop::Bool = false,
    check_start_value::Bool = !loop)

    @unpack state = tracker

    t₁ == t₀ && return (state.status = CT_SUCCESS)

    init!(tracker, x₁, t₁, t₀;
         setup_patch=setup_patch,
         check_start_value=check_start_value,
         loop=loop)

    while !is_terminated(state.status)
        step!(tracker)
        check_terminated!(tracker)
    end

    state.status == CT_SUCCESS && refine!(tracker)

    state.status
end


"""
    step!(tracker::CT)

Perform one step of the path tracker. This tries to move forward by `Δt`, i.e.,
obtaining an approximation of `x(t+Δt)`.
Returns `true` if the step was successfull.
"""
function step!(tracker::CT; debug::Bool = true)
    @unpack homotopy, predictor, state, options = tracker
    @unpack x, x̂, x̄, ẋ = state

    # @unpack currently doesn't work with getproperty
    t, Δt = state.t, state.Δt

    debug && printstyled("\nt: ", round(t, sigdigits=5),
                         " Δt: ", round(Δt; sigdigits=5), "\n"; color=:yellow)

    # Use the current approximation of x(t) to obtain estimate
    # x̂ ≈ x(t + Δt) using the provided predictor
    predict!(x̂, predictor, homotopy, x, t, Δt, ẋ, state.jacobian)

    # Correct the predicted value x̂ to obtain x̄.
    # If the correction is successfull we have x̄ ≈ x(t+Δt).
    result = correct!(x̄, tracker, x̂, t + Δt)

    debug && show(result)

    if is_converged(result)
        # move forward
        x .= x̄
        state.s += state.Δs
        state.accuracy = result.accuracy

        # update step size
        state.Δs_prev = state.Δs
        update_stepsize!(tracker, result)

        # If we use dynamic affine charts, now we update to a new one
        if state.patch !== nothing && options.update_patch
            changepatch!(state.patch, x)
        end

        # If we use a weighted norm, now we update this
        update!(state.norm, x)

        # Compute the new tangent ẋ(t)
        compute_ẋ!(tracker)

        # tell the predictors about the new derivative if they need to update something
        update!(predictor, homotopy, x, ẋ, t + Δt, state.jacobian)

        # Update other state
        state.accepted_steps += 1
        state.steps_jacobian_info_update += 1
    else
        # Step failed, so we have to try with a new (smaller) step size
        update_stepsize!(tracker, result)

        # Update other state
        state.rejected_steps += 1
    end
    state.last_step_failed = is_converged(result)
end

## Step size update

function update_stepsize!(tracker::CT, result::NewtonCorrector2Result)
    @unpack state, options = tracker

    if options.simple_step_size_alg
        Δs = simple_step_size_alg!(state, options, result)
    else
        Δs = adaptive_step_size_alg!(state, options, result, order(tracker.predictor))
    end

    # Make sure to not overshoot.
    Δs_cap = length(state.segment) - state.s
    state.Δs = min(Δs_cap, Δs)
    # We do not check minimal step size in the case that we need a (very) small step
    # to finish the tracking
    Δs_cap < Δs && check_min_step_size!(state, options)

    nothing
end

function check_min_step_size!(state::CTS, options::CTO)
    terminate = let
        if options.logarithmic_time_scale
            exp(-state.s) - exp(-(state.s + Δs)) < options.min_step_size
        else
            state.Δs < options.min_step_size
        end
    end
    if terminate
        state.status = CT_TERMINATED_STEP_SIZE_TOO_SMALL
    end
    nothing
end


## Simple step size algorithm

function simple_step_size_alg!(state::CTS, options::CTO,
    result::NewtonCorrector2Result)

    if !is_converged(result)
        state.consecutive_successfull_steps = 0
        return state.Δs / 2
    end

    state.consecutive_successfull_steps = (state.consecutive_successfull_steps + 1) % 5
    state.consecutive_successfull_steps == 0 ? 2state.Δs : state.Δs
end

## Adaptive step size algorithm

"Use the adaptive step size control described in https://arxiv.org/abs/1902.02968"
function adaptive_step_size_alg!(state::CTS, options::CTO, result::NewtonCorrector2Result, ord::Int)

    model = state.step_size

    # We have to handle the special case that there is only 1 iteration
    # in this case we cannot estimate ω and therefore just assume ω = 2
    # Also note ||x̂-x|| = ||Δx₀||
    if result.iters == 1
        ω = isnan(model.ω) ? 2.0 : 0.5model.ω
        d_x̂_x̄ = result.norm_Δx₀
    else
        ω = result.ω₀
        d_x̂_x̄ = state.norm(state.x̂, state.x)
    end
    Δx₀ = result.norm_Δx₀
    if is_converged(result)
        δ_N_ω = δ(options, ω, 0.5)
        # This is to handle the edge case that g(δ_N_ω) >> (ω * d_x̂_x̄) > 0 but
        # at the same time δ_N_ω < eps(). Since then g(δ_N_ω) = 0
        δ_N_ω = max(δ_N_ω, 1e-15) #
        λ = g(δ_N_ω) / (ω * d_x̂_x̄)
        Δs = nthroot(λ, ord) * state.Δs
        if state.last_step_failed
            Δs = min(Δs, state.Δs)
        end
    else
        if isfinite(model.ω)
            ω = max(ω, model.ω)
        end
        δ_N_ω = δ(options, ω, 0.5)
        ω_η = 0.5ω * Δx₀
        if δ_N_ω < ω_η
            λ = g(δ_N_ω) / g(ω_η)
            Δs = min(nthroot(λ, ord), 0.9) * state.Δs
        else
            Δs = 0.5state.Δs
        end
        # If we fail consecutively reduce step size more aggressively.
        if state.last_step_failed
            Δs = min(Δs, 0.75 * state.Δs)
        end
    end
    model.expected_Δx₀ = 2δ_N_ω / ω
    model.ω = ω
    Δs
end

g(Θ) = sqrt(1. + 4Θ) - 1.0
δ(opts::CTO, ω, μ) = min(√(0.5ω) * τ(opts, μ), 0.25)
function τ(opts::CTO, μ)
    # special case most common case: 2(2 - 0.5) == 3
    if μ == 0.5 && opts.max_corrector_iters == 2
        cbrt(opts.accuracy)
    else
        opts.accuracy^(1 / (2(opts.max_corrector_iters - μ)))
    end
end


function check_terminated!(tracker::CT)
    # an equality is here okay, since the final step size is computed from
    # Δs = length(tracker.state.segment) - tracker.state.s
    if tracker.state.s ≈ length(tracker.state.segment)
        tracker.state.status = CT_SUCCESS
    elseif iters(tracker) ≥ tracker.options.max_steps
        tracker.state.status = CT_TERMINATED_MAX_ITERS
    end
    nothing
end

function refine!(tracker::CT, accuracy = tracker.options.refinement_accuracy)
    if tracker.state.accuracy < accuracy
        return
    end
    result = correct!(tracker.state.x̄, tracker;
            accuracy = accuracy,
            max_iters = tracker.options.max_refinement_iters)
    if is_converged(result)
        tracker.state.x .= tracker.state.x̄
        tracker.state.accuracy = result.accuracy
    end
    nothing
end


"""
affine_tracking(tracker)

Returns `true` if the path tracker tracks in affine space. If `false` then then the path tracker
tracks in some affine chart of the projective space.
"""
affine_tracking(tracker::CT) = !isa(tracker.state.x, ProjectiveVectors.PVector)

#################
## Query State ##
#################
"""
 current_t(tracker::CT)

Current `t`.
"""
current_t(tracker::CT) = current_t(tracker.state)
current_t(state::CTS) = state.t

"""
 current_Δt(tracker::CT)

Current step_size `Δt`.
"""
current_Δt(tracker::CT) = current_Δt(tracker.state)
current_Δt(state::CTS) = state.Δt

"""
 steps(tracker::CT)

Current number of steps.
"""
steps(tracker::CT) = steps(tracker.state)
steps(state::CTS) = state.accepted_steps + state.rejected_steps

@deprecate iters(tracker::CT) steps(tracker)



"""
 status(tracker::CT)

Current status.
"""
status(tracker::CT) = status(tracker.state)
status(state::CTS) = state.status

"""
current_x(tracker::CT)

Return the current value of `x`.
"""
current_x(tracker::CT) = current_x(tracker.state)
current_x(state::CTS) = state.x


"""
LinearAlgebra.cond(tracker::CT)

Returns the currently computed approximation of the condition number of the
Jacobian.
"""
cond(tracker::CT) = LinearAlgebra.cond(tracker.state)
cond(state::CTS) = state.jacobian.cond


"""
digits_lost(tracker::CT)

Returns the currently computed approximation of the number of digits lost
during the linear system solving in Newton's method.
"""
digits_lost(tracker::CT) = digits_lost(tracker.state)
digits_lost(state::CTS) = unpack(state.jacobian.digits_lost, 0.0)


"""
norm(tracker::CT)

Returns the norm used to compute distances during the path tracking.
"""
norm(tracker::CT) = norm(tracker.state)
norm(state::CTS) = state.norm

"""
options(tracker::CT)

Returns the options used in the tracker.
"""
options(tracker::CT) = tracker.options

##################
# Modify options #
##################
"""
 accuracy(tracker::CT)

Current accuracy.
"""
accuracy(tracker::CT) = tracker.options.accuracy

"""
 set_accuracy!(tracker::CT, accuracy, update_max_lost_digits=true)

Set the current accuracy to `accuracy`. If `update_max_lost_digits` is `true` then
the setting `max_lost_digits` will be updated to the default setting.
"""
function set_accuracy!(tracker::CT, accuracy, update_max_lost_digits::Bool = true)
    @unpack options = tracker
    options.accuracy = accuracy
    if update_max_lost_digits
        options.max_lost_digits = default_max_lost_digits(options.precision, accuracy)
    end
    tracker
end

"""
 refinement_accuracy(tracker::CT)

Current refinement accuracy.
"""
refinement_accuracy(tracker::CT) = tracker.options.refinement_accuracy

"""
 set_max_refinement_iters!(tracker::CT, accuracy)

Set the current refinement accuracy to `accuracy`.
"""
set_refinement_accuracy!(T::CT, accuracy) = T.options.refinement_accuracy = accuracy

"""
 max_refinement_iters(tracker::CT)

Current refinement max_steps.
"""
max_refinement_iters(T::CT) = T.options.max_refinement_iters

"""
 set_max_refinement_iters!(tracker::CT, n)

Set the current refinement max_steps to `n`.
"""
set_max_refinement_iters!(T::CT, n) = T.options.max_refinement_iters = n

"""
 max_corrector_iters(tracker::CT)

Current correction max_steps.
"""
max_corrector_iters(T::CT) = T.options.max_corrector_iters

"""
 set_max_corrector_iters!(tracker::CT, n)

Set the correction max_steps to `n`.
"""
set_max_corrector_iters!(T::CT, n) = T.options.max_corrector_iters = n

"""
 max_step_size (tracker::CT)

Current maximal step size.
"""
max_step_size(T::CT) = T.options.max_step_size

"""
 set_max_corrector_iters!(tracker::CT, Δs)

Set the maximal step size to `Δs`.
"""
set_max_step_size!(T::CT, Δs) = T.options.max_step_size  = Δs

################
# PathIterator2 #
################
struct PathIterator2{Tracker <: CoreTracker2}
    tracker::Tracker
    t_real::Bool
end
Base.IteratorSize(::Type{<:PathIterator2}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:PathIterator2}) = Base.HasEltype()

"""
iterator(tracker::CT, x₁, t₁=1.0, t₀=0.0; affine=true)

Prepare a tracker to make it usable as a (stateful) iterator. Use this if you want to inspect a specific
path. In each iteration the tuple `(x,t)` is returned.
If `affine == true` then `x` is the affine solution (internally we compute in projective space).

## Example

Assume you have `CoreTracker2` `tracker` and you wan to track `x₁` from 1.0 to 0.25:
```julia
for (x,t) in iterator(tracker, x₁, 1.0, 0.25)
println("x at t=\$t:")
println(x)
end
```

Note that this is a stateful iterator. You can still introspect the state of the tracker.
For example to check whether the tracker was successfull
(and did not terminate early due to some problem) you can do
```julia
println("Success: ", status(tracker) == CT_SUCCESS)
```
"""
function iterator(tracker::CT, x₁, t₁ = 1.0, t₀ = 0.0; kwargs...)
    init!(tracker, x₁, t₁, t₀; kwargs...)
    PathIterator2(tracker, typeof(t₁ - t₀) <: Real)
end

function current_x_t(iter::PathIterator2)
    x = current_x(iter.tracker)
    t = current_t(iter.tracker)
    (copy(x), iter.t_real ? real(t) : t)
end

function Base.iterate(iter::PathIterator2, state = nothing)
    state === nothing && return current_x_t(iter), 1
    iter.tracker.state.status != CT_TRACKING && return nothing

    step_done = false
    while !step_done && (iter.tracker.state.status == CT_TRACKING)

        step!(iter.tracker)
        check_terminated!(iter.tracker)

        if iter.tracker.state.status == CT_SUCCESS
            refine!(iter.tracker)
        end

        step_done = !iter.tracker.state.last_step_failed
    end
    current_x_t(iter), state + 1
end


#############################
# Convencience constructors #
#############################
"""
coretracker2_startsolutions(args...; kwargs...)

Construct a [`CoreTracker2`](@ref) and `startsolutions` in the same way `solve`
does it. This also takes the same input arguments as `solve`. This is convenient if you want
to investigate single paths.
"""
function coretracker2_startsolutions(args...; kwargs...)
    supported, rest = splitkwargs(kwargs, problem_startsolutions_supported_keywords)
    prob, startsolutions = problem_startsolutions(args...; supported...)
    tracker = CoreTracker2(prob, start_solution_sample(startsolutions); rest...)
    (tracker = tracker, startsolutions = startsolutions)
end

"""
coretracker2(args...; kwargs...)

Construct a [`CoreTracker2`](@ref) in the same way `solve`
does it. This also takes the same input arguments as `solve` with the exception that you do not need to specify startsolutions.
This is convenient if you want to investigate single paths.

## Examples

### Obtain single solution
We want to construct a path tracker to track a parameterized system `f` with parameters `p`
from the parameters `a` to `b`.
```julia
tracker = coretracker2(f, parameters=p, p₁=a, p₀=b)
```
You then can obtain a single solution at `b` by using
```julia
x_b = track(tracker, x_a).x
```

### Trace a path
To trace a path you can use the [`iterator`](@ref) method.

```julia
tracker = coretracker2(f, parameters=p, p₁=a, p₀=b, max_step_size =0.01)
for (x, t) in iterator(tracker, x₁)
@show (x,t)
end
```

If we want to guarantee smooth traces we can limit the maximal step size.
```julia
tracker = coretracker2(f, parameters=p, p₁=a, p₀=b, max_step_size =0.01)
for (x, t) in iterator(tracker, x₁)
@show (x,t)
end
```
"""
function coretracker2(args...; kwargs...)
    tracker, _ = coretracker2_startsolutions(args...; kwargs...)
    tracker
end
