

using ReinforcementLearning
using LinearAlgebra
using IntervalSets
import ReinforcementLearning.AbstractEnv

abstract type AbstractWeightMethod end
abstract type AbstractPathIntegralPolicy end
abstract type AbstractGMPPI_Policy <: AbstractPathIntegralPolicy end

struct Cross_Entropy <: AbstractWeightMethod
    elite_threshold::Float64
    num_elite_samples::Int
end

struct Information_Theoretic <: AbstractWeightMethod
    λ::Float64
end

function action_space_size(act_space::ClosedInterval)
    return length(leftendpoint(act_space))
end
function action_space_size(act_space::Base.OneTo)
    return 1
end

function block_diagm(A::Vector{Float64}, rep_number::Int)
    return diagm(repeat(A, rep_number))
end

function block_diagm(A::Matrix{Float64}, rep_number::Int)
    r = size(A)[1]
    rm1 = r -1
    B = zeros(Float64, r*rep_number, r*rep_number)
    for ii in 1:r:(r*rep_number)
        B[ii:ii+rm1, ii:ii+rm1] = A
    end
    return B
end

function get_model_controls(action_space::ClosedInterval, V::Vector{Float64})
    return get_model_controls(action_space, V, 1)
end
function get_model_controls(action_space::Base.OneTo, V::Vector{Float64})
    return get_model_controls(action_space, V, 1)
end
function get_model_controls(action_space::ClosedInterval, V::Vector{Float64}, horizon::Int)
    as = action_space_size(action_space)
    min_controls = leftendpoint(action_space)
    max_controls = rightendpoint(action_space)
    control_mat = reshape(V, as, horizon)
    for rᵢ ∈ 1:size(control_mat)[1]
        control_mat[rᵢ,:] = clamp.(control_mat[rᵢ,:], min_controls[rᵢ], max_controls[rᵢ])
    end
    if as == 1
        control_mat = vec(control_mat)
    end
    return control_mat
end
function get_model_controls(action_space::Base.OneTo, V::Vector{Float64}, horizon::Int)
    as = 1
    min_controls = minimum(action_space)
    max_controls = maximum(action_space)
    
    control_mat = reshape(V, as, horizon)
    control_mat[1,:] = clamp.(control_mat[1,:], min_controls, max_controls)
    control_mat = round.(Int, vec(control_mat))
    return control_mat
end

function all_perm(xs, n) 
    return vec(map(collect, Iterators.product(ntuple(_ -> xs, n)...)))
end

function get_min_max_control_set(action_space::ClosedInterval, horizon::Int)
    as = action_space_size(action_space)
    min_controls = leftendpoint(action_space)
    max_controls = rightendpoint(action_space)
    min_max_mat = hcat(min_controls, max_controls)
    U = Vector{Matrix{Float64}}(undef, 2^as + 1)
    min_max_controls = all_perm([1, 2], as)
    control_vec = zeros(Float64, as)
    for ii ∈ 1:length(min_max_controls)
        for aᵢ ∈ 1:as
            control_vec[aᵢ] = min_max_mat[aᵢ, min_max_controls[ii][aᵢ]]
        end
        U[ii] = repeat(control_vec, 1, horizon)
    end
    U[end] = repeat(mean(min_max_mat, dims=2), 1, horizon)
    return U
end

function compute_weights(weight_method::Information_Theoretic, costs::Vector{Float64})
    λ = weight_method.λ
    ρ = minimum(costs)
    normalized_costs = -1/λ *(costs .- ρ)
    weights = exp.(normalized_costs)
    η = sum(weights)
    return weights ./ η
end

function (env::CartPoleEnv{<:Base.OneTo{Int}})(a::Vector)
    length(a) == 1 || error("Only implented for 1 step")
    env(a[1])
end
function (env::CartPoleEnv{<:ClosedInterval})(a::Vector)
    length(a) == 1 || error("Only implented for 1 step")
    env(a[1])
end
function (env::MountainCarEnv{<:ClosedInterval})(a::Vector)
    length(a) == 1 || error("Only implented for 1 step")
    env(a[1])
end
function (env::MountainCarEnv{<:Base.OneTo{Int}})(a::Vector)
    length(a) == 1 || error("Only implented for 1 step")
    env(a[1])
end