
using LinearAlgebra
using IntervalSets

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

function compute_weights(costs::Vector{Float64}, λ::Float64)
    ρ = minimum(costs)
    normalized_costs = -1/λ *(costs .- ρ)
    weights = exp.(normalized_costs)
    η = sum(weights)
    return weights ./ η
end