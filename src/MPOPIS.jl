
module MPOPIS

using Reexport
using CSV
using LinearAlgebra
using IntervalSets
using Distributions
using StatsBase
using Random
@reexport import Random.seed!
using CovarianceEstimation
import CovarianceEstimation.LinearShrinkageTarget
import CovarianceEstimation.SimpleCovariance
@reexport using ReinforcementLearning
import ReinforcementLearning.AbstractEnv
import ReinforcementLearning.RLBase: action_space, state_space
using Plots
@reexport import Plots.plot

export
    MPPI_Policy,
    GMPPI_Policy,
    CEMPPI_Policy,
    CMAMPPI_Policy,
    μAISMPPI_Policy,
    μΣAISMPPI_Policy,
    PMCMPPI_Policy,
    NESMPPI_Policy,
    Track,
    CarRacingEnv,
    MultiCarRacingEnv,
    DroneEnv,
    within_track,
    calculate_β,
    exceed_β,
    block_diagm,
    _update_states_envs2env,
    _update_states_env2envs

abstract type AbstractWeightMethod end
abstract type AbstractPathIntegralPolicy end
Random.seed!(pol::AbstractPathIntegralPolicy, seed) = Random.seed!(pol.rng, seed)

abstract type AbstractGMPPI_Policy <: AbstractPathIntegralPolicy end

struct Cross_Entropy <: AbstractWeightMethod
    elite_threshold::Float64
    num_elite_samples::Int
end

struct Information_Theoretic <: AbstractWeightMethod
    λ::Float64
end

include("envs/car_racing_tracks/car_racing_tracks.jl")
include("envs/car_racing.jl")
include("envs/multi-car_racing.jl")
include("envs/drone_env.jl")
include("envs/mppi_RLBase_mods.jl")
include("utils.jl")
include("mppi_mpopi_policies.jl")
include("plots.jl")

end # module