
module MPOPIS

using CSV
using LinearAlgebra
using IntervalSets
using Distributions
using StatsBase
using Random
using CovarianceEstimation
import CovarianceEstimation.LinearShrinkageTarget
import CovarianceEstimation.SimpleCovariance
using ReinforcementLearning
import ReinforcementLearning.AbstractEnv
using Plots
import Plots.plot

using Reexport

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
    exceed_β,
    block_diagm
    # Random.seed!,
    # RLBase.reward,
    # RLBase.action_space,
    # RLBase.state_space,
    # RLBase.is_terminated,
    # RLBase.state,
    # reset!,
    # _update_states_envs2env,
    # _update_states_env2envs

@reexport ReinforcementLearning.RLBase
@reexport Plots
@reexport Random

# RLBase.action_space(env::CarRacingEnv) = env.action_space
# RLBase.state_space(env::CarRacingEnv{T}) where {T} = env.observation_space
# RLBase.is_terminated(env::CarRacingEnv) = env.done
# RLBase.state(env::CarRacingEnv) = env.state

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