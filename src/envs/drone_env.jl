

struct DroneEnvParams{T}
    m::T
    L::T
    Iₘ::Matrix{T}
    Iₘ_inv::Matrix{T}
    k_m::T
    k_F::T
    k_M::T
    ω_min::T
    ω_max::T
end

Base.show(io::IO, params::DroneEnvParams) = print(
    io,
    join(["$p=$(getfield(params, p))" for p in fieldnames(DroneEnvParams)], ","),
)

mutable struct DroneEnv{A,T,R<:AbstractRNG} <: AbstractEnv
    params::DroneEnvParams{T}
    action_space::A
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    done::Bool
    t::Int
    dt::T
    δt::T
    track::Track # Need to adjust this to a 3D track with the ability to add obstacles
    rng::R
end

"""
    DroneEnv(;kwargs...)

# Keyword arguments
- `T = Float64`
- `m = 0.25, 			                        # Mass (kg)`
- `L = 0.1,                                     # Length of the moment arm from the body to a motor`
- `I_m = diagm([2.5e-4, 2.5e-4, 1.2e-3]),     	# Moment of inertia in body frame (kg m^2)`
- `k_m = 20,	 			                    # Motor constant (1/s)`
- `k_F = 6.11e-8,   	                        # Motor force constant (N/rpm²)`
- `k_M = 1.5e-9,                                # Motor moment constant (N/rpm²)`
- `ω_min = 1200.0,                              # Motor min rotation speed (rpm)`
- `ω_max = 7800.0,                              # Motor max rotation speed (rpm)`
- `dt = 0.1`,                                   # Time step between actions
- `δt = 0.01,                                   # Time step used for integration`
- `track = string(@__DIR__, "/car_racing_tracks/curve.csv"),   # Track to load`
- `lane_width = 5.0,                            # Track lane_width`
- `rng = Random.GLOBAL_RNG`
"""
function DroneEnv(;
    T = Float64,
    m = 0.25, 		
    L = 0.1,	
    I_m = diagm([2.5e-4, 2.5e-4, 1.2e-3]), 			
    k_m = 20,
    k_F = 6.11e-8,
    k_M = 1.5e-9,
    ω_min = 1200.0,
    ω_max = 7800.0,
    dt = 0.1,
    δt = 0.01,
    track = string(@__DIR__, "/car_racing_tracks/curve.csv"),
    lane_width = 5.0,
    rng = Random.GLOBAL_RNG,
)

    params = DroneEnvParams{T}(
        m,
        L,
        I_m,
        I_m^(-1),
        k_m,
        k_F,
        k_M,
        ω_min,
        ω_max,
    )
    
    return DroneEnv(params, T=T, dt=dt, δt=δt, track=track, rng=rng)
end

"""
    DroneEnv(params::DroneEnvParams;kwargs...)

# Keyword arguments
- `T = Float64`
- `dt = 0.1`,                                   # Time step between actions
- `δt = 0.01,                                   # Time step used for integration`
- `track = string(@__DIR__, "/car_racing_tracks/curve.csv"),   # Track to load`
- `lane_width = 5.0,                            # Track lane_width`
- `rng = Random.GLOBAL_RNG`
"""
function DroneEnv(params::DroneEnvParams;
    T = Float64,
    dt = 0.1,
    δt = 0.01,
    track = string(@__DIR__, "/car_racing_tracks/curve.csv"),
    lane_width = 5.0,
    rng = Random.GLOBAL_RNG,
)

    action_space = ClosedInterval{Vector{T}}(
        [-1.0, -1.0, -1.0, -1.0], 
        [ 1.0,  1.0,  1.0,  1.0],
        )

    observation_space = Space([
        -Inf..Inf,                          # X position 
        -Inf..Inf,                          # Y position 
        -Inf..Inf,                          # Z position 
        -Inf..Inf,                          # Vx (translational velocity)
        -Inf..Inf,                          # Vy 
        -Inf..Inf,                          # Vz 
        -π..π,                              # Euler angle 1 (roll, about x, ϕ)
        -π/2..π/2,                          # Euler angle 2 (pitch, about y, θ)
        -π..π,                              # Euler angle 3 (yaw, about z, ψ)
        -Inf..Inf,                          # Body rate 1 (roll rate, p)
        -Inf..Inf,                          # Body rate 2 (pitch rate, q)
        -Inf..Inf,                          # Body rate 4 (yaw rate, r)
        0.0..Inf,                           # Motor 1 speed (ω₁)
        0.0..Inf,                           # Motor 2 speed (ω₂)
        0.0..Inf,                           # Motor 3 speed (ω₃)
        0.0..Inf,                           # Motor 4 speed (ω₄)
        ])
    state = zeros(T,length(observation_space))
    state[end-3:end] = ones(T, 4) * 1200.0

    env = DroneEnv(
        params, 
        action_space, 
        observation_space, 
        state,
        false, 
        0,
        dt,
        δt,
        Track(track, width=lane_width),
        rng,
        )

    reset!(env)
    env
end

DroneEnv{T}(; kwargs...) where {T} = DroneEnv(; T = T, kwargs...)

Random.seed!(env::DroneEnv, seed) = Random.seed!(env.rng, seed)

RLBase.action_space(env::DroneEnv) = env.action_space
RLBase.state_space(env::DroneEnv{T}) where {T} = env.observation_space
RLBase.is_terminated(env::DroneEnv) = env.done
RLBase.state(env::DroneEnv) = env.state

# function within_track(env::DroneEnv)
#     return within_track(env.track, env.state[1:2])
# end

"""
    reward(env::DroneEnv)

"""
function RLBase.reward(env::DroneEnv{T}) where {T} 
    if any(isnan.(env.state))
        return -100000.0
    end
    rew = 0.0
    
    # Go towards a point
    desired_pos = [10.0, 10.0, 10.0]
    dist_from_desired = norm(desired_pos - env.state[1:3])
    rew += -dist_from_desired

    # Penalize for more time spent away from the point
    if dist_from_desired >= 5
        rew += -1
    end

    return rew
end

function RLBase.reset!(env::DroneEnv{A,T}) where {A,T}
    state = zeros(T,length(env.state))
    state[end-3:end] = ones(T, 4) * 1500.0
    env.state = state
    env.t = 0
    env.done = false
    nothing
end

function RLBase.reset!(env::DroneEnv{A,T}, state::Vector{T}) where {A,T}
    env.state = state
    env.t = 0
    env.done = false
    nothing
end

"""
    a::Vector{Float64}
    size = (4,)
    a[1] = Desired motor speed for motor 1 (ω₁, rpm)
    a[2] = Desired motor speed for motor 2 (rpm)
    a[3] = Desired motor speed for motor 3 (rpm)
    a[4] = Desired motor speed for motor 4 (rpm)
"""
function (env::DroneEnv{<:ClosedInterval})(a::Vector{Float64})
    a in env.action_space || error("Action is not in action space")
    length(a) == 4 || error("Incorrect action size")
    _step!(env, a)
end

function (env::DroneEnv{<:ClosedInterval})(a::Vector{Int})
    env(Float64.(a))
end

function (env::DroneEnv{<:ClosedInterval})(a::Matrix{Float64})
    size(a)[2] == 1 || error("Only implented for one step")
    env(vec(a))
end

"""
Quadrotor dynamics NOT considering complications due to varying thrust based
on angle of attack, blade flapping, or airflow disruption.

N. Michael, D. Mellinger, Q. Lindsey, V. Kumar, The GRASP Multiple Micro UAV Testbed
"""
function _step!(env::DroneEnv, a::Vector{Float64})   
    env.t += 1
    r = env.state[1:3]     # Position
    r_dot = env.state[4:6]  # Velocity
    Φ = env.state[7:9]      # Euler angles
    ω = env.state[10:12]    # body rates
    Ω = env.state[13:16]    # motor speeds

    dt = env.dt
    δt = env.δt

    # Scaled input controls from -1 to 1. Tis adjusts
    #  them to the appropriate rate (-1 = min, +1 = max)
    Ωᵈ = env.params.ω_min .+ (a .+1 ) .* ((env.params.ω_max-env.params.ω_min)/2)

    integration_steps = round(Int, dt/δt)
    for _ in 1:integration_steps

        Ω_dot = env.params.k_m .* (Ωᵈ - Ω)
        Ω += Ω_dot .* δt # Updated motor speeds
        
        F = env.params.k_F .* Ω.^2
        M = env.params.k_M .* Ω.^2

        temp_vec = [
            env.params.L * (F[2] - F[4]),
            env.params.L * (F[3] - F[1]),
            M[1] - M[2] + M[3] - M[4],
        ]
        ω_dot = env.params.Iₘ_inv * (temp_vec - cross(ω, env.params.Iₘ*ω))

        ω += ω_dot .* δt # Updated body rates

        temp_mat = [
            cos(Φ[2])  0 -cos(Φ[1])*sin(Φ[2]);
                0      1      sin(Φ[1])      ;
            sin(Φ[2])  0  cos(Φ[1])*cos(Φ[2])
        ]
        Φ_dot = temp_mat^(-1) * ω
        Φ += Φ_dot .* δt # Updated Euler angles
        Φ = mod.(Φ, 2π)

        R_11 = cos(Φ[3])*cos(Φ[2]) - sin(Φ[1])*sin(Φ[3])*sin(Φ[2])
        R_12 = -cos(Φ[1])*sin(Φ[3])
        R_13 = cos(Φ[3])*sin(Φ[2]) + cos(Φ[2])*sin(Φ[1])*sin(Φ[3])

        R_21 = cos(Φ[2])*sin(Φ[3]) + cos(Φ[3])*sin(Φ[1])*sin(Φ[2])
        R_22 = cos(Φ[1])*cos(Φ[3])
        R_23 = sin(Φ[3])*sin(Φ[2]) - cos(Φ[3])*cos(Φ[2])*sin(Φ[1])

        R_31 = -cos(Φ[1])*sin(Φ[2])
        R_32 = sin(Φ[1])
        R_33 = cos(Φ[1])*cos(Φ[2])

        R = [
             R_11 R_12 R_13;
             R_21 R_22 R_23;
             R_31 R_32 R_33
        ]

        r_ddot = [0,0,-9.81] + R*[0,0,sum(F)/env.params.m]
        r_dot += r_ddot .* δt
        r += r_dot .* δt
    end

    env.state[1:3] = r
    env.state[4:6] = r_dot
    env.state[7:9] = Φ
    env.state[10:12] = ω
    env.state[13:16] = Ω
    return env
end
