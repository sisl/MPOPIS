using ReinforcementLearning
using IntervalSets
using Random

include("./CarRacingTracks/CarRacingTracks.jl")

struct CarRacingEnvParams{T}
    m::T
    Izz::T
    h_cm::T
    l_f::T
    l_r::T
    C_D0::T
    C_D1::T
    C_αf::T
    C_αr::T
    μ_f::T
    μ_r::T
    δ_max::T
    δ_dot_max::T
    Fx_max::T
    Fx_min::T
    λ_brake::T
    λ_drive::T
    β_limit::T
end

Base.show(io::IO, params::CarRacingEnvParams) = print(
    io,
    join(["$p=$(getfield(params, p))" for p in fieldnames(CarRacingEnvParams)], ","),
)

mutable struct CarRacingEnv{A,T,R<:AbstractRNG} <: AbstractEnv
    params::CarRacingEnvParams{T}
    action_space::A
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    done::Bool
    t::Int
    dt::T
    δt::T
    track::Track
    rng::R
end

"""
    CarRacingEnv(;kwargs...)

# Keyword arguments
- `T = Float64`
- `m = 2000.0, 			                        # Mass (kg)`
- `I_zz = 3764.0, 		                    	# Moment of inertia in vertical direction (kg m^2)`
- `h_cm = 0.3,	 			                    # Height of CoM above ground (m)`
- `l_f = 1.53, 			                        # Distance from CoM to front axle (m) ["a" in paper]`
- `l_r = 1.23, 			                        # Distance from CoM to rear axle (m) ["b" in paper]`
- `C_D0 = 241, 		    	                    # Constant drag term (N)`
- `C_D1 = 25.1, 	                        	# Linear drag coefficient (N s/m)`
- `C_αf = 150000.0,     	                    # Front cornering stiffness (N/rad)`
- `C_αr = 280000.0,                         	# Rear cornering stiffness (N/rad)`
- `μ_f = 0.9, 			                        # Front tire friction`
- `μ_r = 0.9, 				                    # Rear tire friction`
- `δ_max = deg2rad(18),	                        # Steering angle limits (rad)`
- `δ_dot_max = deg2rad(90),                    	# Stearing angle rate limits (rad/s)`
- `Fxmax = 7200,                                # maximum force due to motor torque (N)`
- `Fxmin = 22500                                # maximum force due to breaking`
- `λ_brake = 0.6,                               # Fixed brake distribution`
- `λ_drive = 0.0,                               # Fixed drive distribution`
- `dt = 0.1`,                                   # Time step between actions
- `δt = 0.01,                                   # Time step used for integration`
- `β_limit = deg2rad(45)                        # Beta penalty limit`
- `track = "./envs/CarRacingTracks/curve.csv,   # Track to load`
- `rng = Random.GLOBAL_RNG`
"""
function CarRacingEnv(;
    T = Float64,
    m = 2000.0, 			
    I_zz = 3764.0, 			
    h_cm = 0.3,	 			
    l_f = 1.53, 			
    l_r = 1.23, 			
    C_D0 = 241, 			
    C_D1 = 25.1, 			
    C_αf = 150000.0, 		
    C_αr = 280000.0, 		
    μ_f = 0.9, 				
    μ_r = 0.9, 				
    δ_max = deg2rad(18),	
    δ_dot_max = deg2rad(90),	
    Fx_max = 7200.0,  
    Fx_min = 22500.0,         
    λ_brake = 0.6,          
    λ_drive = 0.0,         
    dt = 0.1,
    δt = 0.01,
    β_limit = deg2rad(45),
    track = "./envs/CarRacingTracks/curve.csv",
    track_sample_factor = 20,
    rng = Random.GLOBAL_RNG,
)

    params = CarRacingEnvParams{T}(
        m,
        I_zz,
        h_cm,
        l_f,
        l_r,
        C_D0,
        C_D1,
        C_αf,
        C_αr,
        μ_f,
        μ_r,
        δ_max,
        δ_dot_max,
        Fx_max,
        Fx_min,
        λ_brake,
        λ_drive,
        β_limit,
    )
    
    return CarRacingEnv(params, T=T, dt=dt, δt=δt, track=track, track_sample_factor=track_sample_factor, rng=rng)
end

"""
    CarRacingEnv(params::CarRacingEnvParams;kwargs...)

# Keyword arguments
- `T = Float64`
- `dt = 0.1`,                                   # Time step between actions
- `δt = 0.01,                                   # Time step used for integration`
- `track = "./envs/CarRacingTracks/curve.csv,   # Track to load`
- `rng = Random.GLOBAL_RNG`
"""
function CarRacingEnv(params::CarRacingEnvParams;
    T = Float64,
    dt = 0.1,
    δt = 0.01,
    track = "./envs/CarRacingTracks/curve.csv",
    track_sample_factor = 10,
    rng = Random.GLOBAL_RNG,
)

    action_space = ClosedInterval{Vector{T}}(
        [-1.0, -1.0], 
        [ 1.0,  1.0],
        )
    observation_space = Space([
        -Inf..Inf,                          # X position in XY plane (x = north, y = west)
        -Inf..Inf,                          # Y position in XY plane (x = north, y = west)
        -π..π,                              # yaw (rotation from x axis toward y axis [north to west])
        -Inf..Inf,                          # Longitudinal velocity
        -Inf..Inf,                          # Lateral velocity
        -Inf..Inf,                          # yaw rate
        -params.δ_max..params.δ_max,        # steering angle
        -1.0..1.0,                          # acceleration/brake amount [-1, 1]
        ])
        
    env = CarRacingEnv(
        params, 
        action_space, 
        observation_space, 
        zeros(T,8),
        false, 
        0,
        dt,
        δt,
        Track(track, sample_factor=track_sample_factor),
        rng,
        )

    reset!(env)
    env
end

CarRacingEnv{T}(; kwargs...) where {T} = CarRacingEnv(; T = T, kwargs...)

Random.seed!(env::CarRacingEnv, seed) = Random.seed!(env.rng, seed)

RLBase.action_space(env::CarRacingEnv) = env.action_space
RLBase.state_space(env::CarRacingEnv{T}) where {T} = env.observation_space
RLBase.is_terminated(env::CarRacingEnv) = env.done
RLBase.state(env::CarRacingEnv) = env.state

function within_track(env::CarRacingEnv)
    return within_track(env.track, env.state[1:2])
end

"""
    reward(env::CarRacingEnv)

    - Reward for going fast and staying within the track boundaries.
    - Boundary violation: -10000
    - Excessive β       :  -5000
    - Speed             : +2.0 * (Vx² + Vy²)^(1/2)
"""
function RLBase.reward(env::CarRacingEnv{T}) where {T} 
    reward = 0.0
    if !within_track(env)
        reward += -10000.0
    end
    β = atan(env.state[5], env.state[4])
    if abs(β) > env.params.β_limit
        reward += -5000.0
    end
    reward += 2.0*norm(env.state[4:5])
    return reward
end

function RLBase.reset!(env::CarRacingEnv{A,T}) where {A,T}
    ss_size = length(env.state)
    env.state = zeros(T, ss_size)
    env.t = 0
    env.done = false
    nothing
end

function RLBase.reset!(env::CarRacingEnv{A,T}, state::Vector{T}) where {A,T}
    env.state = state
    env.t = 0
    env.done = false
    nothing
end

"""
    a::Vector{Float64}
    size = (2,)
    a[1] = Turn angle [-max turn angle, max turn angle] (-1 right turn, +1 left turn)
    a[2] = Pedal amount (-1 = full brake, 1 = full throttle)
"""
function (env::CarRacingEnv{<:ClosedInterval})(a::Vector{Float64})
    a in env.action_space || error("Action is not in action space")
    _step!(env, a)
end

function (env::CarRacingEnv{<:ClosedInterval})(a::Vector{Int})
    env(Float64.(a))
end

function (env::CarRacingEnv{<:ClosedInterval})(a::Matrix{Float64})
    size(a)[2] == 1 || error("Only implented for one step")
    env(vec(a))
end

function calc_tire_fy(α, μ, C_α, fzt, fxt)
    fy_max = sqrt(max((μ * fzt)^2 - fxt^2, 1e-8))
    ta = tan(α)
    if abs(α) < atan(3 * fy_max / C_α)
        return -C_α*ta + (C_α^2/(3*fy_max))*abs(ta)*ta - ((C_α^3)/(27*fy_max^2))*ta^3
    else
        return -fy_max * sign(α)
    end
end

function calc_tire_fz(params::CarRacingEnvParams, fx, tire::Char)
    mass = params.m
    l_t = params.l_f
    h_cm = params.h_cm
    L = params.l_r + params.l_f
    if tire == 'f'
        l_t = params.l_r
        h_cm *= -1
    end
    return (mass * l_t * 9.81 + h_cm * fx) / L
end

"""
Planar single-track model with tire forces coming from a 
single-friction-coefficient brush tire model.

M. Brown and J. C. Gerdes, Coordinating Tire Forces to Avoid Obstacles Using 
Nonlinear Model Predictive Control, in IEEE Transactions on Intelligent 
Vehicles, vol. 5, no. 1, pp. 21-31, March 2020, doi: 10.1109/TIV.2019.2955362.
"""
function _step!(env::CarRacingEnv, a::Vector{Float64})   
    env.t += 1
    x = env.state[1]
    y = env.state[2]
    Ψ = env.state[3]
    Vx = env.state[4]
    Vy = env.state[5]
    Ψ_dot = env.state[6]
    δ = env.state[7]

    dt = env.dt
    δt = env.δt
    
    commanded_Δδ_rate = abs(a[1]*env.params.δ_max - δ) / dt # Commanded turn rate
    Δδ_rate = min(commanded_Δδ_rate, env.params.δ_dot_max) * sign(a[1]*env.params.δ_max - δ)
    pedal = a[2] # Assume instant torque for accel or brake

    integration_steps = round(Int, dt/δt)
    for _ in 1:integration_steps
        δ += Δδ_rate * δt
        
        # Slip angles for the tires
        # α_f = -δ
        # α_r = 0.0 
        # if Vx != 0
        α_f = atan((Vy + env.params.l_f*Ψ_dot) , Vx) - δ
        α_r = atan((Vy - env.params.l_r*Ψ_dot) , Vx)
        # end
        
        # Simple drag
        fx_aero = (env.params.C_D0 + env.params.C_D1*abs(Vx)) * sign(Vx)
        
        accel = env.params.Fx_max * max(pedal, 0.0)
        brake = env.params.Fx_min * min(pedal, 0.0) * sign(Vx) # Opposite direction of long velocity
        fx = accel + brake

        # Distribution of the forces on the front and rear tires
        fxf = (pedal <= 0 ? env.params.λ_brake : env.params.λ_drive) * fx
        fxr = (1 - (pedal <= 0 ? env.params.λ_brake : env.params.λ_drive)) * fx
        fzf = calc_tire_fz(env.params, fx, 'f')
        fzr = calc_tire_fz(env.params, fx, 'r')
        fyf = calc_tire_fy(α_f, env.params.μ_f, env.params.C_αf, fzf, fxf)
        fyr = calc_tire_fy(α_r, env.params.μ_r, env.params.C_αr, fzr, fxr)
        
        Ψ_ddot = (1/env.params.Izz) * (env.params.l_f * (fxf*sin(δ) + fyf*cos(δ))  - env.params.l_r*fyr)
        Vy_dot = (1/env.params.m) * (fyf*cos(δ) + fxf*sin(δ) + fyr) - Ψ_dot*Vx
        Vx_dot = (1/env.params.m) * (fxf*cos(δ) - fyf*sin(δ) + fxr - fx_aero) + Ψ_dot*Vy

        Ψ_dot += Ψ_ddot * δt                # Updated yaw rate
        Vx += Vx_dot * δt                   # Updated longitudinal velocity
        Vy += Vy_dot * δt                   # Updated lateral velocity
        Ψ += Ψ_dot * δt                     # Updated yaw (heading)
        Ψ = atan(sin(Ψ), cos(Ψ))
        x += (Vx*cos(Ψ) - Vy*sin(Ψ)) * δt   # Updated x position yaw heading is counterclockwise, x is north, y is west
        y += (Vx*sin(Ψ) + Vy*cos(Ψ)) * δt   # Updated y position yaw is counterclockwise, x is north, y is west
    end

    env.state[1] = x
    env.state[2] = y
    env.state[3] = Ψ
    env.state[4] = Vx
    env.state[5] = Vy
    env.state[6] = Ψ_dot
    env.state[7] = δ
    env.state[8] = pedal
    return env
end

