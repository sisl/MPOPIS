
using Plots
import Plots.plot

include("../agents/mppi.jl")
include("./CarRacingTracks/CarRacingTracks.jl")
include("CarRacing.jl")

function plot(track::Track)

    lx = track.boundary_left[:,1]
    ly = track.boundary_left[:,2]
    rx = track.boundary_right[:,1]
    ry = track.boundary_right[:,2]
    
    lt = [Tuple(track.boundary_left[ii,:]) for ii in 1:length(track.boundary_left[:,1])]
    rt = [Tuple(track.boundary_right[ii,:]) for ii in length(track.boundary_right[:,1]):-1:1]
    tuple_list = [lt; rt]

    trk_shape = Plots.Shape(tuple_list)
    
    plt = plot(trk_shape, 
            fillcolor=plot_color(:grey, 1.0), 
            lw=0,
            legend=:none, 
            grid=false,
            background_color=palette(:algae)[50],
            size=(1500, 1500),
            xlim=(minimum([lx; rx])-10, maximum([lx; rx])+10),
            ylim=(minimum([ly; ry])-10, maximum([ly; ry])+10),
    )
    plot!(plt, lx, ly, color=:black, lw=3, legend=:none)
    plot!(plt, rx, ry, color=:black, lw=3, legend=:none)
    plot!(plt, aspect_ratio=:equal)
    return plt

end

function plot(env::CarRacingEnv; car_only::Bool=false, car_color::Int=1)
    if mod(car_color, 6) == 0 car_color += 1 end
    x, y, ψ = env.state[1:3]
    l_f = env.params.l_f
    l_r = env.params.l_r
    w = (l_f + l_r)*0.4

    rot_mat = [cos(ψ) -sin(ψ) ;
               sin(ψ)  cos(ψ)
              ]

    fl = rot_mat * [l_f; w/2] + [x; y]
    fr = rot_mat * [l_f; -w/2] + [x; y]
    rl = rot_mat * [-l_r; w/2] + [x; y]
    rr = rot_mat * [-l_r; -w/2] + [x; y]
    
    ar = rot_mat * [-l_r*0.8; 0] + [x; y]
    r = (l_f+l_r) * 0.8
    u, v = r * cos(ψ), r * sin(ψ)

    car_xy = vcat(fl', fr', rr', rl', fl')

    if !car_only 
        plt = plot(env.track) 
    end
    
    plot!(car_xy[:,1], car_xy[:,2], 
        linestype = :path, 
        linewidth = 2, 
        linecolor = palette(:lighttest)[mod1(car_color, 7)], 
        legend = false
    )
    arrow0!(ar[1], ar[2] , u, v, as=0.35, lc=:black)
end

function plot(env::CarRacingEnv, pol::AbstractPathIntegralPolicy, perc=1.0)
    K = pol.params.num_samples
    trajs = pol.logger.trajectories
    traj_weights = pol.logger.traj_weights
    
    mod_fac = 10 / perc
    mod_fac = round(Int, mod_fac)

    order = sortperm(traj_weights, rev=true)
    
    p = plot(env)

    for (ii, k) ∈ enumerate(order)
        col_idx = (K - ii + 1) / K
        if mod(ii*10, mod_fac) == 0 && ii != 1
            x = trajs[k][:,1]
            y = trajs[k][:,2]
            p = plot!(x,y, 
                linewidth=0.05,
                linecolor=cgrad(:RdYlGn_9)[col_idx]
            )
        end
    end
    return p
end

function plot(env::MultiCarRacingEnv)
    p = plot(env.envs[1], car_color=1)
    for ii in 2:env.N
        p = plot(env.envs[ii], car_only=true, car_color=ii)
    end
    return p
end


function plot(env::MultiCarRacingEnv, pol::AbstractPathIntegralPolicy, perc=1.0)
    K = pol.params.num_samples
    trajs = pol.logger.trajectories
    traj_weights = pol.logger.traj_weights
    
    mod_fac = 10 / perc
    mod_fac = round(Int, mod_fac)

    order = sortperm(traj_weights, rev=true)
    
    p = plot(env)

    for (ii, k) ∈ enumerate(order)
        col_idx = (K - ii + 1) / K
        if mod(ii*10, mod_fac) == 0 && ii != 1
            for jj ∈ 1:env.N
                idx = round(Int, (jj-1)*pol.params.ss/env.N)
                x = trajs[k][:,1+idx]
                y = trajs[k][:,2+idx]
                p = plot!(x,y, 
                    linewidth=0.05,
                    linecolor=cgrad(:RdYlGn_9)[col_idx]
                )
            end
        end
    end
    return p
end


function plot(env::DroneEnv; drone_only=false)
    x, y, z = env.state[1:3]
    vx, vy, vz = env.state[4:6]
    L = 2*env.params.L

    ψ = atan(vy,vx)
    rot_mat = [cos(ψ) -sin(ψ) ;
               sin(ψ)  cos(ψ)
              ]

    fl = rot_mat * [L; L] + [x; y]
    fr = rot_mat * [L; -L] + [x; y]
    rl = rot_mat * [-L; L] + [x; y]
    rr = rot_mat * [-L; -L] + [x; y]
    
    ar = rot_mat * [-L*0.8; 0] + [x; y]
    r = 2*L*0.8
    u, v = r * cos(ψ), r * sin(ψ)

    car_xy = vcat(fl', fr', rr', rl', fl')

    if !drone_only 
        plt = plot(env.track) 
    end
    
    plot!(car_xy[:,1], car_xy[:,2], linestype=:path, linewidth=2, linecolor=:red, legend=false)
    arrow0!(ar[1], ar[2] , u, v, as=0.35, lc=:black)
    return plt
end


# as: arrow head size 0-1 (fraction of arrow length)
# la: arrow alpha transparency 0-1
function arrow0!(x, y, u, v; as=0.07, lc=:black, la=1)
    nuv = sqrt(u^2 + v^2)
    v1, v2 = [u;v] / nuv,  [-v;u] / nuv
    v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
    v5 = v4 - 2*(v4'*v2)*v2
    v4, v5 = as*nuv*v4, as*nuv*v5
    plot!([x,x+u], [y,y+v], lc=lc,la=la)
    plot!([x+u,x+u-v5[1]], [y+v,y+v-v5[2]], lc=lc, la=la)
    plot!([x+u,x+u-v4[1]], [y+v,y+v-v4[2]], lc=lc, la=la)
end