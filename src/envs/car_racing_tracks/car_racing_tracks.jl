
struct Track
    x::Vector{Float64}
    y::Vector{Float64}
    lane_width::Vector{Float64}  
    x′::Vector{Float64}
    y′::Vector{Float64}
    lane_width′::Vector{Float64}
    boundary_left::Matrix{Float64}
    boundary_right::Matrix{Float64}
    sample_factor::Int
end

function Track(infile::String, width::Vector{Float64}; sample_factor=20)
    csv_data = CSV.File(infile, header=false)
    csv_data.cols == 2 || error("Can only have 2 columns for a track file")
    length(csv_data.Column1) == length(width) || error("Supplied width vector does not match length of track file")
    x = csv_data.Column1
    y = csv_data.Column2
    lane_width = width
    x′ = x[1:sample_factor:end]
    y′ = y[1:sample_factor:end]
    lane_width′ = lane_width[1:sample_factor:end]

    l_boundary, r_boundary = calculate_boundary(x, y, lane_width)

    return Track(x, y, lane_width, x′, y′, lane_width′, l_boundary, r_boundary, sample_factor)
end

function Track(infile::String; width::Float64=15.0, sample_factor=20)
    csv_data = CSV.File(infile, header=false)
    lane_width = ones(Float64, length(csv_data)) * width
    return Track(infile, lane_width, sample_factor=sample_factor)
end

function calculate_boundary(x::Vector{Float64}, y::Vector{Float64}, w::Vector{Float64})

    r_bound = zeros(Float64, length(x), 2)
    l_bound = zeros(Float64, length(x), 2)
    
    Δx = x[2] - x[1]
    Δy = y[2] - y[1]
    p = [-Δy, Δx] ./ norm([-Δy, Δx])
    l_bound[1,:] = [x[1], y[1]] + w[1]*p 
    r_bound[1,:] = [x[1], y[1]] - w[1]*p
    
    for ii in 2:(length(x)-1)
        Δx = x[ii+1] - x[ii-1]
        Δy = y[ii+1] - y[ii-1]
        p = [-Δy, Δx] ./ norm([-Δy, Δx])
        l_bound[ii,:] = [x[ii], y[ii]] + w[ii]*p 
        r_bound[ii,:] = [x[ii], y[ii]] - w[ii]*p
    end

    Δx = x[end] - x[end-1]
    Δy = y[end] - y[end-1]
    p = [-Δy, Δx] ./ norm([-Δy, Δx])
    l_bound[end,:] = [x[end], y[end]] + w[end]*p 
    r_bound[end,:] = [x[end], y[end]] - w[end]*p

    return l_bound, r_bound
end

function within_track(track::Track, pos::Vector{Int})
    return within_track(track::Track, Float64.(pos))
end

function within_track(track::Track, pos::Vector{Float64})
    length(pos) == 2 || error("Position is only 2D")

    dists = (track.x′ .- pos[1]).^2 .+ (track.y′ .- pos[2]).^2

    _ , min_idx  = findmin(dists)
    min_idx = Int(min_idx)
    min_idx_m1 = mod1(min_idx - 1, length(track.x′))
    min_idx_p1 = mod1(min_idx + 1, length(track.x′))
    dist_m1 = norm([track.x′[min_idx_m1], track.y′[min_idx_m1]] - pos)
    dist_p1 = norm([track.x′[min_idx_p1], track.y′[min_idx_p1]] - pos)
    min_idx_2 = dist_m1 <= dist_p1 ? min_idx_m1 : min_idx_p1

    # Projecting the point (p3) onto the line defined by the closest two points (p1, p2)
    # The line is parametrized as p1 + t(p2 - p1)
    # The projection falls where t = [(p3-p1) ⋅ (p2-p1)] / |p2-p1|^2
    p1 = [track.x′[min_idx], track.y′[min_idx]]
    p2 = [track.x′[min_idx_2], track.y′[min_idx_2]]
    p3 = pos
    t = ((p3-p1) ⋅ (p2-p1)) /  ((p2-p1) ⋅ (p2-p1))
    projected_point = p1 + t * (p2 - p1)
    dist_to_pt = norm(projected_point - pos)
    return (within=(dist_to_pt < track.lane_width′[min_idx]), dist=dist_to_pt)

end