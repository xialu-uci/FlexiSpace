# first let's simulate data with no noise
using FlexiBasicLearning
using JLD2

function sim_data(num_points, dofs; std = 0.05, shape = crooked_flexi, save_name = nothing)

    # helper true flexi params
    true_params = shape(dofs)
    # num_points must be in
    x = LinRange(0.0,1.0, num_points)
    y = FlexiFunctions.evaluate_decompress.(x, Ref(true_params))
    # add noise to y
    noise = randn(num_points).*std # this makes it possible for vals outside [0,1]. Should I clamp?
    y_noisy = y .+ noise
    data = [x y_noisy]
    if !isnothing(save_name)
        savedir = "../FlexiSpaceLocal/data"
        @save joinpath(savedir, save_name) data
        println("Saved $save_name to $savedir")
    end
    return data
    
end
# using FlexiBasicLearning

# params = FlexiBasicLearning.FlexiFunctions.generate_flexi_ig(5)
# println(params)

function crooked_flexi(dofs)
    params = zeros(dofs)
    # make all odd indices 1
    params[1:2:end] .= 1.0
    # make it unit
    return params / LinearAlgebra.norm(params)
end

function cu_flexi(dofs)
    params = collect(1:dofs)
    return params / LinearAlgebra.norm(params)
end

function cd_flexi(dofs)
    params = collect(dofs:-1:1)   # explicit descending step
    return params / LinearAlgebra.norm(params)
end


# test: passed
# small_data = sim_data(10, 5)

NUM_POINTS = 100
DOFS = 5

# crooked
# sim_data(NUM_POINTS, DOFS; save_name = "sim_data_crooked_5seg.jld2")

# cu 
sim_data(NUM_POINTS, DOFS; shape = cu_flexi, save_name = "sim_data_cu_5seg.jld2")

# cd 
sim_data(NUM_POINTS, DOFS; shape = cd_flexi, save_name = "sim_data_cd_5seg.jld2")