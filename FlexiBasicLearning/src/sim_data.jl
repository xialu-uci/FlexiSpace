# first let's simulate data with no noise
using FlexiBasicLearning
using JLD2

function sim_data(num_points, dofs; std = 0.02, shape = crooked_flexi, save_name = nothing)

    # helper true flexi params
    true_params = shape(dofs)
    # num_points must be in
    x = LinRange(0.0,1.0, num_points)
    y = FlexiFunctions.evaluate_decompress.(x, Ref(true_params))
    # add noise to y
    noise = randn(num_points).*std # this makes it possible for vals outside [0,1]. Should I clamp?
    data = [x y]
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

# test: passed
# small_data = sim_data(10, 5)

NUM_POINTS = 100
DOFS = 5

# crooked
sim_data(NUM_POINTS, DOFS; save_name = "sim_data_crooked_5seg.jld2")
