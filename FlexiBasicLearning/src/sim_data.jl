# first let's simulate data with no noise
using FlexiBasicLearning
using JLD2

function sim_data(num_points, dofs; std = 0.05, func_form = make_flexi_func, shape = crooked_flexi, save_name = nothing)

    # helper true flexi params
    #true_params = shape(dofs)
    true_func = func_form(dofs; shape = shape)
    # num_points must be in
    x = LinRange(0.0,1.0, num_points)
    y = true_func.(x)
    # add noise to y
    noise = randn(num_points).*std # this makes it possible for vals outside [0,1]. Should I clamp?
    y_noisy = y .+ noise
    data = [x y_noisy]
    if !isnothing(save_name)
        savedir = "../FlexiSpaceLocal/data"
        full_path = joinpath(savedir, save_name)
        mkpath(dirname(full_path))
        @save full_path data true_func
        println("Saved $save_name to $savedir")
    end
    return data
    
end
# using FlexiBasicLearning

# params = FlexiBasicLearning.FlexiFunctions.generate_flexi_ig(5)
# println(params)

# flexi shapes
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
function make_flexi1_func(dofs; shape = crooked_flexi)
    params = shape(dofs)
    return x -> FlexiFunctions.evaluate_decompress(x, params)
end

function make_flexi1_alg1_func(dofs; shape = crooked_flexi)
    params = shape(dofs)
    return x -> x .* FlexiFunctions.evaluate_decompress(x, params)
end
# small_data = sim_data(10, 5)

# NUM_POINTS = 100
# DOFS = 5

# crooked
# sim_data(NUM_POINTS, DOFS; save_name = "sim_data_crooked_5seg.jld2")

# cu 
# sim_data(NUM_POINTS, DOFS; shape = cu_flexi, save_name = "sim_data_cu_5seg.jld2")

# # cd 
# sim_data(NUM_POINTS, DOFS; shape = cd_flexi, save_name = "sim_data_cd_5seg.jld2")

# naming convention
# --- naming helpers ---
function func_name(f)
    f === make_flexi1_func      && return "flexi1"
    f === make_flexi1_alg1_func && return "flexi1alg1"
    error("Unknown func_form: $f")
end

function shape_name(s)
    s === crooked_flexi && return "crooked"
    s === cu_flexi       && return "cu"
    s === cd_flexi       && return "cd"
    error("Unknown shape: $s")
end

# func-#dof-#obs/sim_data_shape


num_points = 20
dofs = [5, 20, 50]
shapes = [crooked_flexi, cu_flexi, cd_flexi]
funcs = [make_flexi1_func, make_flexi1_alg1_func]

for f in funcs, d in dofs, s in shapes
    fname = func_name(f)
    sname = shape_name(s)
    save_name = joinpath("$(fname)-$(d)dof-$(num_points)obs", "sim_data_$(sname).jld2")
    sim_data(num_points, d; func_form = f, shape = s, save_name = save_name)
end