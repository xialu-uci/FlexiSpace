# first let's simulate data with no noise (from Jun)
using FlexiBasicLearning
using JLD2
using OrdinaryDiffEq  


function sim_data(num_points, dofs; std = 0.05, func_form = make_flexi1_func, shape = crooked_flexi, save_name = nothing)

    # helper true flexi params
    true_params = shape(dofs)
    # true_func = func_form(dofs; shape = shape)

    true_func = func_form(true_params)
    # num_points must be in
    x = LinRange(0.0,1.0, num_points)
    y = true_func.(x)
    # add noise to y
    # skip noise comp if std = 0.0
    if std == 0.0
        noise =0.0
    else
        noise = randn(num_points).*std # this makes it possible for vals outside [0,1]. Should I clamp?
    end
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

# # test: passed
# function make_flexi1_func(dofs; shape = crooked_flexi)
#     params = shape(dofs)
#     return x -> FlexiFunctions.evaluate_decompress(x, params)
# end


# function make_flexi1_alg1_func(dofs; shape = crooked_flexi)
#     params = shape(dofs)
#     return x -> x .* FlexiFunctions.evaluate_decompress(x, params)
# end

# try this instead? useful for other stuff
function make_flexi1_func(params)
    return x -> FlexiFunctions.evaluate_decompress(x, params)
end


function make_flexi1_alg1_func(params)
    return x -> x .* FlexiFunctions.evaluate_decompress(x, params)
end


# function make_flexi1_ode1_func(params; alg = Tsit5(), reltol = 1e-8, abstol = 1e-8,
#                                 x_max = 1e4)
#     f = y -> FlexiFunctions.evaluate_decompress(y, params)  # dy/dx = f(y)
#     dydx(y, p, x) = f(y)

#     # stop integration if y exits [0, 1]
#     condition(y, x, integrator) = (y - 1.0) * y  # zero when y == 0 or y == 1
#     affect!(integrator) = terminate!(integrator)
#     cb = ContinuousCallback(condition, affect!)

#     prob = ODEProblem(dydx, 0.0, (0.0, x_max))
#     sol = solve(prob, alg; reltol = reltol, abstol = abstol, callback = cb)

#     return sol
# end

function make_flexi1_ode1_func(params; alg = Tsit5(), reltol = 1e-8, abstol = 1e-8)
    f = y ->  FlexiFunctions.evaluate_decompress(y, params) # dy/dx = f(y)
    dydx(y, p, x) = f(y)  # out-of-place form; y and x are scalars here

    prob = ODEProblem(dydx, 0.1, (0.0, 1.0))  # y(0) = 0, integrate x in [0,1]
    sol = solve(prob, alg; reltol = reltol, abstol = abstol)

    return x -> sol(x)  # callable, returns interpolated y(x)
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
    f === make_flexi1_ode1_func && return "flexi1ode1"
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
# dofs = [5, 20, 50]
# shapes = [crooked_flexi, cu_flexi, cd_flexi]
# funcs = [make_flexi1_func, make_flexi1_alg1_func]
# num_points = 20
dofs = [3,4,5,20,50]
shapes = [crooked_flexi, cu_flexi, cd_flexi]
funcs = [make_flexi1_ode1_func]

for f in funcs, d in dofs, s in shapes
    fname = func_name(f)
    sname = shape_name(s)
    save_name = joinpath("no-noise/$(fname)-$(d)dof-$(num_points)obs", "sim_data_$(sname).jld2")
    sim_data(num_points, d; std = 0.0, func_form = f, shape = s, save_name = save_name)
end

# sim_data(20, 2, func_form = make_flexi1_func, shape = cu_flexi, )
