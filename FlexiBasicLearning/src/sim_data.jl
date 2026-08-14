# first let's simulate data with no noise (from Jun)
using FlexiBasicLearning
using JLD2
# using OrdinaryDiffEqCore
using OrdinaryDiffEq  
# using SciMLBase


function sim_data(num_points, dofs; std = 0.05, func_form = make_flexi1_func, shape = crooked_flexi, ode = false, save_name = nothing)

    # helper true flexi params
    true_params = shape(dofs)
    # true_func = func_form(dofs; shape = shape)

    x_max, true_func = func_form(true_params; for_sim = true)
    # num_points must be in
   x = LinRange(0.0,x_max, 20)
   
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
        @save full_path data func_form true_params
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

# TODO: something looks weird about gt flexi for y' = flexi(y) with dofs = 3 for cu and cd shapes.
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
function make_flexi1_func(params; for_sim = false)
    if for_sim
        return 1.0, x -> FlexiFunctions.evaluate_decompress(x, params)
    else
        return x -> FlexiFunctions.evaluate_decompress(x, params)
    end
end


function make_flexi1_alg1_func(params; for_sim = false)
    if for_sim
        return 1.0, x -> x .* FlexiFunctions.evaluate_decompress(x, params)
    else
        return x -> x .* FlexiFunctions.evaluate_decompress(x, params)
    end
end


function make_flexi1_ode1_func(params; alg = Tsit5(), reltol = 1e-8, abstol = 1e-8,
                                x_max = 1e4, for_sim = false)
    f = y -> FlexiFunctions.evaluate_decompress(y, params)  # dy/dx = f(y)
    dydx(y, p, x) = f(y)

    # stop integration if y exits [0, 1]
    condition(y, x, integrator) = (y - 1.0) * y  # zero when y == 0 or y == 1
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)

    prob = ODEProblem(dydx, 0.1, (0.0, x_max))
    sol = solve(prob, alg; reltol = reltol, abstol = abstol, callback = cb)
    if for_sim
        return sol.t[end], x -> sol(x)
    else
        return x -> sol(x)
    end
end

# function make_flexi1_ode1_func(params; alg = Tsit5(), reltol = 1e-8, abstol = 1e-8)
#     f = y ->  FlexiFunctions.evaluate_decompress(y, params) # dy/dx = f(y)
#     dydx(y, p, x) = f(y)  # out-of-place form; y and x are scalars here

#     prob = ODEProblem(dydx, 0.1, (0.0, 1.0))  # y(0) = 0, integrate x in [0,1]
#     sol = solve(prob, alg; reltol = reltol, abstol = abstol)

#     return x -> sol(x)  # callable, returns interpolated y(x)
# end

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

# comment out so it doesn't resave data every instantiation.

# num_points = 20

# dofs = [5, 20, 50]
# shapes = [crooked_flexi, cu_flexi, cd_flexi]
# funcs = [make_flexi1_func, make_flexi1_alg1_func]
# num_points = 20
num_points = [2, 4, 8, 16, 32, 64, 128, 254, 512]
# dofs = [2, 4, 8, 16, 32, 64, 128, 254]
dofs = [4]
keys = ["crooked", "cu", "cd"]
# # funcs = [make_flexi1_ode1_func]
funcs = [make_flexi1_func, make_flexi1_alg1_func, make_flexi1_ode1_func]

for n in num_points, f in funcs, d in dofs, sname in keys
    fname = func_name(f)
    s        = FlexiBasicLearning.shapes[sname]
    save_name = joinpath("w_true_params/no-noise/$(fname)-$(d)dof-$(n)obs", "sim_data_$(sname).jld2")
    sim_data(n, d; std = 0.0, func_form = f, shape = s, save_name = save_name)
end

