# first let's simulate data with no noise (from Jun)
using FlexiBasicLearning
using JLD2
using ComponentArrays
# using OrdinaryDiffEqCore
using OrdinaryDiffEq  
# using SciMLBase
#TODO: execute outside of src

function sim_data(num_points, dofs; std = 0.05, a = 1.0, func_form = make_flexi1_func, shape = id_flexi, ode = false, save_name = nothing)

    # helper true flexi params
    true_params = shape(dofs; a = a)
    # true_func = func_form(dofs; shape = shape)

    x_max, true_func, flexi_arg_func = func_form(true_params; for_sim = true)
    # num_points must be in
    x = LinRange(0.0,x_max, num_points)
   
    y = true_func.(x)
    flexi_args = flexi_arg_func.(x)
    Y = stack(y, dims =1)
    # println(y)
    # add noise to y
    # skip noise comp if std = 0.0
    if std == 0.0
        noise = 0.0
    else
        noise = randn(size(Y)).*std # this makes it possible for vals outside [0,1]. Should I clamp?
    end
    
    y_noisy = Y .+ noise
    data = [x y_noisy]
    if !isnothing(save_name)
        savedir = joinpath(FlexiBasicLearning.asset_dir, "data")
        full_path = joinpath(savedir, save_name)
        mkpath(dirname(full_path))
        @save full_path data func_form true_params flexi_args
        println("Saved $save_name to $savedir")
    end
    return data
    
end
# using FlexiBasicLearning

# params = FlexiBasicLearning.FlexiFunctions.generate_flexi_ig(5)
# println(params)

# flexi shapes
function crooked_flexi(dofs; a =1.0)
    flex1_params = zeros(dofs)
    # make all odd indices 1
    flex1_params[1:2:end] .= 1.0
    # make it unit
    flex1_params = flex1_params / LinearAlgebra.norm(flex1_params)
    params = ComponentArray(
        p_classical  = ComponentArray(a = a),
        flex1_params = flex1_params
    )
    return params
end

function cu_flexi(dofs; a = 1.0)
    flex1_params = collect(1:dofs)
    flex1_params = flex1_params / LinearAlgebra.norm(flex1_params)
    params = ComponentArray(
        p_classical  = ComponentArray(a = a),
        flex1_params = flex1_params
    )
    return params
end

function cd_flexi(dofs; a = 1.0)
    flex1_params = collect(dofs:-1:1)   # explicit descending step
    flex1_params = flex1_params / LinearAlgebra.norm(flex1_params)
    params = ComponentArray(
        p_classical  = ComponentArray(a = a),
        flex1_params = flex1_params
    )
end

function id_flexi(dofs; a = 1.0)
    flex1_params = FlexiBasicLearning.FlexiFunctions.generate_flexi_ig(dofs)
    params = ComponentArray(
        p_classical  = ComponentArray(a = a),
        flex1_params = flex1_params
    )
end    

# --- new shape: outputs a mixed ComponentArray(p_classical=(a=...), flex1_params=...) ---
# function mixed_id_flexi(dofs; a = 1.0)
#     return ComponentArray(
#         p_classical  = ComponentArray(a = a),
#         flex1_params = id_flexi(dofs),
#     )
# end

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
        return 1.0, x -> FlexiFunctions.evaluate_decompress(x, params.flex1_params), x -> x
    else
        return x -> FlexiFunctions.evaluate_decompress(x, params.flex1_params)
    end
end


function make_flexi1_alg1_func(params; for_sim = false)
    if for_sim
        return 1.0, x -> x .* FlexiFunctions.evaluate_decompress(x, params.flex1_params), x -> x
    else
        return x -> x .* FlexiFunctions.evaluate_decompress(x, params.flex1_params)
    end
end



function make_flexi1_ode1_func(params; alg = Tsit5(), reltol = 1e-8, abstol = 1e-8,
                                x_max = 1e4, for_sim = false)
    f = y -> FlexiFunctions.evaluate_decompress(y, params.flex1_params)  # dy/dx = f(y)
    dydx(y, p, x) = f(y)

    # stop integration if y exits [0, 1]
    condition(y, x, integrator) = (y - 1.0) * y  # zero when y == 0 or y == 1
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)

    prob = ODEProblem(dydx, 0.1, (0.0, x_max))
    sol = solve(prob, alg; reltol = reltol, abstol = abstol, callback = cb)
    if for_sim
        return sol.t[end], x -> sol(x), x -> sol(x) # y is the flexi arg
    else
        return x -> sol(x)
    end
end

function make_flexi1_lv_func(params; alg = Tsit5(), reltol = 1e-8, abstol = 1e-8,
                                x_max = 1e2, for_sim = false, a = 1.0)
    # params can be either:
    #   - a bare flexi-params vector/ComponentArray (old style; `a` comes from the kwarg)
    #   - a mixed ComponentArray(p_classical=(a=...,), flex1_params=...), e.g. from
    #     mixed_id_flexi -- `a` is read out of params.p_classical.a, kwarg ignored
    # if params isa ComponentArray && haskey(params, :p_classical)
    a = params.p_classical.a
    flex_params = params.flex1_params
    # else
    #     flex_params = params
    # end

    f = z -> FlexiFunctions.evaluate_decompress(z, flex_params)  # expects arg in [0,1)

    function dydx(y, p, x)
        y1, y2 = y
        y1_mod = y1 / (y1 + 1.0) # test flexi_arg range
        dy1 = (y1 + 1.0) * f(y1_mod) - y1 * y2 # test flexi_arg range
        dy2 = -a * y2 + y1 * y2
        return [dy1, dy2]
    end

    prob = ODEProblem(dydx, [1.0, 2.5], (0.0, x_max))
    sol = solve(prob, alg; reltol = reltol, abstol = abstol)

    if for_sim
        return sol.t[end], x -> sol(x), x -> sol(x)[1]/(sol(x)[1]+1.0)
    else
        return x -> sol(x)
    end
end
    



# naming convention
# --- naming helpers ---
function func_name(f)
    f === make_flexi1_func      && return "flexi1"
    f === make_flexi1_alg1_func && return "flexi1alg1"
    f === make_flexi1_ode1_func && return "flexi1ode1"
    f === make_flexi1_lv_func && return "flexi1lv2"
    error("Unknown func_form: $f")
end

# --- plotting label helper ---
function func_form_labels(f)
    if f === make_flexi1_func
        return (xlabel = "x", y_labels = ["y"], flexi_x_label = "x",
                title = "Flexi1 Fit:  y = flexi(x)",
                flexi_title = "Flexi Function (of x)")
    elseif f === make_flexi1_alg1_func
        return (xlabel = "x", y_labels = ["y"], flexi_x_label = "x",
                title = "Flexi1-Alg1 Fit:  y = x · flexi(x)", 
                flexi_title = "Flexi Function (of x)")
    elseif f === make_flexi1_ode1_func
        return (xlabel = "t", y_labels = ["y"], flexi_x_label = "y",
                title = "Flexi1-ODE1 Fit:  dy/dx = flexi(y)",
                flexi_title = "Flexi Function (of y)")
    elseif f === make_flexi1_lv_func
        return (xlabel = "t", y_labels = ["y1 (prey)", "y2 (predator)"], flexi_x_label = "y1/(y1+1)",
                title = "Flexi1-LV Fit",
                flexi_title = "Flexi Function (of y1/(y1+1))")
    else
        error("Unknown func_form: $f")
    end
end

function shape_name(s)
    s === crooked_flexi && return "crooked"
    s === cu_flexi       && return "cu"
    s === cd_flexi       && return "cd"
    s === id_flexi  && return "id"
    error("Unknown shape: $s")
end



# func-#dof-#obs/sim_data_shape

# comment out so it doesn't resave data every instantiation.

# num_points = 20

# dofs = [5, 20, 50]
# shapes = [crooked_flexi, cu_flexi, cd_flexi]
# funcs = [make_flexi1_func, make_flexi1_alg1_func]
# num_points = [40]
# num_points = [2, 4, 8, 16, 32, 64, 128, 254, 512, 1024, 2048, 4096, 8182, 16364]
# dofs = [4]
# # dofs = [2, 4, 8, 16, 32, 64, 128, 254, 512, 1024, 2048, 4096, 8182, 16364]
# # num_points = [32]
# keys = ["crooked", "cu", "cd"]
# # # funcs = [make_flexi1_ode1_func]
# # funcs = [make_flexi1_func, make_flexi1_alg1_func, make_flexi1_ode1_func]

# a = 1.0
# funcs = [FlexiBasicLearning.make_flexi1_func, FlexiBasicLearning. make_flexi1_alg1_func, FlexiBasicLearning.make_flexi1_ode1_func, FlexiBasicLearning.make_flexi1_lv_func]
# # # num_points = [4, 8, 16, 32, 64, 128, 254, 512]
# # # dofs = [4, 8, 16, 32, 64, 128, 254, 512]
# num_points = [1024,2048]
# dofs = [4]
# skeys = ["crooked", "cu", "cd","id"]


# for n in num_points, f in funcs, d in dofs, sname in skeys
#     fname = func_name(f)
#     s        = FlexiBasicLearning.shapes[sname]
#     save_name = joinpath("mixed_true_params/no-noise/a$(a)/$(fname)-$(d)dof-$(n)obs", "sim_data_$(sname).jld2")
#     sim_data(n, d; std = 0.0, func_form = f, shape = s, save_name = save_name)
# end

# # sim_data(32, 4; std= 0.0, func_form = make_flexi1_lv_func, shape = mixed_id_flexi, save_name = "w_true_params_flexi_args/no-noise/flexi1lv2-4dof-32obs/sim_data_mixed_id.jld2")