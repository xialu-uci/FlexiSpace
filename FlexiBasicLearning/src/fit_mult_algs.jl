using JLD2
using FlexiBasicLearning
using CairoMakie
using LinearAlgebra
using Random
#load crooked sim data

# datafile = "../FlexiSpaceLocal/data/sim_data_cu_5seg.jld2"

#080502026 edits:

    #(1) saving datafile to results dict, plotting functions now load datafile from results dict, instead of passing it in as an argument
    #(2) helper functions to set_up_prob(), fit_cmaes(), fit_gd() to make it easier to call separately.
    #(3) if time_grads = true, skip cmaes and only run gd, since cmaes is not needed for timing grads

function ig_fit_all_algs(datafile, savedir, make_model; ig = nothing, optimizers = [:gradient_descent], maxiters = 10000, save_parameters = false, time_grads = false)
    @load datafile data
    
    
    my_prob, my_model = FlexiBasicLearning.set_up_prob(data, make_model)
    if isnothing(ig)
        ig = deepcopy(my_model.params)
    end
    
    # save to savedir
    mkpath(savedir) # creates the directory only if it doesn't already exist
    
    result = Dict(
    "datafile" => datafile,
    "save_dir" => savedir,
    "my_model" => my_model,
    "ig" => ig
    )

    if !time_grads
        # skip cmaes if timing grads
        cmaes_result = FlexiBasicLearning.cmaes_learn(my_prob, ig)
        result["cmaes_result"] = cmaes_result # time in cmaes_result
        # save cmaes results
        # @save joinpath(savedir, "cmaes_result.jld2") cmaes_result # maybe no save

    end
    # cmaes_time, cmaes_result = fit_cmaes(my_prob, ig)
    # gd_time, gd_result = fit_gd(my_prob, ig; optimizers=optimizers, maxiters=maxiters, save_parameters=save_parameters, time_grads=time_grads)
    for optimizer in optimizers
        println("Using optimizer: $optimizer")
        gd_result = FlexiBasicLearning.gradient_descent_learn(my_prob, ig; optimizer=optimizer, maxiters=maxiters, save_parameters = save_parameters, time_grads = time_grads)
        result["gd_$(optimizer)_result"] = gd_result # time in gd_result
    end
    
    result["optimizers"] = optimizers 

    return result
end

# function set_up_prob(data, make_model)
#     # @load datafile data
#     num_points = size(data)[1]
#     full = Vector{Bool}(trues(num_points))
#     my_model = make_model()
#     my_prob = LearningProblem(
#         data = data,
#         model = my_model,
#         mask = full
#     )
#     return my_prob, my_model
# end



function ig_make_fitting_figs(result)
    
    datafile = result["datafile"]
    savedir = result["save_dir"] # for saving

    # general
    my_model = result["my_model"]
    cmaes_result   = result["cmaes_result"]
    optimizers = result["optimizers"]
    gd_results = [result["gd_$(optimizer)_result"] for optimizer in result["optimizers"]]
    

    
    # for fit overlay
    @load datafile data
    @load datafile func_form
    @load datafile true_params
    x_data = data[:, 1]
    y_data = data[:, 2]
    x_grid = collect(LinRange(0.0, maximum(x_data), 500))
    x_grid_flexi = collect(LinRange(0.0, 1.0, 500))
    cmaes_fit_params = cmaes_result.fit_params
    gd_fit_params_list = [gd_result.fit_params for gd_result in gd_results]
    true_func = func_form(true_params)
    
    y_true  = true_func.(x_grid)
    y_cmaes = fw(x_grid, cmaes_fit_params, my_model)
    # println("y_cmaes:  $(size(y_cmaes))")
    y_gd_list = [fw(x_grid, gd_fit_params, my_model) for gd_fit_params in gd_fit_params_list]
    y_pred_list = vcat([y_cmaes], y_gd_list)
    labels_fits = vcat(["cmaes fit"],["gd fit ($optimizer)" for optimizer in optimizers])

    fig1 = make_fit_overlay_fig(x_data, y_data, x_grid, y_true, y_pred_list; title = "Fit Comparison", labels = labels_fits)

    # for flexifunction only overlay
    flexi_true = FlexiBasicLearning.FlexiFunctions.evaluate_decompress.(x_grid_flexi, Ref(true_params))
    flexi_cmaes = FlexiBasicLearning.FlexiFunctions.evaluate_decompress.(x_grid_flexi, Ref(cmaes_fit_params))
    flexi_gd_list = [FlexiBasicLearning.FlexiFunctions.evaluate_decompress.(x_grid_flexi, Ref(gd_fit_params)) for gd_fit_params in gd_fit_params_list]
    flexi_pred_list = vcat([flexi_cmaes],flexi_gd_list)
    labels_flexi = vcat(["cmaes flexi"], ["gd flexi ($optimizer)" for optimizer in optimizers])
    fig2 = make_fit_overlay_fig([0.0], [0.0], x_grid_flexi, flexi_true, flexi_pred_list; title = "Flexifunction only comparison", labels = labels_flexi) # don't plot datapoints


    # for loss history
    cmaes_loss_history = cmaes_result.loss_history
    cmaes_time = cmaes_result.time
    gd_loss_histories = [gd_result.loss_history for gd_result in gd_results]
    gd_times = [gd_result.time for gd_result in gd_results]

    all_loss_histories = vcat([cmaes_loss_history], gd_loss_histories)
    all_times = vcat([cmaes_time],gd_times)
    labels_loss = vcat(["cmaes loss"],["gd loss ($optimizer)" for optimizer in optimizers])

    fig3 = make_loss_history_figs(all_loss_histories, all_times, labels_loss)

    # save figs 
    save(joinpath(savedir, "fit_overlay.png"), fig1)
    save(joinpath(savedir, "flexi_overlay.png"), fig2)
    save(joinpath(savedir, "loss_history.png"), fig3)

    return fig1, fig2, fig3
end

function make_fit_overlay_fig(x, y_data, x_grid, y_true, y_pred_list; title = "Fit Comparison", labels = ["cmaes fit", "gd fit"])
    fig = Figure(size = (800, 500))
    ax = CairoMakie.Axis(fig[1, 1], xlabel = "t", ylabel = "y", title = title)
    CairoMakie.lines!(ax, x_grid, y_true,  label = "true", linewidth = 2, color = :black, linestyle = :dash)

    n = length(y_pred_list)
    colors = n == 1 ? [:red] : cgrad(:tab10, n, categorical = true)

    for (y_pred, label, color) in zip(y_pred_list, labels, colors)
        #println("y_pred:  $(size(y_pred))")
        #println("x_grid:  $(size(x_grid))")


        CairoMakie.lines!(ax, x_grid, y_pred, label = label, linewidth = 2, color = color)
    end
    CairoMakie.scatter!(ax, x, y_data, label = "noisy data", markersize = 5, color = (:orange))
    axislegend(ax, position = :rb)
    return fig
end


function make_loss_history_figs(all_loss_histories, all_times, labels)
    nx = 2
    n = length(all_loss_histories)
    ny = n ÷ nx + (n % nx > 0 ? 1 : 0)
    fig = CairoMakie.Figure(size = (900, 400 * ny))

    colors = n == 1 ? [:red] : CairoMakie.cgrad(:tab10, n, categorical = true)

    first_ax = nothing
    for (i, (loss_history, time, label, color)) in enumerate(zip(all_loss_histories, all_times, labels, colors))
        row = (i - 1) ÷ nx + 1
        col = (i - 1) % nx + 1
        ax = CairoMakie.Axis(fig[row, col], xlabel = "iteration", ylabel = "loss", title = "$label ($time s)", yscale = log10)
        CairoMakie.lines!(ax, loss_history, color = color, linewidth = 2)

        if i == 1
            first_ax = ax
        else
            CairoMakie.linkyaxes!(first_ax, ax)
        end
    end
    return fig
end
   

function make_fitting_figs(results)
    for result in results
        ig_make_fitting_figs(result)
    end
end


#take igs as a list of initial guesses, and return a list of results for each ig
function fit_all_algs(datafile, savedir, make_model; igs = [nothing], optimizers = [:gradient_descent], maxiters = 10000, save_parameters = false, time_grads = false)
    results = []
    for (idx, ig) in enumerate(igs)
        println("Fitting with initial guess $idx")
        subdir = joinpath(savedir, "fit_ig$(idx)")
        result = ig_fit_all_algs(datafile, subdir, make_model; ig = ig, optimizers = optimizers, maxiters = maxiters, save_parameters = save_parameters, time_grads = time_grads)
        push!(results, result)
    end
   
    @save joinpath(savedir, "results_all_ig.jld2") results
    return results  # always return the list of results for type consistency
    # end
end
   
# test 1: not entering ig should use default ig from model
# test 2: entering 1 ig should use that ig
# test 3: entering multiple igs should run multiple fits and return a list of results
