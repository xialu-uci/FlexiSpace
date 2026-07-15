using JLD2
using FlexiBasicLearning
using CairoMakie
using LinearAlgebra
using Random
#load crooked sim data

datafile = "../FlexiSpaceLocal/data/sim_data_cu_5seg.jld2"
#

function ig_fit_cmaes_and_gd(datafile, savedir, make_model; ig = nothing)
    @load datafile data
    num_points = size(data)[1]
    full = Vector{Bool}(trues(num_points))
    my_model = make_model()
    my_prob = LearningProblem(
        data = data,
        model = my_model,
        mask = full
    )

    if isnothing(ig)
        ig = deepcopy(my_model.params)
    end
    # ig = deepcopy(my_model.params)
    
    # check sensitivity 
    # base_loss = get_loss(ig; learning_problem=my_prob)
    # for sigma in [0.01, 0.05, 0.1, 0.14]
    #     for trial in 1:5
    #         direction = randn(length(ig))
    #         perturbed = ig .+ sigma .* direction
    #         Δ = get_loss(perturbed; learning_problem=my_prob) - base_loss
    #         println("sigma=$sigma, trial=$trial: Δloss = $Δ")
    #     end
    # end
    
    cmaes_fit_params, cmaes_loss_history = FlexiBasicLearning.cmaes_learn(my_prob, ig)
    gd_fit_params, gd_loss_history, gd_grads = FlexiBasicLearning.gradient_descent_learn(my_prob, ig)
    # save to savedir
    mkpath(savedir) # creates the directory only if it doesn't already exist
    # save cmaes results
    @save joinpath(savedir, "cmaes_fit.jld2") cmaes_fit_params cmaes_loss_history
    # save gd results
    @save joinpath(savedir, "gd_fit.jld2") gd_fit_params gd_loss_history gd_grads
    println("Saved cmaes and gd results to $savedir")
    # make a result (that holds both results for easier use in plotting functions)
    result = Dict(
        "save_dir" => savedir,
        "my_model" => my_model,
        "cmaes_fit_params" => cmaes_fit_params,
        "cmaes_loss_history" => cmaes_loss_history,
        "gd_fit_params" => gd_fit_params,
        "gd_loss_history" => gd_loss_history,
        "gd_grad_norm_history" => gd_grads
    )
    return result
end





function plot_fits(x, y_data, x_grid, y_true, y_cmaes, y_gd; title = "Flexi fit comparison")
    fig = Figure(size = (800, 500))
    ax = CairoMakie.Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = title)
    CairoMakie.scatter!(ax, x, y_data, label = "noisy data", markersize = 4, color = (:gray, 0.4))
    CairoMakie.lines!(ax, x_grid, y_true,  label = "true",             linewidth = 2, color = :black, linestyle = :dash)
    CairoMakie.lines!(ax, x_grid, y_cmaes, label = "cmaes fit",        linewidth = 2, color = :red)
    CairoMakie.lines!(ax, x_grid, y_gd,    label = "grad descent fit", linewidth = 2, color = :blue)
    axislegend(ax, position = :rb)
    return fig
end

# --- helper: loss histories, side by side ---
function plot_loss(cmaes_loss_history, gd_loss_history)
    fig = Figure(size = (900, 400))
    ax_cmaes = CairoMakie.Axis(fig[1, 1], xlabel = "iteration", ylabel = "loss",
                    title = "CMA-ES loss", yscale = log10)
    lines!(ax_cmaes, cmaes_loss_history, color = :red, linewidth = 2)
    ax_gd = CairoMakie.Axis(fig[1, 2], xlabel = "iteration", ylabel = "loss",
                title = "Gradient descent loss", yscale = log10)
    lines!(ax_gd, gd_loss_history, color = :blue, linewidth = 2)
    linkyaxes!(ax_cmaes, ax_gd)
    return fig
end

# helper: plot gd grads
function plot_grads(grads)
    fig = Figure(size = (900, 400))
    ax = CairoMakie.Axis(fig[1, 2], xlabel = "iteration", ylabel = "Gradient Norms",
                title = "GD Gradient Norms During Fitting")
    lines!(ax, grads, color = :blue, linewidth = 2)
    # linkyaxes!(ax_cmaes, ax_gd)
    return fig
end

# --- combined driver: builds both plots from a fit result + raw data, and saves them ---
function ig_plot_loss_and_fits(result, datafile)
    @load datafile data
    @load datafile true_func
    x = data[:, 1]
    y_data = data[:, 2]
    x_grid = collect(LinRange(0.0, 1.0, 500))

    savedir = result["save_dir"]
    my_model = result["my_model"]
    cmaes_fit_params   = result["cmaes_fit_params"]
    cmaes_loss_history = result["cmaes_loss_history"]
    gd_fit_params      = result["gd_fit_params"]
    gd_loss_history    = result["gd_loss_history"]
    gd_grads = result["gd_grad_norm_history"]


    y_true  = true_func.(x_grid)
    y_cmaes = fw(x_grid, cmaes_fit_params, my_model)
    y_gd    = fw(x_grid, gd_fit_params, my_model)
    mkpath(savedir)

    
    title = "Flexi fit comparison (dofs=$(length(my_model.params)))"
    fig1 = plot_fits(x, y_data, x_grid, y_true, y_cmaes, y_gd; title = title)
    save(joinpath(savedir, "fit_overlay.png"), fig1)

    fig2 = plot_loss(cmaes_loss_history, gd_loss_history)
    save(joinpath(savedir, "loss_history.png"), fig2)

    fig3 = plot_grads(gd_grads)
    save(joinpath(savedir, "gd_grads.png"), fig3)



    println("Saved fit_overlay.png and loss_history.png to $savedir")

    return fig1, fig2, fig3
end

function plot_loss_and_fits(results, datafile) 
    for result in results
        ig_plot_loss_and_fits(result, datafile)
    end

end

# new function: overload fit_cmaes_and_gd to take igs as a list of initial guesses, and return a list of results for each ig
function fit_cmaes_and_gd(datafile, savedir, make_model; igs = [nothing])
    results = []
    for (idx, ig) in enumerate(igs)
        println("Fitting with initial guess $idx")
        subdir = joinpath(savedir, "fit_ig$(idx)")
        result = ig_fit_cmaes_and_gd(datafile, subdir, make_model; ig = ig)
        push!(results, result)
    end
    # if length(results) == 1
    #     return results[1]  # if only one result, return it directly
    # else
    return results  # otherwise, return the list of results
    # end
end
   
# test 1: not entering ig should use default ig from model
# test 2: entering 1 ig should use that ig
# test 3: entering multiple igs should run multiple fits and return a list of results
# function plot_flexi(flexi_params::AbstractVector{<:Real}; npoints::Int=500)
#     """
#     Generate a figure of the flexifunction defined by `flexi_params`.

#     The flexifunction is computed using `FlexiFunctions.evaluate_decompress`
#     on the interval [0,1]. A line plot is created and saved to
#     `savedir/flexi_plot.png`. The figure is returned for further use.
#     """

#     # prepare evaluation points
#     xs = range(0.0, 1.0; length=npoints)
#     ys = [FlexiFunctions.evaluate_decompress(x, flexi_params) for x in xs]

#     # create figure
#     fig = Figure(size=(800, 400))
#     ax = Makie.Axis(fig[1,1],
#                     xlabel="x",
#                     ylabel="flexi(x)",
#                     title="Flexi function")
#     lines!(ax, xs, ys, color=:blue, linewidth=2)
#     ax.xgridvisible = true
#     ax.ygridvisible = true

#     # save and report
#     # save_path = joinpath(savedir, "flexi_plot.png")
#     # save(save_path, fig)
#     # println("Flexi plot saved to: $save_path")

#     return fig
# end