using JLD2
using FlexiBasicLearning
using CairoMakie
using LinearAlgebra
#load crooked sim data

datafile = "../FlexiSpaceLocal/data/sim_data_cu_5seg.jld2"
# savedir = "../FlexiSpaceLocal/data/06302026-flexi-basic-cu"
# mkpath(savedir)  # creates the directory if it doesn't already exist

# @load datafile data
# num_points = size(data)[1]
# # no mask for now
# full = Vector{Bool}(trues(num_points))

# # set up model, LearningProblem
# my_model = FlexiBasicLearning.make_ModelFlexi1()
# my_prob = LearningProblem(
#     data = data,
#     model = my_model,
#     mask = full
# )

# # fit with cmaes
# ig = my_model.params
# cmaes_fit_params, cmaes_loss_history = FlexiBasicLearning.cmaes_learn(my_prob, ig)

# # fit with grad descent
# gd_fit_params, gd_loss_history = FlexiBasicLearning.gradient_descent_learn(my_prob,ig)

function fit_cmaes_and_gd(datafile, savedir, make_model)
    @load datafile data
    num_points = size(data)[1]
    full = Vector{Bool}(trues(num_points))
    my_model = make_model()
    my_prob = LearningProblem(
        data = data,
        model = my_model,
        mask = full
    )
    ig = my_model.params    
    cmaes_fit_params, cmaes_loss_history = FlexiBasicLearning.cmaes_learn(my_prob, ig)
    gd_fit_params, gd_loss_history = FlexiBasicLearning.gradient_descent_learn(my_prob,ig)
    # save to savedir
    mkpath(savedir) # creates the directory only if it doesn't already exist
    # save cmaes results
    @save joinpath(savedir, "cmaes_fit.jld2") cmaes_fit_params cmaes_loss_history
    # save gd results
    @save joinpath(savedir, "gd_fit.jld2") gd_fit_params gd_loss_history
    println("Saved cmaes and gd results to $savedir")
    # make a result (that holds both results for easier use in plotting functions)
    result = Dict(
        "cmaes_fit_params" => cmaes_fit_params,
        "cmaes_loss_history" => cmaes_loss_history,
        "gd_fit_params" => gd_fit_params,
        "gd_loss_history" => gd_loss_history
    )
    return result
end



# --- Plot 1: overlay of true function, data, and fits ---
#TODO: make plot_loss_and_fits(results) use helper functions plot_loss and plot_fits()

# x = data[:, 1]
# y_data = data[:, 2]
# x_grid = collect(LinRange(0.0, 1.0, 500))

# # crooked
# # true_dofs_params = let
# #     p = zeros(5)
# #     p[1:2:end] .= 1.0
# #     p / norm(p)
# # end

# # cu
# true_dofs_params = let
#     p = collect(1:5)
#     p / norm(p)
    
# end

# # cd 
# # true_dofs_params = let 
# #     p = collect(5:-1:1)
# #     p / norm(p)  
# # end

# y_true  = FlexiBasicLearning.FlexiFunctions.evaluate_decompress.(x_grid, Ref(true_dofs_params))
# y_cmaes = FlexiBasicLearning.FlexiFunctions.evaluate_decompress.(x_grid, Ref(cmaes_fit_params))
# y_gd    = FlexiBasicLearning.FlexiFunctions.evaluate_decompress.(x_grid, Ref(gd_fit_params))

# fig1 = Figure(size = (800, 500))
# ax1 = Axis(fig1[1, 1], xlabel = "x", ylabel = "y",
#            title = "Flexi fit comparison (cu, 5 segments)")

# CairoMakie.scatter!(ax1, x, y_data, label = "noisy data", markersize = 4, color = (:gray, 0.4))
# CairoMakie.lines!(ax1, x_grid, y_true,  label = "true",            linewidth = 2, color = :black, linestyle = :dash)
# CairoMakie.lines!(ax1, x_grid, y_cmaes, label = "cmaes fit",        linewidth = 2, color = :red)
# CairoMakie.lines!(ax1, x_grid, y_gd,    label = "grad descent fit", linewidth = 2, color = :blue)
# axislegend(ax1, position = :rb)
# save(joinpath(savedir, "cu_fit_overlay.png"), fig1)

# # --- Plot 2: loss histories, side by side ---

# fig2 = Figure(size = (900, 400))

# ax_cmaes = Axis(fig2[1, 1], xlabel = "iteration", ylabel = "loss",
#                 title = "CMA-ES loss", yscale = log10)
# lines!(ax_cmaes, cmaes_loss_history, color = :red, linewidth = 2)

# ax_gd = Axis(fig2[1, 2], xlabel = "iteration", ylabel = "loss",
#              title = "Gradient descent loss", yscale = log10)
# lines!(ax_gd, gd_loss_history, color = :blue, linewidth = 2)

# linkyaxes!(ax_cmaes, ax_gd)

# save(joinpath(savedir, "cu_loss_history.png"), fig2)

# println("Saved cu_fit_overlay.png and cu_loss_history.png to $savedir")

# helper plot fits

function plot_fits(x, y_data, x_grid, y_true, y_cmaes, y_gd; title = "Flexi fit comparison")
    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = title)
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
    ax_cmaes = Axis(fig[1, 1], xlabel = "iteration", ylabel = "loss",
                    title = "CMA-ES loss", yscale = log10)
    lines!(ax_cmaes, cmaes_loss_history, color = :red, linewidth = 2)
    ax_gd = Axis(fig[1, 2], xlabel = "iteration", ylabel = "loss",
                title = "Gradient descent loss", yscale = log10)
    lines!(ax_gd, gd_loss_history, color = :blue, linewidth = 2)
    linkyaxes!(ax_cmaes, ax_gd)
    return fig
end

# --- combined driver: builds both plots from a fit result + raw data, and saves them ---
function plot_loss_and_fits(result, datafile, savedir;
                             title = "Flexi fit comparison (cu, 5 segments)",
                             fname_prefix = "cu")
    @load datafile data
    @load datafile true_params
    x = data[:, 1]
    y_data = data[:, 2]
    x_grid = collect(LinRange(0.0, 1.0, 500))

    cmaes_fit_params   = result["cmaes_fit_params"]
    cmaes_loss_history = result["cmaes_loss_history"]
    gd_fit_params      = result["gd_fit_params"]
    gd_loss_history    = result["gd_loss_history"]

    y_true  = FlexiBasicLearning.FlexiFunctions.evaluate_decompress.(x_grid, Ref(true_params))
    y_cmaes = FlexiBasicLearning.FlexiFunctions.evaluate_decompress.(x_grid, Ref(cmaes_fit_params))
    y_gd    = FlexiBasicLearning.FlexiFunctions.evaluate_decompress.(x_grid, Ref(gd_fit_params))

    mkpath(savedir)

    fig1 = plot_fits(x, y_data, x_grid, y_true, y_cmaes, y_gd; title = title)
    save(joinpath(savedir, "$(fname_prefix)_fit_overlay.png"), fig1)

    fig2 = plot_loss(cmaes_loss_history, gd_loss_history)
    save(joinpath(savedir, "$(fname_prefix)_loss_history.png"), fig2)

    println("Saved $(fname_prefix)_fit_overlay.png and $(fname_prefix)_loss_history.png to $savedir")

    return fig1, fig2
end

# --- example usage ---
# savedir = "../FlexiSpaceLocal/data/06302026-flexi-basic-cu"
# my_model = FlexiBasicLearning.make_ModelFlexi1
# result = fit_cmaes_and_gd(datafile, savedir, my_model)
# @load datafile data   # reload so `data` is in scope for plotting
# true_dofs_params = let
#     p = collect(1:5)
#     p / norm(p)
# end
# plot_loss_and_fits(result, data, true_dofs_params, savedir; title = "Flexi fit comparison (cu, 5 segments)", fname_prefix = "cu")